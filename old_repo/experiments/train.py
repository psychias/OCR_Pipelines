
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

import argparse
import random
import gc
import shutil
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import mine_hard_negatives
from datasets import Dataset as HFDataset


training_model_path = "./trained_models/"
os.makedirs(training_model_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"TRAIN on: {device}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_random_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def tag(text: str, model_name: str, kind: str):
    s = "" if text is None else str(text)
    if "multilingual-e5" in model_name or "e5" in model_name.lower():
        return ("query: " if kind == "query" else "passage: ") + s
    return s

def random_window(text: str, max_words: int = 360, rng: random.Random = None) -> str:
    if text is None: return ""
    words = str(text).split(); n = len(words)
    if n <= max_words: return " ".join(words)
    rng = rng or random
    start = rng.randint(0, n - max_words)
    return " ".join(words[start:start + max_words])

class ChunkedPermutationSampler(Sampler):
    def __init__(self, total_len: int, chunk_size: int = 8, base_seed: int = 0):
        assert total_len % chunk_size == 0, "total_len must be divisible by chunk_size"
        self.total_len = total_len; self.chunk_size = chunk_size
        self.num_chunks = total_len // chunk_size; self.base_seed = base_seed
    def __iter__(self):
        rng = random.Random(self.base_seed)
        order = list(range(self.num_chunks)); rng.shuffle(order)
        for cid in order:
            start = cid * self.chunk_size
            for i in range(self.chunk_size):
                yield start + i
    def __len__(self): return self.num_chunks * self.chunk_size

def _need_cols(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise ValueError(f"{name} missing columns: {miss}")

def _all_qq_from(df, pairs, seed):
    out = []
    if df is None or len(df) == 0: return out
    for _, row in df.sample(frac=1.0, random_state=seed).iterrows():
        for (c1, c2) in pairs:
            a, b = row.get(c1), row.get(c2)
            if a is None or b is None: continue
            sa, sb = str(a).strip(), str(b).strip()
            if not sa or not sb or sa == sb: continue
            out.append(InputExample(texts=[sa, sb], label=1.0))  # tag later
    return out

def _build_qq_pool(mono_df, mono_bl_df, mono_snp_df, cross_df, exp, seed, qq_target_total, rng):
    exp = exp.strip().lower()
    cand = []
    if exp in ("mono", "mono_batches"):
        _need_cols(mono_df, ["german","german_noise_random05","french","french_noise_random05"], "mono_df")
        cand += _all_qq_from(mono_df, [("german","german_noise_random05"),
                                       ("french","french_noise_random05")], seed)
    elif exp == "mono_bl_real":
        _need_cols(mono_bl_df, ["german","german_noise_BLDS","french","french_noise_BLDS"], "mono_bl_df")
        cand += _all_qq_from(mono_bl_df, [("german","german_noise_BLDS"),
                                          ("french","french_noise_BLDS")], seed)
    elif exp == "mono_snp_real":
        _need_cols(mono_snp_df, ["german","german_noise_SNP","french","french_noise_SNP"], "mono_snp_df")
        cand += _all_qq_from(mono_snp_df, [("german","german_noise_SNP"),
                                           ("french","french_noise_SNP")], seed)
    elif exp == "x_mono":
        _need_cols(cross_df, ["german","german_noise_random05","french","french_noise_random05"], "cross_df")
        cand += _all_qq_from(cross_df, [("german","german_noise_random05"),
                                        ("french","french_noise_random05")], seed)
    elif exp == "cross":
        _need_cols(cross_df, ["german","french_noise_random05","french","german_noise_random05"], "cross_df")
        cand += _all_qq_from(cross_df, [("german","french_noise_random05"),
                                        ("french","german_noise_random05")], seed)
    elif exp == "cross_clean":
        _need_cols(cross_df, ["german","french"], "cross_df")
        cand += _all_qq_from(cross_df, [("german","french"), ("french","german")], seed)
    elif exp == "mono+cross":
        _need_cols(mono_df,  ["german","german_noise_random05","french","french_noise_random05"], "mono_df")
        _need_cols(cross_df, ["german","french_noise_random05","french","german_noise_random05"], "cross_df")
        cand += _all_qq_from(mono_df,  [("german","german_noise_random05"),
                                        ("french","french_noise_random05")], seed)
        cand += _all_qq_from(cross_df, [("german","french_noise_random05"),
                                        ("french","german_noise_random05")], seed)
    else:
        raise ValueError(f"Unknown qq_experiment='{exp}'")
    rng.shuffle(cand)
    if len(cand) >= qq_target_total: return cand[:qq_target_total]
    need = qq_target_total - len(cand)
    if len(cand) == 0: raise ValueError("No valid q→q pairs found; check paper CSV columns.")
    return cand + rng.choices(cand, k=need)


def mine_semantic_negs_for_qd(
    model: SentenceTransformer,
    qd_items: List[Dict],
    corpus_texts: List[str],
    relative_margin: float = 0.05,
    range_min: int = 10,
    range_max: int = 200,
    num_negatives: int = 5,
    mining_batch_size: int = 64,
    verbose: bool = True,
) -> Dict[Tuple[str, str], List[str]]:
    """
    Returns mapping: (query_text, answer_text) -> [neg_text, ...]
    Uses sentence_transformers.util.mine_hard_negatives on CPU with use_faiss=False.
    """
    q_list = [d["query_text"] for d in qd_items]
    a_list = [d["pos_text"]   for d in qd_items]

    hf_ds = HFDataset.from_dict({"query": q_list, "answer": a_list})

    mined = mine_hard_negatives(
        dataset=hf_ds,
        model=model,
        anchor_column_name="query",
        positive_column_name="answer",
        corpus=corpus_texts,
        relative_margin=relative_margin,
        range_min=range_min,
        range_max=range_max,
        num_negatives=num_negatives,
        sampling_strategy="top",
        use_faiss=False,
        output_format="n-tuple",
        verbose=verbose,
        batch_size=8
    )

    neg_cols = [c for c in mined.column_names if c.startswith("negative")]
    mapping = {}
    for row in mined:
        key = (row["query"], row["answer"])
        mapping[key] = [row[c] for c in neg_cols if row.get(c)]
    return mapping

def build_doc_mix_samples_with_semantic_hardnegs(
    model_name: str,
    _sample_size_ignored: int,
    seed: int,
    qq_target_total: int = 20000,
    qd_target_total: int = 20000,
    dd_target_total: int = 20000,
    max_words_window: int = 512,
):
    assert args.batch_size == 16, "This packer assumes batch_size=16 for (8,4,4) mixing"
    rng = random.Random(seed)

    qq_pool = _build_qq_pool(
        mono_df=mono_df, mono_bl_df=mono_bl_df, mono_snp_df=mono_snp_df,
        cross_df=cross_df, exp=args.qq_experiment,
        seed=seed, qq_target_total=qq_target_total, rng=rng
    )

    if "doc_id" not in docs_df.columns:
      docs_df["doc_id"] = np.arange(len(docs_df), dtype=int)

    dd_by_docid = {}
    corpus_texts = []
    neg_text_to_docid = {}

    qd_items = []
    for _, row in docs_df.iterrows():
        rid = int(row["doc_id"])
        d_clean = row.get("doc_text");         d_noisy = row.get("doc_text_noised")
        t_clean = row.get("translation");      t_noisy = row.get("translation_noised")

        dd_pair = None
        # if d_clean and d_noisy:
        #     dd_pair = InputExample(
        #         texts=[tag(random_window(str(d_clean), max_words_window, rng), model_name, "doc"),
        #                tag(random_window(str(d_noisy), max_words_window, rng), model_name, "doc")],
        #         label=1.0
        #     )
        #     neg_corpus_text = str(d_noisy).strip()
        # elif t_clean and t_noisy:
        #     dd_pair = InputExample(
        #         texts=[tag(random_window(str(t_clean), max_words_window, rng), model_name, "doc"),
        #                tag(random_window(str(t_noisy), max_words_window, rng), model_name, "doc")],
        #         label=1.0
        #     )
        #     neg_corpus_text = str(t_noisy).strip()
        # else:
        #     neg_corpus_text = None
        if isinstance(d_clean, str) and isinstance(d_noisy, str) and d_clean.strip() and d_noisy.strip():
          dd_pair = InputExample(
              texts=[tag(random_window(str(d_clean), max_words_window, rng), model_name, "doc"),
                    tag(random_window(str(d_noisy), max_words_window, rng), model_name, "doc")],
              label=1.0
          )
          neg_corpus_text = str(d_noisy).strip()

        if dd_pair is not None:
            dd_by_docid[rid] = dd_pair

        if neg_corpus_text:
            corpus_texts.append(neg_corpus_text)
            neg_text_to_docid.setdefault(neg_corpus_text, rid)

        q_txt = row.get("query")
        target = d_noisy if (d_noisy and str(d_noisy).strip()) else t_noisy
        if q_txt and target:
            qd_items.append({
                "ex": InputExample(
                    texts=[tag(str(q_txt), model_name, "query"),
                           tag(random_window(str(target), max_words_window, rng), model_name, "doc")],
                    label=1.0
                ),
                "query_text": str(q_txt).strip(),
                "pos_doc_id": rid,
                "pos_text": str(target).strip()
            })
    print(f"the len of qq is {len(qq_pool)}")
    print(f"the len of qd is {len(qd_items)}")
    print(f"the len of dd is {len(dd_by_docid)}")
    # raise ValueError("Insufficient samples")
    rng.shuffle(qd_items)
    if len(qd_items) < qd_target_total:
        qd_items = qd_items + rng.choices(qd_items, k=qd_target_total - len(qd_items))
    else:
        qd_items = qd_items[:qd_target_total]

    dd_pool_ids = list(dd_by_docid.keys())
    rng.shuffle(dd_pool_ids)
    if len(dd_pool_ids) < dd_target_total:
        dd_pool_ids = dd_pool_ids + rng.choices(dd_pool_ids, k=dd_target_total - len(dd_pool_ids))
    else:
        dd_pool_ids = dd_pool_ids[:dd_target_total]

    print("[mine_hard_negatives] ")
    mining_model = SentenceTransformer(args.model_name, trust_remote_code=True, device="cuda")
    mined_map = mine_semantic_negs_for_qd(
        model=mining_model,
        qd_items=qd_items,
        corpus_texts=corpus_texts,
        relative_margin=0.00,
        range_min=1,
        range_max=2000,
        num_negatives=4,
        mining_batch_size=64,
        verbose=True
    )
    del mining_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    batches = min(len(qq_pool)//8, len(qd_items)//4, len(dd_pool_ids)//4)
    qi = di = si = 0
    samples: List[InputExample] = []

    for _ in range(batches):
        batch: List[InputExample] = []

        batch += [
            InputExample(
                texts=[tag(ex.texts[0], model_name, "query"),
                       tag(ex.texts[1], model_name, "query")],
                label=1.0
            )
            for ex in qq_pool[qi:qi+8]
        ]
        qi += 8

        qd_chunk = qd_items[di:di+4]; di += 4
        for item in qd_chunk:
            batch.append(item["ex"])
        topic_to_docids = (docs_df
                   .groupby("topic")["doc_id"]
                   .apply(list)
                   .to_dict())
        used_doc_ids = set()
        for item in qd_chunk:
            key = (item["query_text"], item["pos_text"])
            neg_list = mined_map.get(key, [])
            dd_added = False
            for neg_text in neg_list:
                did = neg_text_to_docid.get(str(neg_text).strip())
                if did is None or did == item["pos_doc_id"] or did in used_doc_ids:
                    continue
                dd_candidate = dd_by_docid.get(did)
                if dd_candidate is None:
                    continue
                batch.append(dd_candidate)
                used_doc_ids.add(did)
                dd_added = True
                break

            if not dd_added:
                while si < len(dd_pool_ids) and (dd_pool_ids[si] in used_doc_ids or dd_pool_ids[si] == item["pos_doc_id"]):
                    si += 1
                did = dd_pool_ids[si] if si < len(dd_pool_ids) else random.choice(list(dd_by_docid.keys()))
                if si < len(dd_pool_ids): si += 1
                batch.append(dd_by_docid[did]); used_doc_ids.add(did)

        if len(batch) != 16:
            raise RuntimeError(f"Packed {len(batch)} != 8")
        samples.extend(batch)

    print(f"[exact-mix+semantic-hardneg] batches={len(samples)//8} | qq_used={qi} qd_used={di} dd_injected={(len(samples)//8)*2} | total={len(samples)}")
    return samples

parser = argparse.ArgumentParser(description="doc_mix_training with semantic hard negatives (no FAISS) + windowing")
parser.add_argument("--model_name", type=str, default="Alibaba-NLP/gte-multilingual-base")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--fp16", action="store_true", default=True)
parser.add_argument("--qq_experiment", type=str, default="mono")  # mono | mono_bl_real | mono_snp_real | x_mono | cross | cross_clean | mono+cross
parser.add_argument("--qd_target_total", type=int, default=40000)
parser.add_argument("--dd_target_total", type=int, default=40000)
parser.add_argument("--mono_file", type=str, default="./finetuning_data/TED_data_random_noise.csv")
parser.add_argument("--mono_bl_real_file", type=str, default="./finetuning_data/TED_data_realistic_noise.csv")
parser.add_argument("--mono_snp_real_file", type=str, default="./finetuning_data/TED_data_realistic_noise.csv")
parser.add_argument("--cross_file", type=str, default="./finetuning_data/X-News_data_random_noise.csv")
parser.add_argument("--doc_fr_de_file", type=str, default="./finetuning_data/sample_dataset_random_noise_de.csv")
parser.add_argument("--doc_de_fr_file", type=str, default="./finetuning_data/sample_dataset_random_noise_fr.csv")
args = parser.parse_args([])


mono_df        = pd.read_csv(args.mono_file)
mono_bl_df     = pd.read_csv(args.mono_bl_real_file)
mono_snp_df    = pd.read_csv(args.mono_snp_real_file)
cross_df       = pd.read_csv(args.cross_file)
doc_fr_de_df   = pd.read_csv(args.doc_fr_de_file)
doc_de_fr_df   = pd.read_csv(args.doc_de_fr_file)

docs_df = pd.concat([doc_fr_de_df, doc_de_fr_df], ignore_index=True)
docs_df = docs_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Loaded docs_df: {len(docs_df)} rows")


def train_one_seed(seed: int):
    print(f"\n===== Seed {seed} =====")
    set_random_seed(seed)

    train_samples = build_doc_mix_samples_with_semantic_hardnegs(
        model_name=args.model_name,
        _sample_size_ignored=0,
        seed=seed,
        qq_target_total=20000,
        qd_target_total=args.qd_target_total,
        dd_target_total=args.dd_target_total,
        max_words_window=512,
    )
    assert len(train_samples) % args.batch_size == 0, "Packed samples must be multiple of 8"
    print(f"Built {len(train_samples)} samples")

    model = SentenceTransformer(args.model_name, trust_remote_code=True, device=str(device))
    model.max_seq_length = 512

    first = getattr(model, "_first_module", None)
    if callable(first): first = model._first_module()
    else:
        try: first = model[0]
        except Exception: first = None
    if first is not None:
        am = getattr(first, "auto_model", None)
        if am is not None:
            if hasattr(am, "gradient_checkpointing_enable"): am.gradient_checkpointing_enable()
            if hasattr(am, "config") and hasattr(am.config, "use_cache"): am.config.use_cache = False

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    for epoch in range(args.epochs):
        sampler = ChunkedPermutationSampler(
            total_len=len(train_samples),
            chunk_size=args.batch_size,
            base_seed=seed + 9973 * epoch
        )
        train_dataloader = DataLoader(
            train_samples,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=model.smart_batching_collate,
        )
        warmup = 0 if epoch > 0 else max(10, int(0.1 * len(train_dataloader)))
        print(f"Epoch {epoch+1}/{args.epochs} | steps={len(train_dataloader)} | warmup={warmup}")

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup,
            show_progress_bar=True,
            use_amp=bool(args.fp16),
            max_grad_norm=args.max_grad_norm,
            weight_decay=0.01,
        )

    save_name = f"{args.model_name.replace('/', '_')}-doc_mix_training-no_faiss-40k-mix-seed{seed}"
    save_path = os.path.join(training_model_path, save_name)
    model.save(save_path)
    print(f"Saved: {save_path}")
    shutil.make_archive(save_path, "zip", save_path)
    return save_path

seeds = [42, 100, 123, 777, 999]
saved = []
for s in seeds:
    try:
        p = train_one_seed(s)
        saved.append(p)
        zip_path = shutil.make_archive(p, "zip", p)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM at seed {s}] Consider reducing window size or batch_size. Error: {e}")
        raise
    except Exception as e:
        print(f"[Error at seed {s}] {e}")
        raise

print("Saved model folders:")
