
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_DISABLED"] = "true"   

import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
import shutil
from google.colab import files
import gc

training_model_path = "./trained_models/"
if not os.path.exists(training_model_path):
  os.mkdir(training_model_path)

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tag(text: str, model_name: str, kind: str):
    s = "" if text is None else str(text)
    if "multilingual-e5" in model_name:
        return ("query: " if kind == "query" else "passage: ") + s
    return s

def add_pair(samples, left, right, kind_left, kind_right, model_name):
    if left is None or right is None:
        return False
    a, b = str(left).strip(), str(right).strip()
    if not a or not b:
        return False
    samples.append(
        InputExample(
            texts=[tag(a, model_name, kind_left), tag(b, model_name, kind_right)],
            label=1
        )
    )
    return True


parser = argparse.ArgumentParser(description="Colab doc_mix_training multi-seed run")
parser.add_argument("--model_name", type=str, default="Alibaba-NLP/gte-multilingual-base")
parser.add_argument("--sample_size", type=int, default=40000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--fp16", action="store_true", default=True)
parser.add_argument("--qq_experiment", type=str, default="mono")  # mono, mono_bl_real, mono_snp_real, x_mono, cross, cross_clean, mono+cross
parser.add_argument("--qd_target_total", type=int, default=20000)  # out of docs_df
parser.add_argument("--dd_target_total", type=int, default=20000)  # out of docs_df

args = parser.parse_args([])  # In Colab, ignore CLI and use defaults above

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Data paths (ensure these CSVs exist in your Colab workspace)
# ----------------------------
mono_file = "./finetuning_data/TED_data_random_noise.csv"
mono_bl_real_file = "./finetuning_data/TED_data_realistic_noise.csv"
mono_snp_real_file = "./finetuning_data/TED_data_realistic_noise.csv"
cross_file = "./finetuning_data/X-News_data_random_noise.csv"
fr_en_file = "./finetuning_data/X-News_data_random_noise.csv"
de_en_file = "./finetuning_data/X-News_data_random_noise.csv"
doc_fr_de_file = "./generate_random_noise/sample_dataset_random_noise_de.csv"
doc_de_fr_file = "./generate_random_noise/sample_dataset_random_noise_fr.csv"

# ----------------------------
# Load datasets once
# ----------------------------
mono_df = pd.read_csv(mono_file)
mono_bl_df = pd.read_csv(mono_bl_real_file)
mono_snp_df = pd.read_csv(mono_snp_real_file)
cross_df = pd.read_csv(cross_file)
fr_en_df = pd.read_csv(fr_en_file)
de_en_df = pd.read_csv(de_en_file)
doc_fr_de_df = pd.read_csv(doc_fr_de_file)
doc_de_fr_df = pd.read_csv(doc_de_fr_file)
docs_df = pd.concat([doc_fr_de_df, doc_de_fr_df], ignore_index=True)
docs_df = docs_df.sample(frac=1,random_state=42).reset_index(drop=True)


def build_doc_mix_samples(
    model_name: str,
    _sample_size_ignored: int,
    seed: int,
    qq_target_total: int = 20000,   # fixed: paper-like q→q
    qd_target_total: int = None,    # set from args
    dd_target_total: int = None,    # set from args
):
    assert args.batch_size == 8, "This packer assumes batch_size=8 for (4,2,2) mixing"
    if qd_target_total is None: qd_target_total = args.qd_target_total
    if dd_target_total is None: dd_target_total = args.dd_target_total
    rng = random.Random(seed)

    # ---------- helpers ----------
    def need_cols(df, cols, name):
        miss = [c for c in cols if c not in df.columns]
        if miss: raise ValueError(f"{name} missing columns: {miss}")

    def add_pair(pool, a, b, kind_a, kind_b):

        if a is None or b is None: return 0
        s, t = str(a).strip(), str(b).strip()
        if not s or not t or s == t: return 0
        pool.append(InputExample(texts=[tag(s, model_name, kind_a),
                                        tag(t, model_name, kind_b)], label=1))
        return 1

    # ---------- q→q from paper data (build ALL candidates, then trim/upsample to 20k) ----------
    def all_qq_from(df, pairs):
        out = []
        if df is None or len(df) == 0: return out
        for _, row in df.sample(frac=1.0, random_state=seed).iterrows():
            for (c1, c2) in pairs:
                add_pair(out, row.get(c1), row.get(c2), "query", "query")
        return out

    def build_qq_pool_from_paper(mono_df, mono_bl_df, mono_snp_df, cross_df, exp):
        exp = exp.strip().lower()
        cand = []
        if exp in ("mono", "mono_batches"):
            need_cols(mono_df, ["german","german_noise_random05","french","french_noise_random05"], "mono_df")
            cand += all_qq_from(mono_df, [("german","german_noise_random05"),
                                          ("french","french_noise_random05")])
        elif exp == "mono_bl_real":
            need_cols(mono_bl_df, ["german","german_noise_BLDS","french","french_noise_BLDS"], "mono_bl_df")
            cand += all_qq_from(mono_bl_df, [("german","german_noise_BLDS"),
                                             ("french","french_noise_BLDS")])
        elif exp == "mono_snp_real":
            need_cols(mono_snp_df, ["german","german_noise_SNP","french","french_noise_SNP"], "mono_snp_df")
            cand += all_qq_from(mono_snp_df, [("german","german_noise_SNP"),
                                              ("french","french_noise_SNP")])
        elif exp == "x_mono":
            need_cols(cross_df, ["german","german_noise_random05","french","french_noise_random05"], "cross_df")
            cand += all_qq_from(cross_df, [("german","german_noise_random05"),
                                           ("french","french_noise_random05")])
        elif exp == "cross":
            need_cols(cross_df, ["german","french_noise_random05","french","german_noise_random05"], "cross_df")
            cand += all_qq_from(cross_df, [("german","french_noise_random05"),
                                           ("french","german_noise_random05")])
        elif exp == "cross_clean":
            need_cols(cross_df, ["german","french"], "cross_df")
            cand += all_qq_from(cross_df, [("german","french"),
                                           ("french","german")])
        elif exp == "mono+cross":
            need_cols(mono_df,  ["german","german_noise_random05","french","french_noise_random05"], "mono_df")
            need_cols(cross_df, ["german","french_noise_random05","french","german_noise_random05"], "cross_df")
            cand += all_qq_from(mono_df,  [("german","german_noise_random05"),
                                           ("french","french_noise_random05")])
            cand += all_qq_from(cross_df, [("german","french_noise_random05"),
                                           ("french","german_noise_random05")])
        else:
            raise ValueError(f"Unknown qq_experiment='{exp}'")

        rng.shuffle(cand)
        if len(cand) >= qq_target_total:
            return cand[:qq_target_total]
        need = qq_target_total - len(cand)
        if len(cand) == 0:
            raise ValueError("No valid q→q pairs found; check paper CSV columns.")
        print(f"[qq] had {len(cand)} < {qq_target_total}; upsample +{need}.")
        return cand + rng.choices(cand, k=need)

    qq_pool = build_qq_pool_from_paper(
        mono_df=mono_df, mono_bl_df=mono_bl_df, mono_snp_df=mono_snp_df,
        cross_df=cross_df, exp=args.qq_experiment
    )

    # ---------- q→d and d↔d from ALL rows in docs_df ----------
    # Build candidates by iterating every row (no disjoint split).
    qd_pool, dd_pool = [], []

    for _, row in docs_df.iterrows():

        # q → d: prefer translation_noised, fallback to doc_text_noised
        add_pair(qd_pool, row.get("query"), row.get("translation_noised"), "query", "doc")
        add_pair(qd_pool, row.get("query"), row.get("doc_text_noised"), "query", "doc")

        # d ↔ d: prefer doc_text ↔ doc_text_noised, fallback to translation ↔ translation_noised
        add_pair(dd_pool, row.get("doc_text"), row.get("doc_text_noised"), "doc", "doc")
        add_pair(dd_pool, row.get("translation"), row.get("translation_noised"), "doc", "doc")

    if len(qd_pool) < qd_target_total:
        need = qd_target_total - len(qd_pool)
        if len(qd_pool) == 0:
            raise ValueError("No q→d pairs could be built; check docs_df columns/values.")
        qd_pool += rng.choices(qd_pool, k=need)
        print(f"[qd] upsample +{need} to reach {qd_target_total}.")
    if len(dd_pool) < dd_target_total:
        need = dd_target_total - len(dd_pool)
        if len(dd_pool) == 0:
            raise ValueError("No d↔d pairs could be built; check docs_df columns/values.")
        dd_pool += rng.choices(dd_pool, k=need)
        print(f"[dd] upsample +{need} to reach {dd_target_total}.")

    rng.shuffle(qq_pool); rng.shuffle(qd_pool); rng.shuffle(dd_pool)

    # ---------- exact packing to 40k: (qq, qd, dd) = (4,2,2) ----------
    batches = qq_target_total // 4  # 20000 / 4 = 5000
    # assert batches == qd_target_total // 2 == dd_target_total // 2, "Targets must align to (4,2,2)"
    qi = di = si = 0
    samples = []
    for _ in range(batches):
        batch = []
        batch += qq_pool[qi:qi+4]; qi += 6
        batch += qd_pool[di:di+2]; di += 5
        batch += dd_pool[si:si+2]; si += 5
        if len(batch) != 8: raise RuntimeError(f"Packed {len(batch)} != 8")
        samples.extend(batch)

    print(f"[exact-mix-40k] batches={len(samples)//8} | qq=20000, qd={qd_target_total}, dd={dd_target_total} | total={len(samples)}")
    return samples



def train_one_seed(seed: int):
    print(f"\n===== Seed {seed} =====")
    set_random_seed(seed)

    # Build samples for this seed
    train_samples = build_doc_mix_samples(args.model_name, args.sample_size, seed)
    print(f"Built {len(train_samples)} samples for doc_mix_training")

    # Create model
    # Note: device parameter here avoids an extra .to(device) call
    model = SentenceTransformer(args.model_name, trust_remote_code=True, device=str(device))
    model.max_seq_length = 512

    # Memory savers
    first = getattr(model, "_first_module", None)
    if callable(first):
        first = model._first_module()
    else:
        # Fallback: index access if available
        try:
            first = model[0]
        except Exception:
            first = None

    if first is not None:
        am = getattr(first, "auto_model", None)
        if am is not None:
            if hasattr(am, "gradient_checkpointing_enable"):
                am.gradient_checkpointing_enable()
            if hasattr(am, "config") and hasattr(am.config, "use_cache"):
                am.config.use_cache = False

    # Dataloader
    train_dataloader = DataLoader(
        train_samples,
        batch_size=args.batch_size,   # 8
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=model.smart_batching_collate,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup = max(10, int(0.1 * len(train_dataloader)))
    print(f"Warmup steps: {warmup} (of {len(train_dataloader)} iters)")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        show_progress_bar=True,
        use_amp=bool(args.fp16),
        max_grad_norm=args.max_grad_norm,
        weight_decay=0.01,

    )

    # Save
    save_name = f"{args.model_name.replace('/', '_')}-doc_mix_training-{args.sample_size}-samples-seed{seed}"
    save_path = os.path.join(training_model_path, save_name)
    model.save(save_path)
    print(f"Saved: {save_path}")
    return save_path

# ----------------------------
# Run all seeds
# ----------------------------
seeds = [42, 100, 123, 777, 999]
saved = []
for s in seeds:
    # p = train_one_seed(s)

    try:
        p = train_one_seed(s)
        saved.append(p)
        zip_path = shutil.make_archive(p, "zip", p)
        # files.download(zip_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM at seed {s}] Consider lowering max_seq_len or batch_size. Error: {e}")
        raise
    except Exception as e:
        print(f"[Error at seed {s}] {e}")
        raise

print("\nAll done. Saved model folders:")
for p in saved:
    print(" -", p)
