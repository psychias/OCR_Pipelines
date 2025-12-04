#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-batch OCR-robust fine-tuning (clean, plug-and-play)

- Exactly 1 batch per task per cycle, in the order you asked.
- Pools auto-adjust when a dataset path is empty or missing.
- Unique run names include data signature, enabled tasks, steps, bs, seed, timestamp.
- Prompt prefixes added for GTE/E5 models ("query:", "summary:", "document:") when enabled.
- LUX pairs are deduped and mirrored (a,b) & (b,a) to strengthen symmetry.
"""

import os, re, json, math, random, gc, shutil
from datetime import datetime
from types import SimpleNamespace
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# --------------------------
# Repro & small utilities
# --------------------------
def set_random_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def parse_seeds(spec) -> List[int]:
    if spec is None: return [42]
    if isinstance(spec, int): return [spec]
    if isinstance(spec, list): return [int(x) for x in spec]
    s = str(spec).replace(' ',''); parts = s.split(',')
    out = []
    for p in parts:
        if '-' in p:
            a,b = p.split('-',1); out.extend(range(int(a), int(b)+1))
        elif p != '':
            out.append(int(p))
    return sorted(set(out))

import hashlib

def slugify(text: str, maxlen: int = 50) -> str:
    base = re.sub(r'[^A-Za-z0-9._-]+','-', str(text)).strip('-')
    if len(base) <= maxlen:
        return base
    h = hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]
    return base[:maxlen-9] + '-' + h

def short_tag_from_path(path: str, maxlen: int = 30) -> str:
    base = os.path.basename(path)
    return slugify(base, maxlen=maxlen)

# --------------------------
# IO helpers (comment-out friendly)
# --------------------------
def optional_read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path or not str(path).strip():
        print("[info] CSV path empty -> skipping"); return None
    p = str(path)
    if not os.path.exists(p):
        print(f"[warn] CSV not found at {p} -> skipping"); return None
    try:
        df = pd.read_csv(p)
        print(f"[load] {p} -> {len(df)} rows"); return df
    except Exception as e:
        print(f"[warn] Failed to read {p}: {e} -> skipping"); return None

def optional_load_jsonl(path: Optional[str]) -> Optional[List[dict]]:
    if not path or not str(path).strip():
        print("[info] JSONL path empty -> skipping"); return None
    if not os.path.exists(path):
        print(f"[warn] JSONL not found at {path} -> skipping"); return None
    dataset = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f: dataset.append(json.loads(line))
        print(f"[load] {path} -> {len(dataset)} records"); return dataset
    except Exception as e:
        print(f"[warn] Failed to read {path}: {e} -> skipping"); return None

# --------------------------
# LUX helpers
# --------------------------
_slug_re = re.compile(r'[^A-Za-z0-9\s]+')
def clean_text_from_punctuation(text: str) -> str:
    return _slug_re.sub(' ', str(text)).strip()

def extract_parallel_sentences(lines: Optional[List[dict]], src_col: str, tgt_col: str, min_chars: int = 5) -> List[Tuple[str,str]]:
    if not lines: return []
    out = []
    for data in lines:
        translations = data.get("translation", [])
        for sp in translations:
            if not isinstance(sp, dict): continue
            s = sp.get(src_col); t = sp.get(tgt_col)
            if isinstance(s, list) or isinstance(t, list): continue
            if s and t:
                s_clean = clean_text_from_punctuation(s); t_clean = clean_text_from_punctuation(t)
                if len(s_clean) >= min_chars and len(t_clean) >= min_chars:
                    out.append((s.strip(), t.strip()))
    return out

def sample_pairs(pool: List[Tuple[str,str]], k: Optional[int], seed: int = 42) -> List[Tuple[str,str]]:
    if not pool: return []
    rng = random.Random(seed)
    if k is None or k < 0 or k >= len(pool): return pool[:]
    return rng.sample(pool, k)

def dedup_pairs(pairs: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    seen, out = set(), []
    for a,b in pairs:
        if (a,b) not in seen: seen.add((a,b)); out.append((a,b))
    return out

# --------------------------
# Pool builders (auto column match)
# --------------------------
def cols_exist(df: pd.DataFrame, a: str, b: str) -> bool:
    return a in df.columns and b in df.columns

def safe_pair_list_from_df(df: Optional[pd.DataFrame], left_candidates: List[str], right_candidates: List[str], name: str) -> List[Tuple[str,str]]:
    if df is None or len(df) == 0: return []
    for L in left_candidates:
        for R in right_candidates:
            if cols_exist(df, L, R):
                pairs = []
                for _, row in df[[L, R]].dropna().iterrows():
                    a = str(row[L]).strip(); b = str(row[R]).strip()
                    if a and b: pairs.append((a,b))
                if pairs:
                    print(f"[pool] {name}: using ({L} -> {R}) | {len(pairs)} pairs")
                    return pairs
    print(f"[warn] {name}: no matching column pair; skipping")
    return []

def build_pools(summary_df, doc_df_de, doc_df_fr, mono_df, lux_pairs) -> Dict[str, List[Tuple[str,str]]]:
    pools = {}
    # From query_doc_dataset_random_noise
    pools["sum2doc"]       = safe_pair_list_from_df(summary_df, ["summary","summ","abstract"], ["text","document","doc"], "summary->text")
    pools["sum2sum_noise"] = safe_pair_list_from_df(summary_df, ["summary","summ","abstract"], ["summary_noise","summary_noised","summary_noise_random05","summ_noise"], "summary->summary_noise")
    pools["q2doc"]         = safe_pair_list_from_df(summary_df, ["query","q"], ["text","document","doc"], "query->text")
    pools["doc2doc"]         = safe_pair_list_from_df(summary_df, ["text","document","doc"], ["text_noised","document_noised","doc_noised"], "doc->doc")
    pools["q2sum"]         = safe_pair_list_from_df(summary_df, ["query","q"], ["summary","summ","abstract"], "query->summary")
    # From de_docs_random_noise / fr_docs_random_noise
    pools["dd_german"]     = safe_pair_list_from_df(doc_df_de, ["text","sentence"], ["text_noised","text_noise_random05","sentence_noise_random05"], "de:text->text_noised")
    pools["dd_french"]     = safe_pair_list_from_df(doc_df_fr, ["text","sentence"], ["text_noised","text_noise_random05","sentence_noise_random05"], "fr:text->text_noised")
    # From TED_data_random_noise_concat
    pools["ted_q2q"]       = safe_pair_list_from_df(mono_df, ["sentence","text","de","fr"], ["sentence_noise_random05","text_noised","de_005","fr_005"], "TED:sentence->sentence_noise")
    # From LUX (lb <-> {de, fr, en})
    if lux_pairs:
        pools["lux_q2q"] = lux_pairs[:]
        print(f"[pool] lux_q2q: {len(pools['lux_q2q'])} pairs")
    return {k:v for k,v in pools.items() if v}

# --------------------------
# Batch schedule (exactly 1 batch per available task)
# --------------------------
TASK_ORDER = [
    ("sum2doc",       1),
    ("sum2sum_noise", 1),
    ("q2doc",         1),
    ("q2sum",         1),
    ("dd_german",     2),
    ("dd_french",     2),
    ("ted_q2q",       2),
    ("lux_q2q",       1),
]

def _maybe_prefix(text: str, role: str, enable_prompts: bool, model_name: str) -> str:
    if not enable_prompts: return text
    lower = model_name.lower()
    if   "multilingual-e5" in lower or "e5-" in lower:
        if role == "query":   return f"query: {text}"
        if role == "summary": return f"summary: {text}"
        return f"document: {text}"
    return text

def sample_batch_from_pool(pool: List[Tuple[str,str]], batch_size: int, replacement: bool, rng: random.Random):
    n = len(pool)
    if n == 0: return []
    if not replacement and batch_size <= n:
        idx = rng.sample(range(n), batch_size); return [pool[i] for i in idx]
    return [pool[rng.randrange(n)] for _ in range(batch_size)]

def build_epoch_examples(
    pools: Dict[str, List[Tuple[str,str]]],
    model_name: str,
    batch_size: int,
    batches_per_epoch: int,
    replacement: bool,
    seed: int,
    enable_prompts: bool,
    task_allowlist: Optional[List[str]] = None,
    include_lux_reverse: bool = True,
) -> List[InputExample]:
    """
    Create InputExamples for exactly `batches_per_epoch` batches,
    1 batch per task in TASK_ORDER if available (cyclic schedule).
    """
    pools = {k:v for k,v in pools.items() if v}
    if task_allowlist:
        pools = {k:v for k,v in pools.items() if k in set(task_allowlist)}

    # Mirror LUX to improve symmetry
    if include_lux_reverse and "lux_q2q" in pools:
        lux = pools["lux_q2q"]
        pools["lux_q2q"] = dedup_pairs(lux + [(b, a) for (a, b) in lux])

    active = [(k,c) for (k,c) in TASK_ORDER if k in pools]
    if not active:
        raise ValueError("No active pools found after filtering.")

    ROLE_MAP = {
        "sum2doc"      : ("summary","document"),
        "sum2sum_noise": ("summary","summary"),
        "q2doc"        : ("query","document"),
        "q2sum"        : ("query","summary"),
        "doc2doc"      : ("document","document"),
        "dd_german"    : ("document","document"),
        "dd_french"    : ("document","document"),
        "ted_q2q"      : ("document","document"),
        "lux_q2q"      : ("query","document"),
    }

    steps_per_cycle = sum(c for _,c in active)
    cycles = math.ceil(batches_per_epoch / steps_per_cycle)
    rng = random.Random(seed)
    produced, target = 0, batches_per_epoch
    examples: List[InputExample] = []

    for _ in range(cycles):
        for task_key, num_batches in active:
            pool = pools[task_key]
            lrole, rrole = ROLE_MAP.get(task_key, ("document","document"))
            for _ in range(num_batches):
                if produced >= target: break
                pairs = sample_batch_from_pool(pool, batch_size, replacement, rng)
                for (a,b) in pairs:
                    ap = _maybe_prefix(a, "query" if lrole=="query" else lrole, enable_prompts, model_name)
                    bp = _maybe_prefix(b, "document" if rrole=="document" else rrole, enable_prompts, model_name)
                    examples.append(InputExample(texts=[ap, bp], label=1.0))
                produced += 1
            if produced >= target: break
        if produced >= target: break

    expected = batch_size * target
    if len(examples) != expected:
        print(f"[warn] Built {len(examples)} vs expected {expected}.")
    print(f"[info] Active tasks: {', '.join(k for k,_ in active)} | steps/cycle={steps_per_cycle} | cycles={cycles}")
    return examples

class PrebuiltExampleDataset(Dataset):
    def __init__(self, items: List[InputExample]): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx: int): return self.items[idx]

# --------------------------
# Naming helpers
# --------------------------
def task_tags(enabled_task_keys: List[str]) -> str:
    tag_map = {"sum2doc":"SUM","sum2sum_noise":"SUMN","q2doc":"QDOC","q2sum":"QSUM","dd_german":"DE","dd_french":"FR","ted_q2q":"TED","lux_q2q":"LUX"}
    return "-".join(tag_map[k] for k in enabled_task_keys)

def make_run_name(model_name: str, data_sig: str, enabled_task_keys: List[str], steps: int, bs: int, seed: int) -> str:
    base = model_name.replace("/", "_")
    tags = task_tags(enabled_task_keys)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{base}+{data_sig}-steps{steps}-bs{bs}-seed{seed}-{now}"
    return name[:240]

# --------------------------
# Train once (task-batch regime)
# --------------------------


def run_two_stage(seed: int, stage1_exp: dict, stage2_exp: dict,
                  stage2_lr: Optional[float] = None, stage2_epochs: Optional[int] = None):
    """
    Train Stage-1 with stage1_exp; then continue Stage-2 from the Stage-1 saved model,
    using stage2_exp (typically LUX-only). You can optionally change lr/epochs in Stage-2.
    """
    print(f"\n================= Two-Stage Training | Seed {seed} =================")

    # ---- Stage 1 ----
    print(f"\n----- Stage 1: {stage1_exp['data_sig']} -----")
    a1 = SimpleNamespace(**base_args.__dict__.copy())
    a1.seed = seed
    for k, v in stage1_exp.items():
        setattr(a1, k, v)

    out_dir_stage1 = train_task_batches_once(a1)
    print("[stage-1 done] saved at:", out_dir_stage1)

    # free up memory between stages
    del a1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Stage 2 (continue from Stage 1) ----
    print(f"\n----- Stage 2: {stage2_exp['data_sig']} (init from Stage-1) -----")
    a2 = SimpleNamespace(**base_args.__dict__.copy())
    a2.seed = seed

    # IMPORTANT: initialize second stage from the first stage's saved checkpoint
    a2.model_name = out_dir_stage1

    # Optionally tweak LR/epochs for the second stage
    if stage2_lr is not None:
        a2.lr = stage2_lr
    if stage2_epochs is not None:
        a2.epochs = stage2_epochs

    # Keep other base args; then apply stage-2 experiment overrides
    for k, v in stage2_exp.items():
        setattr(a2, k, v)

    # Tag the run name so it’s obvious this is continued
    # a2.data_sig = f"{stage2_exp.get('data_sig','STAGE2')}_from-{os.path.basename(out_dir_stage1)}"
    stage1_tag = short_tag_from_path(out_dir_stage1, maxlen=30)
    a2.data_sig = f"{stage2_exp.get('data_sig','STAGE2')}_from-{stage1_tag}"


    out_dir_stage2 = train_task_batches_once(a2)
    print("[stage-2 done] saved at:", out_dir_stage2)
    return out_dir_stage1, out_dir_stage2

def run_setups(
    set_ups,
    seeds=(42,),
    stage2_lr=2e-5,
    stage2_epochs=1,
    zip_after=True,
):
    saved_paths = []

    for setup in set_ups:
        model_name = setup["model_name"]
        exps = setup["experiments"]
        # normalize to tuple
        if not isinstance(exps, (list, tuple)):
            exps = (exps,)

        model_slug = slugify(model_name)
        model_outdir = os.path.join(base_args.output_dir, model_slug)

        for s in seeds:
            print(f"\n================= Model {model_name} | Seed {s} =================")

            try:
                if len(exps) == 2:
                    # TWO-STAGE: exp[0] -> exp[1]
                    stg1 = dict(exps[0])
                    stg2 = dict(exps[1])

                    # stage-1 must know which base model to start from and where to write
                    stg1["model_name"] = model_name
                    stg1["output_dir"] = model_outdir

                    # stage-2 writes to same root; model_name will be set to stage1 path inside run_two_stage
                    stg2["output_dir"] = model_outdir

                    out1, out2 = run_two_stage(
                        seed=s,
                        stage1_exp=stg1,
                        stage2_exp=stg2,
                        stage2_lr=stage2_lr,
                        stage2_epochs=stage2_epochs,
                    )
                    saved_paths.extend([out1, out2])

                    final_dir = out2  # last stage is the deliverable
                    if zip_after:
                        base_name = final_dir
                        root_dir  = os.path.dirname(final_dir) or "."
                        base_dir  = os.path.basename(final_dir)
                        zip_path  = shutil.make_archive(base_name=base_name, format="zip",
                                                        root_dir=root_dir, base_dir=base_dir)
                        print("[zip] ->", zip_path)

                elif len(exps) == 1:
                    # SINGLE-STAGE
                    exp = dict(exps[0])
                    a = SimpleNamespace(**base_args.__dict__.copy())
                    a.seed = s
                    a.model_name = model_name
                    a.output_dir = model_outdir
                    for k, v in exp.items():
                        setattr(a, k, v)

                    out_dir = train_task_batches_once(a)
                    saved_paths.append(out_dir)

                    if zip_after:
                        base_name = out_dir
                        root_dir  = os.path.dirname(out_dir) or "."
                        base_dir  = os.path.basename(out_dir)
                        zip_path  = shutil.make_archive(base_name=base_name, format="zip",
                                                        root_dir=root_dir, base_dir=base_dir)
                        print("[zip] ->", zip_path)

                else:
                    raise ValueError("Each 'experiments' must be a single dict or a 2-tuple for two-stage.")

            except torch.cuda.OutOfMemoryError as e:
                print(f"[OOM] {model_name} | seed {s}: {e}. Consider lowering batch_size or sequence length.")
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\nSaved runs:")
    for p in saved_paths:
        print(" -", p)

def train_task_batches_once(args):
    """
    args fields:
      model_name, batch_size, batches_per_epoch, epochs, lr, seed, replacement, amp,
      max_seq_length, enable_prompts, lux_sample_k, output_dir, task_allowlist (optional),
      document_file_de, document_file_fr, summary_file, mono, lux_file_en, lux_file_de, lux_file_fr,
      data_sig (for naming)
    """
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")

    # --- load datasets (commenting out -> auto-skip) ---
    doc_df_de  = optional_read_csv(getattr(args, "document_file_de", None))
    doc_df_fr  = optional_read_csv(getattr(args, "document_file_fr", None))
    summary_df = optional_read_csv(getattr(args, "summary_file", None))
    # if summary_df is not None:
    #     if 'language' in summary_df.columns:
    #         summary_df = summary_df[summary_df['language'].isin(['de','fr','es'])]
    #     else:
    #         print("[warn] summary_df has no 'language' column; skipping language filter")


    mono_df    = optional_read_csv(getattr(args, "mono", None))

    lb_de = optional_load_jsonl(getattr(args, "lux_file_de", None))
    lb_fr = optional_load_jsonl(getattr(args, "lux_file_fr", None))
    lb_en = optional_load_jsonl(getattr(args, "lux_file_en", None))

    lb_de_pairs = extract_parallel_sentences(lb_de, src_col="lb", tgt_col="de") if lb_de else []
    lb_fr_pairs = extract_parallel_sentences(lb_fr, src_col="lb", tgt_col="fr") if lb_fr else []
    lb_en_pairs = extract_parallel_sentences(lb_en, src_col="lb", tgt_col="en") if lb_en else []

    lux_pairs = []
    if any([lb_de_pairs, lb_fr_pairs, lb_en_pairs]):
        print(f"Extracted LUX pairs: de={len(lb_de_pairs)} | fr={len(lb_fr_pairs)} | en={len(lb_en_pairs)}")
        k = None if getattr(args, "lux_sample_k", -1) == -1 else getattr(args, "lux_sample_k")
        pools_ = []
        if lb_de_pairs: pools_.extend(sample_pairs(lb_de_pairs, k, seed=args.seed))
        if lb_fr_pairs: pools_.extend(sample_pairs(lb_fr_pairs, k, seed=args.seed))
        if lb_en_pairs: pools_.extend(sample_pairs(lb_en_pairs, k, seed=args.seed))
        lux_pairs = dedup_pairs(pools_)
        print(f"Lux combined pool size (deduped): {len(lux_pairs)}")

    pools = build_pools(summary_df, doc_df_de, doc_df_fr, mono_df, lux_pairs)

    # enabled task keys in fixed order
    enabled_task_keys = [k for (k, _) in TASK_ORDER if k in pools]
    task_allowlist = getattr(args, "task_allowlist", None)
    if task_allowlist:
        allow = set(task_allowlist)
        enabled_task_keys = [k for k in enabled_task_keys if k in allow]
    if not enabled_task_keys:
        raise RuntimeError("No data pools available. Enable at least one dataset or relax task_allowlist.")

    examples = build_epoch_examples(
        pools=pools,
        model_name=args.model_name,
        batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch,
        replacement=args.replacement,
        seed=args.seed,
        enable_prompts=False,
        task_allowlist=task_allowlist,
        include_lux_reverse=False,
    )

    train_ds = PrebuiltExampleDataset(examples)
    model = SentenceTransformer(args.model_name, trust_remote_code=True).to(device)
    # model.max_seq_length = args.max_seq_length

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=model.smart_batching_collate,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = int(0.1 * len(train_loader))
    print(f"[info] batches_per_epoch={len(train_loader)} | warmup_steps={warmup_steps} | epochs={args.epochs}")

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = make_run_name(
        model_name=args.model_name,
        data_sig=args.data_sig,
        enabled_task_keys=enabled_task_keys,
        steps=args.batches_per_epoch,
        bs=args.batch_size,
        seed=args.seed
    )
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[info] Output dir: {save_dir}")

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        use_amp=args.amp,
        output_path=save_dir,
        show_progress_bar=True
    )

    del train_loader, train_ds, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_dir



base_args = SimpleNamespace(
    document_file_de = f"./finetuning_data/de_docs_random_noise.csv",
    document_file_fr = f"./finetuning_data/fr_docs_random_noise.csv",
    summary_file     = f"./finetuning_data/query_doc_dataset_random_noise.csv",
    mono             = f"./finetuning_data/TED_data_random_noise_concat.csv",
    lux_file_en      = f"./finetuning_data/lb_en_training_set.jsonl",
    lux_file_de      = f"./finetuning_data/lb_de_training_set.jsonl",
    lux_file_fr      = f"./finetuning_data/lb_fr_training_set.jsonl",

    # training
    # model_name = "impresso-project/histlux-gte-multilingual-base",
    model_name = "Alibaba-NLP/gte-multilingual-base",
    batch_size = 8,
    batches_per_epoch = 3000,   # ≈ 2–3k steps/epoch as requested
    epochs = 1,
    lr = 2e-5,
    seed = 42,
    replacement = False,
    amp = True,
    # max_seq_length = 512,
    enable_prompts = False,      # add 'query:' / 'document:' / 'summary:' for GTE/E5
    lux_sample_k = -1,          # -1 = all pairs
    output_dir = "./trained_models",

    # naming helper
    data_sig = "BASE",
    task_allowlist = None,
)

# --- Staged experiments you asked for ---
model_names = [
    "impresso-project/histlux-gte-multilingual-base",
    "Alibaba-NLP/gte-multilingual-base",
]

# 1) TED only
exp_TED = dict(
    data_sig="TED",
    mono=base_args.mono,
    lux_file_en="",
    lux_file_de="",
    lux_file_fr="",
    summary_file="",
    document_file_de="",
    document_file_fr="",
    task_allowlist=["ted_q2q"],
)
exp_TED_Impresso = dict(
    data_sig="TED_Impresso",
    mono=base_args.mono,
    lux_file_en="",
    lux_file_de="",
    lux_file_fr="",
    summary_file="",
    document_file_de=base_args.document_file_de,
    document_file_fr=base_args.document_file_fr,
    task_allowlist=["ted_q2q","dd_german","dd_french"],
)
exp_TED_Impresso_SUMDOCS_ = dict(
    data_sig="TED_Impresso_SUMDOCS",
    mono=base_args.mono,
    lux_file_en="",
    lux_file_de="",
    lux_file_fr="",
    summary_file=base_args.summary_file,
    document_file_de=base_args.document_file_de,
    document_file_fr=base_args.document_file_fr,
    task_allowlist=["ted_q2q","dd_german","dd_french","doc2doc"],
)
exp_TED_Impresso_SUMDOCS_SUMMARY = dict(
    data_sig="TED_Impresso_SUMDOCS_SUMMARY",
    mono=base_args.mono,
    lux_file_en="",
    lux_file_de="",
    lux_file_fr="",
    summary_file=base_args.summary_file,
    document_file_de=base_args.document_file_de,
    document_file_fr=base_args.document_file_fr,
    task_allowlist=["ted_q2q","dd_german","dd_french","doc2doc","sum2sum_noise"],
)

# 2) TED + LUX
exp_LUX = dict(
    data_sig="LUX",
    mono="",
    lux_file_en=base_args.lux_file_en,
    lux_file_de=base_args.lux_file_de,
    lux_file_fr=base_args.lux_file_fr,
    summary_file="",
    document_file_de="",
    document_file_fr="",
    task_allowlist=["lux_q2q"],
)

# 3) TED + LUX + (query -> summary) ONLY from summary_df
exp_TED_DOCS = dict(
    data_sig="TED+DOCS",
    mono=base_args.mono,
    lux_file_en="",
    lux_file_de="",
    lux_file_fr="base_args.lux_file_fr",
    summary_file=base_args.summary_file,
    document_file_de=base_args.document_file_de,
    document_file_fr=base_args.document_file_fr,
    task_allowlist=["ted_q2q","dd_german","dd_french","doc2doc"],
)



exp_TED_DOCS_SUM = dict(
    data_sig="TED+DOCS+SUM",
    mono=base_args.mono,
    lux_file_en="",
    lux_file_de="",
    lux_file_fr="",
    summary_file=base_args.summary_file,
    document_file_de=base_args.document_file_de,
    document_file_fr=base_args.document_file_fr,
    task_allowlist=["ted_q2q","dd_german","dd_french","sum2sum_noise"],
)
exp_TED_DOCS_SUM_DOC = dict(
    data_sig="TED+DOCS+SUM+SUMDOC",
    mono=base_args.mono,
    lux_file_en="",
    lux_file_de="",
    lux_file_fr="",
    summary_file="",
    document_file_de="",
    document_file_fr="",
    task_allowlist=["ted_q2q","dd_german","dd_french","sum2doc","sum2sum_noise"],
)


# Example set_ups configurations (commented out - now using command-line args)
# set_ups = [
#     {"model_name":"impresso-project/histlux-gte-multilingual-base", "experiments": exp_TED},
#     {"model_name":"impresso-project/histlux-gte-multilingual-base", "experiments": exp_TED_Impresso},
#     {"model_name":"impresso-project/histlux-gte-multilingual-base", "experiments": exp_TED_Impresso_SUMDOCS_},
#     {"model_name":"impresso-project/histlux-gte-multilingual-base", "experiments": exp_TED_Impresso_SUMDOCS_SUMMARY},
#     {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": (exp_TED, exp_LUX)},
#     {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": (exp_TED_Impresso, exp_LUX)},
#     {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": (exp_TED_Impresso_SUMDOCS_, exp_LUX)},
#     {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": (exp_TED_Impresso_SUMDOCS_SUMMARY, exp_LUX)},
#     {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": exp_TED},
#     {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": exp_TED_Impresso},
#     {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": exp_TED_Impresso_SUMDOCS_},
#     {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": exp_TED_Impresso_SUMDOCS_SUMMARY},
# ]

# STAGE1 = exp_TED_DOCS
# STAGE2 = exp_LUX


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune sentence transformers with OCR-robust data")
    parser.add_argument("--model_name", type=str, default="Alibaba-NLP/gte-multilingual-base",
                        help="Model name to fine-tune")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Update base_args seed to match the command-line argument
    base_args.seed = args.random_seed
    
    # Update the set_ups to use the provided model
    set_ups = [
        {"model_name":"impresso-project/histlux-gte-multilingual-base", "experiments": exp_TED},
        {"model_name":"impresso-project/histlux-gte-multilingual-base", "experiments": exp_TED_Impresso},
        {"model_name":"impresso-project/histlux-gte-multilingual-base", "experiments": exp_TED_Impresso_SUMDOCS_},
        {"model_name":"impresso-project/histlux-gte-multilingual-base", "experiments": exp_TED_Impresso_SUMDOCS_SUMMARY},
        {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": (exp_TED, exp_LUX)},
        {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": (exp_TED_Impresso, exp_LUX)},
        {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": (exp_TED_Impresso_SUMDOCS_, exp_LUX)},
        {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": (exp_TED_Impresso_SUMDOCS_SUMMARY, exp_LUX)},
        {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": exp_TED},
        {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": exp_TED_Impresso},
        {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": exp_TED_Impresso_SUMDOCS_},
        {"model_name":"Alibaba-NLP/gte-multilingual-base", "experiments": exp_TED_Impresso_SUMDOCS_SUMMARY},
    ]
    
    seeds = [args.random_seed]
    zip_after = True
    
    run_setups(
        set_ups=set_ups,
        seeds=seeds,
        stage2_lr=2e-5,
        stage2_epochs=1,
        zip_after=True,
    )
