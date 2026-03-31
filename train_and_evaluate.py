#!/usr/bin/env python3
"""
train_and_evaluate.py

Sample training script for OCR M-GTE (LREC 2026).

Demonstrates the two-step fine-tuning procedure:
  Step 1 (Task A) — Historical Luxembourgish language acquisition
  Step 2 (Task B) — OCR-Error Noise Adaptation across 6 European languages

Usage:
    python train_and_evaluate.py          # uses defaults in CFG
    python train_and_evaluate.py --help
"""

import argparse, os, random, glob
import pandas as pd
import torch
from datasets import Dataset as HFDataset, concatenate_datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

# ── Config ──────────────────────────────────────────────────────────────────
CFG = dict(
    base_model        = "Alibaba-NLP/gte-multilingual-base",
    output_dir        = "./output/ocr-robust-gte-multilingual-base",
    seed              = 42,
    # Task A
    task_a_batch_size = 8,
    task_a_epochs     = 1,
    lux_data_dir      = "noisy_finetuning_data/LREC",
    # Task B
    task_b_batch_size = 8,
    task_b_epochs     = 1,
    acl_data_dir      = "noisy_finetuning_data/ACL",
    lrec_denoise_dir  = "noisy_finetuning_data/LREC",   # MLSum + Historical Articles
    max_pairs_per_src = 10_000,
    # Evaluation
    eval_dataset      = "noisy_evaluation_datasets/ACL/CLSD_WMT19_BLDS_noise.csv",
    device            = "cuda" if torch.cuda.is_available() else "cpu",
)
# ────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_crosslingual_pairs(data_dir: str, max_pairs: int) -> HFDataset:
    """Load Luxembourgish cross-lingual CSV files → HFDataset with (anchor, positive)."""
    # Schema: src_lang, tgt_lang, src, tgt
    # Mirrors both directions (src→tgt) and (tgt→src) for symmetry.
    dfs = []
    for path in glob.glob(os.path.join(data_dir, "luxembourgish*.csv")):
        df = pd.read_csv(path).dropna(subset=["src", "tgt"])
        forward = df[["src", "tgt"]].rename(columns={"src": "anchor", "tgt": "positive"})
        reverse = df[["tgt", "src"]].rename(columns={"tgt": "anchor", "src": "positive"})
        dfs.extend([forward, reverse])
    if not dfs:
        raise FileNotFoundError(f"No luxembourgish*.csv found in {data_dir}")
    combined = pd.concat(dfs).drop_duplicates().head(max_pairs)
    return HFDataset.from_pandas(combined, preserve_index=False)


def load_denoising_pairs(csv_path: str, max_pairs: int) -> HFDataset:
    """Load a monolingual denoising CSV → HFDataset with (anchor, positive).
    
    Expects columns: <lang>, <lang>_noisy
    e.g. 'de', 'de_noisy'  or  'fr', 'fr_noisy'
    """
    df = pd.read_csv(csv_path)
    # Auto-detect column pair: first non-id column + its _noisy counterpart
    clean_col = [c for c in df.columns if not c.endswith("_noisy")][0]
    noisy_col = clean_col + "_noisy"
    if noisy_col not in df.columns:
        raise ValueError(f"Expected column '{noisy_col}' in {csv_path}")
    df = df[[clean_col, noisy_col]].dropna()
    df = df.rename(columns={clean_col: "anchor", noisy_col: "positive"}).head(max_pairs)
    return HFDataset.from_pandas(df, preserve_index=False)


def evaluate_clsd(model: SentenceTransformer, csv_path: str) -> float:
    """Compute Precision@1 on a CLSD-style evaluation CSV.
    
    Expects columns: src, tgt (and optionally src_noisy, tgt_noisy).
    Groups rows by a shared query and ranks the true positive highest.
    Returns Precision@1 as a float in [0, 1].
    """
    df = pd.read_csv(csv_path)
    # Use noisy target if available (tests robustness), otherwise clean target
    tgt_col = "tgt_noisy" if "tgt_noisy" in df.columns else "tgt"
    queries   = model.encode(df["src"].tolist(),     batch_size=64, show_progress_bar=False)
    documents = model.encode(df[tgt_col].tolist(),   batch_size=64, show_progress_bar=False)

    from sentence_transformers.util import cos_sim
    import numpy as np
    sims = cos_sim(queries, documents).numpy()  # (N, N)
    correct = (np.argmax(sims, axis=1) == np.arange(len(df))).mean()
    print(f"  Precision@1 on {os.path.basename(csv_path)}: {correct:.4f}")
    return float(correct)


def train_step(
    model: SentenceTransformer,
    dataset: HFDataset,
    output_dir: str,
    batch_size: int,
    epochs: int,
    step_name: str,
) -> SentenceTransformer:
    loss = losses.MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=50,
        seed=CFG["seed"],
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )
    print(f"\n{'='*60}\n  {step_name}\n{'='*60}")
    trainer.train()
    return model


def main() -> None:
    set_seed(CFG["seed"])
    model = SentenceTransformer(CFG["base_model"])
    model.to(CFG["device"])

    # ── [2] Evaluate before training ────────────────────────────────────────
    print("\n[PRE-TRAINING EVALUATION]")
    evaluate_clsd(model, CFG["eval_dataset"])

    # ── [3] Step 1 — Task A: Historical Luxembourgish ───────────────────────
    lux_dataset = load_crosslingual_pairs(CFG["lux_data_dir"], CFG["max_pairs_per_src"])
    model = train_step(
        model, lux_dataset,
        output_dir=os.path.join(CFG["output_dir"], "task_a"),
        batch_size=CFG["task_a_batch_size"],
        epochs=CFG["task_a_epochs"],
        step_name="Task A — Historical Luxembourgish",
    )

    # ── [4] Step 2 — Task B: OCR-Error Noise Adaptation ─────────────────────
    # Collect all monolingual denoising CSVs from ACL/ and LREC/
    denoise_paths = (
        glob.glob(os.path.join(CFG["acl_data_dir"],   "*.csv")) +
        glob.glob(os.path.join(CFG["lrec_denoise_dir"], "*.csv"))
    )
    # Exclude Luxembourgish cross-lingual files from Task B
    denoise_paths = [p for p in denoise_paths if "luxembourgish" not in os.path.basename(p).lower()]

    task_b_datasets = []
    for path in denoise_paths:
        try:
            ds = load_denoising_pairs(path, CFG["max_pairs_per_src"])
            task_b_datasets.append(ds)
            print(f"  Loaded {len(ds):,} pairs from {os.path.basename(path)}")
        except (ValueError, KeyError) as e:
            print(f"  [SKIP] {os.path.basename(path)}: {e}")

    if not task_b_datasets:
        raise RuntimeError("No denoising datasets found for Task B.")

    combined_task_b = concatenate_datasets(task_b_datasets).shuffle(seed=CFG["seed"])
    model = train_step(
        model, combined_task_b,
        output_dir=os.path.join(CFG["output_dir"], "task_b"),
        batch_size=CFG["task_b_batch_size"],
        epochs=CFG["task_b_epochs"],
        step_name="Task B — OCR-Error Noise Adaptation (all languages)",
    )

    # ── [5] Evaluate after training ──────────────────────────────────────────
    print("\n[POST-TRAINING EVALUATION]")
    evaluate_clsd(model, CFG["eval_dataset"])

    # ── [6] Save ─────────────────────────────────────────────────────────────
    os.makedirs(CFG["output_dir"], exist_ok=True)
    model.save(CFG["output_dir"])
    print(f"\nModel saved to {CFG['output_dir']}")


if __name__ == "__main__":
    main()
