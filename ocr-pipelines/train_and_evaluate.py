"""Train and evaluate OCR-robust multilingual embeddings.

Full two-step fine-tuning pipeline from:
- LREC 2026: "A Recipe for Adapting Multilingual Embedders to OCR-Error
              Robustness and Historical Texts"
- Built upon ACL 2025: "Cheap Character Noise for OCR-Robust Multilingual
                        Embeddings"

Steps:
  [0] Setup & config
  [1] Load base model
  [2] Evaluate BEFORE training (CLSD DE↔FR sanity check)
  [3] Task A — Historical Luxembourgish fine-tuning
  [4] Task B — OCR-Error Noise Adaptation (denoising, all languages)
  [5] Evaluate AFTER training
  [6] Save adapted model
"""

from __future__ import annotations

import glob
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset, concatenate_datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

# ---------------------------------------------------------------------------
# [0] Configuration
# ---------------------------------------------------------------------------
CFG = dict(
    base_model="Alibaba-NLP/gte-multilingual-base",
    data_root=".",  # root of this repo
    output_dir="./output",
    seed=42,
    task_a_batch_size=8,
    task_b_batch_size=8,
    task_a_epochs=1,
    task_b_epochs=1,
    max_pairs_per_src=10_000,  # cap per CSV to manage memory
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_lora=False,  # set True to enable LoRA (PEFT)
    add_gte_prefixes=True,  # "query:" / "document:" prefixes for GTE
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_denoising_pairs(
    csv_path: str,
    lang_col: str,
    noise_col: str,
    max_pairs: int,
) -> HFDataset:
    """Load a denoising CSV and return an HF Dataset with anchor/positive."""
    df = pd.read_csv(csv_path)
    df = df[[lang_col, noise_col]].dropna()
    df = df[(df[lang_col].str.strip() != "") & (df[noise_col].str.strip() != "")]
    if len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=CFG["seed"])
    return HFDataset.from_dict(
        {"anchor": df[lang_col].tolist(), "positive": df[noise_col].tolist()}
    )


def load_crosslingual_pairs(
    csv_path: str,
    max_pairs: int,
) -> HFDataset:
    """Load cross-lingual CSV (src/tgt columns) as anchor/positive pairs."""
    df = pd.read_csv(csv_path)
    df = df[["src", "tgt"]].dropna()
    df = df[(df["src"].str.strip() != "") & (df["tgt"].str.strip() != "")]
    if len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=CFG["seed"])
    return HFDataset.from_dict(
        {"anchor": df["src"].tolist(), "positive": df["tgt"].tolist()}
    )


def evaluate_clsd(model: SentenceTransformer, clsd_csv_path: str) -> float:
    """Evaluate Precision@1 on a CLSD evaluation CSV.

    For each source sentence, compute cosine similarity against all target
    sentences and check whether the correct target is ranked first.
    """
    df = pd.read_csv(clsd_csv_path)
    positives = df[df["label"] == 1].reset_index(drop=True)

    if len(positives) == 0:
        print("  [WARN] No positive pairs found in", clsd_csv_path)
        return 0.0

    src_texts = positives["src"].tolist()
    tgt_texts = positives["tgt"].tolist()

    prefix_q = "query: " if CFG["add_gte_prefixes"] else ""
    prefix_d = "document: " if CFG["add_gte_prefixes"] else ""

    src_emb = model.encode(
        [prefix_q + s for s in src_texts],
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    tgt_emb = model.encode(
        [prefix_d + t for t in tgt_texts],
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    # Cosine similarity matrix
    src_norm = torch.nn.functional.normalize(src_emb, p=2, dim=1)
    tgt_norm = torch.nn.functional.normalize(tgt_emb, p=2, dim=1)
    sim_matrix = src_norm @ tgt_norm.T  # (N, N)

    # Precision@1: for each row, check if argmax == diagonal index
    preds = sim_matrix.argmax(dim=1)
    correct = (preds == torch.arange(len(preds), device=preds.device)).sum().item()
    p_at_1 = correct / len(preds)

    print(f"  CLSD Precision@1 = {p_at_1:.4f}  ({correct}/{len(preds)})")
    return p_at_1


def _apply_lora(model: SentenceTransformer) -> SentenceTransformer:
    """Optionally wrap the model with LoRA adapters."""
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
    return model


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(CFG["seed"])
    data_root = CFG["data_root"]

    # -----------------------------------------------------------------------
    # [1] Load base model
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("[1] Loading base model:", CFG["base_model"])
    print("=" * 60)
    model = SentenceTransformer(CFG["base_model"], device=CFG["device"])

    if CFG["use_lora"]:
        print("  Applying LoRA adapters ...")
        model = _apply_lora(model)

    # -----------------------------------------------------------------------
    # [2] Evaluate BEFORE training
    # -----------------------------------------------------------------------
    clsd_eval_path = os.path.join(
        data_root,
        "noisy_evaluation_datasets",
        "ACL",
        "CLSD_WMT19_blackletter_scanned_noise.csv",
    )
    print("\n" + "=" * 60)
    print("[2] Evaluating BEFORE training")
    print("=" * 60)
    if os.path.exists(clsd_eval_path):
        evaluate_clsd(model, clsd_eval_path)
    else:
        print(f"  [SKIP] Eval file not found: {clsd_eval_path}")

    # -----------------------------------------------------------------------
    # [3] Task A — Historical Luxembourgish fine-tuning
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[3] Task A: Historical Luxembourgish fine-tuning")
    print("=" * 60)

    lux_hist_path = os.path.join(
        data_root, "noisy_finetuning_data", "LREC", "luxembourgish_historical_crosslingual.csv"
    )
    lux_modern_path = os.path.join(
        data_root, "noisy_finetuning_data", "LREC", "luxembourgish_modern_crosslingual.csv"
    )

    task_a_datasets: list[HFDataset] = []
    for path in [lux_hist_path, lux_modern_path]:
        if os.path.exists(path):
            ds = load_crosslingual_pairs(path, CFG["max_pairs_per_src"])
            task_a_datasets.append(ds)
            print(f"  Loaded {len(ds)} pairs from {os.path.basename(path)}")
        else:
            print(f"  [SKIP] {path} not found")

    if task_a_datasets:
        task_a_data = concatenate_datasets(task_a_datasets)
        print(f"  Total Task A pairs: {len(task_a_data)}")

        loss_a = losses.MultipleNegativesRankingLoss(model)

        training_args_a = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(CFG["output_dir"], "task_a_ckpt"),
            num_train_epochs=CFG["task_a_epochs"],
            per_device_train_batch_size=CFG["task_a_batch_size"],
            warmup_ratio=0.1,
            seed=CFG["seed"],
            logging_steps=50,
            save_strategy="no",
        )

        trainer_a = SentenceTransformerTrainer(
            model=model,
            args=training_args_a,
            train_dataset=task_a_data,
            loss=loss_a,
        )
        trainer_a.train()
        print("  Task A training complete.")
    else:
        print("  [SKIP] No Task A data found.")

    # -----------------------------------------------------------------------
    # [4] Task B — OCR-Error Noise Adaptation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[4] Task B: OCR-Error Noise Adaptation (denoising)")
    print("=" * 60)

    task_b_datasets: list[HFDataset] = []

    # Collect all monolingual denoising CSVs from ACL/ and LREC/
    for sub in ["ACL", "LREC"]:
        pattern = os.path.join(data_root, "noisy_finetuning_data", sub, "*.csv")
        csv_files = sorted(glob.glob(pattern))
        for csv_path in csv_files:
            df_peek = pd.read_csv(csv_path, nrows=0)
            cols = list(df_peek.columns)

            # Skip cross-lingual files (they have src_lang/tgt_lang columns)
            if "src_lang" in cols or "tgt_lang" in cols:
                continue

            # Identify the lang column and its noised counterpart
            lang_col = None
            noise_col = None
            for c in cols:
                if c.endswith("_04"):
                    noise_col = c
                    lang_col = c.replace("_04", "")

            if lang_col and noise_col and lang_col in cols:
                ds = load_denoising_pairs(
                    csv_path, lang_col, noise_col, CFG["max_pairs_per_src"]
                )
                task_b_datasets.append(ds)
                print(f"  Loaded {len(ds)} pairs from {sub}/{os.path.basename(csv_path)}")

    if task_b_datasets:
        # Interleave: cycle through datasets one batch at a time
        task_b_data = concatenate_datasets(task_b_datasets)
        print(f"  Total Task B pairs: {len(task_b_data)}")

        loss_b = losses.MultipleNegativesRankingLoss(model)

        num_batches = math.ceil(len(task_b_data) / CFG["task_b_batch_size"])
        logging_steps = max(1, num_batches // 20)

        training_args_b = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(CFG["output_dir"], "task_b_ckpt"),
            num_train_epochs=CFG["task_b_epochs"],
            per_device_train_batch_size=CFG["task_b_batch_size"],
            warmup_ratio=0.1,
            seed=CFG["seed"],
            logging_steps=logging_steps,
            save_strategy="no",
        )

        trainer_b = SentenceTransformerTrainer(
            model=model,
            args=training_args_b,
            train_dataset=task_b_data,
            loss=loss_b,
        )
        trainer_b.train()
        print("  Task B training complete.")
    else:
        print("  [SKIP] No Task B data found.")

    # -----------------------------------------------------------------------
    # [5] Evaluate AFTER training
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[5] Evaluating AFTER training")
    print("=" * 60)
    if os.path.exists(clsd_eval_path):
        evaluate_clsd(model, clsd_eval_path)
    else:
        print(f"  [SKIP] Eval file not found: {clsd_eval_path}")

    # -----------------------------------------------------------------------
    # [6] Save adapted model
    # -----------------------------------------------------------------------
    save_path = os.path.join(CFG["output_dir"], "ocr-robust-gte-multilingual-base")
    print("\n" + "=" * 60)
    print(f"[6] Saving model to {save_path}")
    print("=" * 60)
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)
    print("  Done.")


if __name__ == "__main__":
    main()
