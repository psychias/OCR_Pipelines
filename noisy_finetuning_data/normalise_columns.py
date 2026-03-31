#!/usr/bin/env python3
"""
normalise_columns.py

Enforces a uniform column schema across all finetuning CSVs in ACL/ and LREC/.

Monolingual denoising pairs → columns: lang, lang_noisy
Cross-lingual pairs          → columns: src_lang, tgt_lang, src, tgt

Idempotent: safe to run multiple times.

Usage:
    python noisy_finetuning_data/normalise_columns.py
"""

import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Monolingual denoising CSVs ────────────────────────────────────────────
# Map of (filename pattern) → list of (clean_col, noisy_col, target_lang)
MONO_RENAMES: dict[str, list[tuple[str, str, str]]] = {
    # ACL: TED random noise (German & French, multiple noise levels)
    "TED_data_random_noise.csv": [
        ("german", "german_noise_random05", "de"),
        ("french", "french_noise_random05", "fr"),
    ],
    "TED_data_random_noise_concat.csv": [
        ("sentence", "sentence_noise_random05", "de"),
    ],
    # ACL: TED realistic noise (BLDS, SNP per language)
    "TED_data_realistic_noise.csv": [
        ("german", "german_noise_BLDS", "de"),
        ("french", "french_noise_BLDS", "fr"),
    ],
    # ACL: X-News
    "X-News_data_random_noise.csv": [
        ("german", "german_noise_random05", "de"),
        ("french", "french_noise_random05", "fr"),
    ],
    # LREC: Historical Articles
    "de_docs_random_noise.csv": [
        ("text", "text_noised", "de"),
    ],
    "fr_docs_random_noise.csv": [
        ("text", "text_noised", "fr"),
    ],
    # LREC: MLSum
    "query_doc_dataset_random_noise.csv": [
        ("text", "text_noised", "de"),
    ],
}


def normalise_monolingual(csv_path: str, mappings: list[tuple[str, str, str]]) -> None:
    """Add standardised `lang` / `lang_noisy` columns to a monolingual denoising CSV."""
    df = pd.read_csv(csv_path)
    changed = False
    for clean_col, noisy_col, lang in mappings:
        target_clean = lang
        target_noisy = f"{lang}_noisy"
        if clean_col in df.columns and target_clean not in df.columns:
            df[target_clean] = df[clean_col]
            changed = True
        if noisy_col in df.columns and target_noisy not in df.columns:
            df[target_noisy] = df[noisy_col]
            changed = True
    if changed:
        df.to_csv(csv_path, index=False)
        print(f"  [MONO] Normalised {os.path.basename(csv_path)}")
    else:
        print(f"  [MONO] Already normalised: {os.path.basename(csv_path)}")


def normalise_crosslingual_jsonl(jsonl_path: str) -> None:
    """Convert Luxembourgish JSONL to standardised CSV with src_lang, tgt_lang, src, tgt."""
    basename = os.path.basename(jsonl_path)
    csv_path = jsonl_path.replace(".jsonl", ".csv")

    # Detect language pair from filename: lb_de, lb_en, lb_fr
    parts = basename.replace("_training_set.jsonl", "").split("_")
    if len(parts) < 2:
        print(f"  [SKIP] Cannot parse language pair from {basename}")
        return

    src_code_short = parts[0]  # lb
    tgt_code_short = parts[1]  # de, en, or fr

    iso3_map = {"lb": "ltz", "de": "deu", "en": "eng", "fr": "fra"}
    src_lang = iso3_map.get(src_code_short, src_code_short)
    tgt_lang = iso3_map.get(tgt_code_short, tgt_code_short)

    rows = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            pairs = record.get("translation", [])
            for pair in pairs:
                src_text = pair.get(src_code_short, "")
                tgt_text = pair.get(tgt_code_short, "")
                if src_text and tgt_text:
                    rows.append({
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                        "src": src_text,
                        "tgt": tgt_text,
                    })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"  [CROSS] Converted {basename} → {os.path.basename(csv_path)} ({len(df)} pairs)")
    else:
        print(f"  [SKIP] No pairs extracted from {basename}")


def main() -> None:
    # Process monolingual CSVs
    for subdir in ["ACL", "LREC"]:
        dirpath = os.path.join(HERE, subdir)
        if not os.path.isdir(dirpath):
            continue
        for csv_file in sorted(os.listdir(dirpath)):
            if csv_file in MONO_RENAMES:
                normalise_monolingual(
                    os.path.join(dirpath, csv_file),
                    MONO_RENAMES[csv_file],
                )

    # Process cross-lingual JSONL files
    for jsonl_path in sorted(glob.glob(os.path.join(HERE, "LREC", "*.jsonl"))):
        normalise_crosslingual_jsonl(jsonl_path)

    print("Done.")


if __name__ == "__main__":
    main()
