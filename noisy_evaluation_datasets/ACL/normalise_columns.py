#!/usr/bin/env python3
"""
normalise_columns.py

Adds standardised column aliases (src, tgt, src_lang, tgt_lang, src_noisy, tgt_noisy)
to the CLSD evaluation CSVs without removing any existing columns.

Idempotent: safe to run multiple times — existing normalised columns are overwritten.

Usage:
    python noisy_evaluation_datasets/ACL/normalise_columns.py
"""

import glob
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Mapping from original column names to standardised aliases
COLUMN_MAP = {
    "French": "src",
    "German": "tgt",
    "fr_adv1": "src_noisy",   # first adversarial variant used as canonical noisy source
    "de_adv1": "tgt_noisy",   # first adversarial variant used as canonical noisy target
}


def normalise(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    # Add standardised columns
    for orig, alias in COLUMN_MAP.items():
        if orig in df.columns:
            df[alias] = df[orig]

    # Add language metadata columns
    df["src_lang"] = "fra"
    df["tgt_lang"] = "deu"

    # Add an id column if missing
    if "id" not in df.columns:
        df.insert(0, "id", range(len(df)))

    df.to_csv(csv_path, index=False)
    print(f"  Normalised {os.path.basename(csv_path)} ({len(df)} rows)")


def main() -> None:
    csvs = sorted(glob.glob(os.path.join(HERE, "CLSD_*.csv")))
    if not csvs:
        print("No CLSD CSV files found.")
        return
    for path in csvs:
        normalise(path)
    print("Done.")


if __name__ == "__main__":
    main()
