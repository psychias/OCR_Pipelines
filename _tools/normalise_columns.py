#!/usr/bin/env python3
"""
normalise_columns.py — Rename CSV columns to the project-wide uniform schema.

Column convention:
  - Clean text:  {lang3}        (ISO 639-3, e.g. deu, fra, eng, spa, rus, tur, ltz)
  - Noisy text:  {lang3}_04     (the _04 suffix is the canonical ACL convention)
  - Other columns (id, label, score, …) are kept unchanged.

Usage:
    python _tools/normalise_columns.py <input_csv> --mapping '{"old":"new",...}' [--output <out_csv>]

The mapping is a JSON dict of old_column_name → new_column_name.
If --output is omitted the file is modified in place.

Idempotent: columns that already have the target name are silently skipped.
"""

import argparse
import json
import sys

import pandas as pd


def normalise(csv_path: str, mapping: dict[str, str], output_path: str | None = None) -> None:
    df = pd.read_csv(csv_path)
    rename = {k: v for k, v in mapping.items() if k in df.columns and v not in df.columns}
    if rename:
        df = df.rename(columns=rename)
        print(f"  Renamed: {rename}")
    else:
        print(f"  Nothing to rename (already normalised or columns not found)")
    out = output_path or csv_path
    df.to_csv(out, index=False)
    print(f"  Written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename CSV columns using a JSON mapping dict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_csv", help="Path to the CSV file.")
    parser.add_argument(
        "--mapping", required=True,
        help='JSON dict, e.g. \'{"French":"fra","German":"deu"}\'',
    )
    parser.add_argument("--output", default=None, help="Output path (default: overwrite input).")
    args = parser.parse_args()

    mapping = json.loads(args.mapping)
    normalise(args.input_csv, mapping, args.output)


if __name__ == "__main__":
    main()
