#!/usr/bin/env python3
"""
generate_random_character_noise.py

Standalone CLI script that applies stochastic character-level perturbations to one or more
columns of a CSV file and writes a new CSV with additional `{col}_noisy` columns.

Supports multiple writing systems via the --script argument.

References:
  - ACL 2025: "Cheap Character Noise for OCR-Robust Multilingual Embeddings"
  - LREC 2026: "A Recipe for Adapting Multilingual Embedders to OCR-Error Robustness
                and Historical Texts"
"""

import argparse
import random
import sys
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Script character pools
# ---------------------------------------------------------------------------
SCRIPT_CHARS: dict[str, str] = {
    "latin": (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
        "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ"
    ),
    "cyrillic": (
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    ),
    "arabic":   "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
    "greek":    "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
    "georgian": "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ",
    "hebrew":   "אבגדהוזחטיכלמנסעפצקרשת",
}

ScriptLiteral = Literal["latin", "cyrillic", "arabic", "greek", "georgian", "hebrew"]


def perturb_text(text: str, char_pool: str, noise_rate: float, rng: random.Random) -> str:
    """Apply random substitution / insertion / deletion to `noise_rate` fraction of chars."""
    chars = list(text)
    n_perturb = max(1, round(len(chars) * noise_rate))
    indices = rng.sample(range(len(chars)), min(n_perturb, len(chars)))
    for idx in sorted(indices, reverse=True):  # reverse so insertions don't shift later indices
        op = rng.choice(["sub", "ins", "del"])
        if op == "sub":
            chars[idx] = rng.choice(char_pool)
        elif op == "ins":
            chars.insert(idx, rng.choice(char_pool))
        else:  # del
            del chars[idx]
    return "".join(chars)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply random character-level OCR noise to CSV column(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_file",      required=True,  help="Path to input CSV.")
    parser.add_argument("--output_file",     required=True,  help="Path to write output CSV.")
    parser.add_argument("--target_columns",  required=True,  nargs="+",
                        help="Column name(s) to apply noise to.")
    parser.add_argument(
        "--script",
        required=True,
        choices=list(SCRIPT_CHARS.keys()),   # enforces Literal at parse time
        help="Writing system — determines the noise character pool.",
    )
    parser.add_argument("--noise_rate",   type=float, default=0.05,
                        help="Fraction of characters to perturb.")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--output_suffix", type=str,  default="_noisy",
                        help="Suffix appended to noised column names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    char_pool = SCRIPT_CHARS[args.script]

    df = pd.read_csv(args.input_file)
    for col in args.target_columns:
        if col not in df.columns:
            print(f"[WARNING] Column '{col}' not found in {args.input_file}. Skipping.")
            continue
        noisy_col = col + args.output_suffix
        df[noisy_col] = df[col].astype(str).apply(
            lambda t: perturb_text(t, char_pool, args.noise_rate, rng)
        )

    df.to_csv(args.output_file, index=False)

    # Summary
    print(f"Processed {len(df):,} rows.")
    print(f"Noised columns: {[c + args.output_suffix for c in args.target_columns]}")
    total_chars = sum(df[c].astype(str).apply(len).sum() for c in args.target_columns if c in df.columns)
    print(f"Approx. target noise rate: {args.noise_rate:.1%} over ~{total_chars:,} chars.")


if __name__ == "__main__":
    main()
