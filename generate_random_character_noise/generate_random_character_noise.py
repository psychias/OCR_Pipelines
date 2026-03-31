"""Generate synthetic OCR noise using script-specific confusable character tables.

Each script type defines a pool of characters (including common OCR-confusable
glyphs) that are used for substitution and insertion errors. Deletion errors
are script-agnostic.

Usage
-----
    python generate_random_character_noise.py input.csv \
        --columns text summary \
        --script latin \
        --cer 0.04 \
        --suffix _04 \
        --seed 42 \
        -o output.csv

This produces new columns ``text_04``, ``summary_04`` in the output CSV.
"""

from __future__ import annotations

import argparse
import random
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Confusable character tables
# ---------------------------------------------------------------------------

ScriptType = Literal[
    "latin", "cyrillic", "greek", "arabic", "hebrew", "georgian",
]

# Latin – taken verbatim from the OCR_Pipelines source charset
LATIN_CHARSET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "    "                         # weighted spaces
    "öüäéèà ÜÄÖ"
)

CYRILLIC_CHARSET = (
    "абвгдежзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "    "
    "ёЁ"
)

GREEK_CHARSET = (
    "αβγδεζηθικλμνξοπρστυφχψω"
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    "    "
    "άέήίόύώ"
)

ARABIC_CHARSET = (
    "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
    "    "
    "ءآأؤإئ"
)

HEBREW_CHARSET = (
    "אבגדהוזחטיכלמנסעפצקרשת"
    "    "
    "ךםןףץ"
)

GEORGIAN_CHARSET = (
    "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"
    "    "
)

CHARSETS: dict[ScriptType, str] = {
    "latin": LATIN_CHARSET,
    "cyrillic": CYRILLIC_CHARSET,
    "greek": GREEK_CHARSET,
    "arabic": ARABIC_CHARSET,
    "hebrew": HEBREW_CHARSET,
    "georgian": GEORGIAN_CHARSET,
}


# ---------------------------------------------------------------------------
# Core noise function
# ---------------------------------------------------------------------------

def apply_ocr_noise(
    text: str,
    script: ScriptType = "latin",
    target_cer: float = 0.04,
) -> str:
    """Return *text* with synthetic OCR errors at roughly *target_cer*."""
    if not isinstance(text, str):
        text = str(text)
    charset = CHARSETS[script]
    n_changes = max(1, int(len(text) * target_cer))
    mutated = list(text)

    for _ in range(n_changes):
        op = random.choice(("substitution", "insertion", "deletion"))

        if op == "substitution" and mutated:
            idx = random.randrange(len(mutated))
            mutated[idx] = random.choice(charset)

        elif op == "insertion":
            idx = random.randrange(max(len(mutated), 1))
            mutated.insert(idx, random.choice(charset))

        elif op == "deletion" and mutated:
            idx = random.randrange(len(mutated))
            del mutated[idx]

    return "".join(mutated)


# ---------------------------------------------------------------------------
# DataFrame helper
# ---------------------------------------------------------------------------

def noise_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    script: ScriptType = "latin",
    target_cer: float = 0.04,
    suffix: str = "_04",
) -> pd.DataFrame:
    """Add noised copies of *columns* with the given *suffix*."""
    out = df.copy()
    for col in columns:
        out[col + suffix] = out[col].apply(
            lambda s: apply_ocr_noise(s, script=script, target_cer=target_cer)
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add synthetic OCR noise columns to a CSV."
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--columns", nargs="+", required=True,
        help="Column names to noise.",
    )
    parser.add_argument(
        "--script", default="latin", choices=list(CHARSETS),
        help="Script / alphabet to use for confusable chars (default: latin).",
    )
    parser.add_argument(
        "--cer", type=float, default=0.04,
        help="Target character error rate (default: 0.04).",
    )
    parser.add_argument(
        "--suffix", default="_04",
        help="Suffix appended to each noised column (default: _04).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path (default: overwrite input).",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    df = pd.read_csv(args.input_csv)
    df = noise_dataframe(
        df,
        columns=args.columns,
        script=args.script,
        target_cer=args.cer,
        suffix=args.suffix,
    )
    out_path = args.output or args.input_csv
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()
