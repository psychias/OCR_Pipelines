"""Apply random character-level noise to CSV columns.

Accompanies the papers:
- ACL 2025: "Cheap Character Noise for OCR-Robust Multilingual Embeddings"
- LREC 2026: "A Recipe for Adapting Multilingual Embedders to OCR-Error
              Robustness and Historical Texts"

Usage example::

    python generate_random_character_noise.py \
        --input_file ../noisy_finetuning_data/LREC/historical_articles_de.csv \
        --output_file ../noisy_finetuning_data/LREC/historical_articles_de_noised.csv \
        --target_columns de \
        --script latin \
        --noise_rate 0.05 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random

import pandas as pd

# ---------------------------------------------------------------------------
# Script-specific character pools
# ---------------------------------------------------------------------------
SCRIPT_CHARS: dict[str, str] = {
    "latin": (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
        "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ"
    ),
    "cyrillic": (
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    ),
    "arabic": "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
    "greek": (
        "αβγδεζηθικλμνξοπρστυφχψω"
        "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    ),
    "georgian": "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ",
    "hebrew": "אבגדהוזחטיכלמנסעפצקרשת",
}


def add_noise(text: str, noise_rate: float, char_pool: str, rng: random.Random) -> str:
    """Apply stochastic character-level perturbations to *text*.

    Each character is independently perturbed with probability *noise_rate*.
    The perturbation type is chosen uniformly from:
      - substitution  (replace with a random char from *char_pool*)
      - insertion     (insert a random char before the current one)
      - deletion      (drop the current character)
    """
    if not text:
        return text

    result: list[str] = []
    for ch in text:
        if rng.random() < noise_rate and ch.strip():
            op = rng.choice(["sub", "ins", "del"])
            if op == "sub":
                result.append(rng.choice(char_pool))
            elif op == "ins":
                result.append(rng.choice(char_pool))
                result.append(ch)
            # op == "del": skip character
        else:
            result.append(ch)
    return "".join(result)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply random character-level noise to a CSV column.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to write the output CSV file.",
    )
    parser.add_argument(
        "--target_columns",
        type=str,
        nargs="+",
        required=True,
        help="Column name(s) to apply noise to.",
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        choices=["latin", "cyrillic", "arabic", "greek", "georgian", "hebrew"],
        help="Writing system of the text — determines the noise character pool.",
    )
    parser.add_argument(
        "--noise_rate",
        type=float,
        default=0.05,
        help="Fraction of characters to perturb (default: 0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_04",
        help="Suffix appended to noised column names (default: '_04').",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    rng = random.Random(args.seed)
    char_pool = SCRIPT_CHARS[args.script]
    df = pd.read_csv(args.input_file)

    total_chars = 0
    total_changed = 0

    for col in args.target_columns:
        if col not in df.columns:
            msg = f"Column '{col}' not found in {args.input_file}."
            raise ValueError(msg)

        noised_col = f"{col}{args.output_suffix}"
        original_texts = df[col].astype(str).tolist()
        noised_texts: list[str] = []
        for text in original_texts:
            noised = add_noise(text, args.noise_rate, char_pool, rng)
            noised_texts.append(noised)
            total_chars += len(text)
            total_changed += sum(a != b for a, b in zip(text, noised))

        df[noised_col] = noised_texts

    df.to_csv(args.output_file, index=False)

    actual_rate = total_changed / total_chars if total_chars else 0.0
    print(f"Rows processed : {len(df)}")
    print(f"Columns noised : {args.target_columns}")
    print(f"Actual noise   : {actual_rate:.4f} ({actual_rate * 100:.2f} %)")
    print(f"Output written : {args.output_file}")


if __name__ == "__main__":
    main()
