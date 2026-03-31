# Generate Random Character Noise

Script-aware random character-level noise generator for creating OCR-like perturbations in text data.

## Paper

This tool accompanies:
- **ACL 2025**: *Cheap Character Noise for OCR-Robust Multilingual Embeddings*
- **LREC 2026**: *A Recipe for Adapting Multilingual Embedders to OCR-Error Robustness and Historical Texts*

## Description

`generate_random_character_noise.py` applies stochastic character-level perturbations (substitution, insertion, deletion) to text columns in a CSV file. The perturbation alphabet is determined by the specified writing system.

### Supported writing systems

| Script | Languages (examples) |
|--------|---------------------|
| `latin` | German, French, English, Spanish, Turkish, etc. |
| `cyrillic` | Russian, Ukrainian, Bulgarian, etc. |
| `arabic` | Arabic, etc. |
| `greek` | Greek |
| `georgian` | Georgian |
| `hebrew` | Hebrew |

## Usage

```bash
python generate_random_character_noise.py \
    --input_file ../noisy_finetuning_data/LREC/historical_articles_de.csv \
    --output_file ../noisy_finetuning_data/LREC/historical_articles_de_noised.csv \
    --target_columns de \
    --script latin \
    --noise_rate 0.05 \
    --seed 42
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_file` | Yes | — | Path to input CSV file |
| `--output_file` | Yes | — | Path to write the output CSV file |
| `--target_columns` | Yes | — | Column name(s) to apply noise to |
| `--script` | Yes | — | Writing system (`latin`, `cyrillic`, `arabic`, `greek`, `georgian`, `hebrew`) |
| `--noise_rate` | No | 0.05 | Fraction of characters to perturb |
| `--seed` | No | 42 | Random seed for reproducibility |
| `--output_suffix` | No | `_04` | Suffix appended to noised column names |

### Output

- Original columns are preserved
- For each target column `col`, a new column `col_04` (or `col{suffix}`) is added with noised text
- The full dataframe is written to `--output_file`
- A summary is printed: rows processed, columns noised, actual noise rate achieved

## Dependencies

```
pandas
```
