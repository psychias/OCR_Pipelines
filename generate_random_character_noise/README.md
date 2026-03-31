# Generate Random Character Noise

Standalone CLI tool for applying stochastic character-level perturbations to CSV columns,
simulating OCR-style errors across multiple writing systems.

## What It Does

Given an input CSV, the script:

1. Reads one or more target columns
2. Applies random **substitution**, **insertion**, and **deletion** operations to a
   configurable fraction of characters
3. Writes the result as new `{column}_noisy` columns in the output CSV

The noise character pool is determined by the `--script` argument, ensuring that
replacement characters are visually plausible for the target writing system.

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_file` | Yes | — | Path to the input CSV file |
| `--output_file` | Yes | — | Path to write the output CSV |
| `--target_columns` | Yes | — | One or more column names to apply noise to |
| `--script` | Yes | — | Writing system: `latin`, `cyrillic`, `arabic`, `greek`, `georgian`, `hebrew` |
| `--noise_rate` | No | `0.05` | Fraction of characters to perturb (0.0–1.0) |
| `--seed` | No | `42` | Random seed for reproducibility |
| `--output_suffix` | No | `_noisy` | Suffix appended to create noisy column names |

## Supported Writing Systems

| Script | Character Pool |
|--------|---------------|
| `latin` | a–z, A–Z, plus accented Latin characters (à, é, ü, ñ, …) |
| `cyrillic` | а–я, А–Я, including ё |
| `arabic` | Standard Arabic alphabet (28 letters) |
| `greek` | α–ω, Α–Ω |
| `georgian` | Mkhedruli alphabet (33 letters) |
| `hebrew` | Standard Hebrew alphabet (22 letters) |

## Example Invocations

### Latin (German / French text)

```bash
python generate_random_character_noise/generate_random_character_noise.py \
    --input_file  noisy_finetuning_data/ACL/TED_data_random_noise.csv \
    --output_file /tmp/ted_noised.csv \
    --target_columns german french \
    --script latin \
    --noise_rate 0.05
```

### Cyrillic (Russian text)

```bash
python generate_random_character_noise/generate_random_character_noise.py \
    --input_file  data/russian_corpus.csv \
    --output_file /tmp/russian_noised.csv \
    --target_columns text \
    --script cyrillic \
    --noise_rate 0.10
```

### Custom suffix and seed

```bash
python generate_random_character_noise/generate_random_character_noise.py \
    --input_file  input.csv \
    --output_file output.csv \
    --target_columns src tgt \
    --script latin \
    --noise_rate 0.03 \
    --seed 123 \
    --output_suffix _ocr_noise
```

This produces columns `src_ocr_noise` and `tgt_ocr_noise`.

## Output Column Naming Convention

For each target column `COL`, the script creates `COL{suffix}` (default: `COL_noisy`).

This maps directly to the dataset schema used throughout the repository:

- **Monolingual denoising pairs**: `de` → `de_noisy`, `fr` → `fr_noisy`
- **Evaluation datasets**: `src` → `src_noisy`, `tgt` → `tgt_noisy`

## References

- **ACL 2025**: *Cheap Character Noise for OCR-Robust Multilingual Embeddings*
- **LREC 2026**: *A Recipe for Adapting Multilingual Embedders to OCR-Error Robustness
  and Historical Texts*
