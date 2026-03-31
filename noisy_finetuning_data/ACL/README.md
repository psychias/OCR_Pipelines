# Noisy Fine-Tuning Data — ACL 2025

This directory contains the **monolingual denoising training pairs** used in the ACL 2025 paper.

## Files

| File | Language(s) | Source | Description |
|------|------------|--------|-------------|
| `TED_data_random_noise.csv` | DE, FR | TED Talks | Random character noise at 5/10/15 % |
| `TED_data_random_noise_concat.csv` | DE | TED Talks | Concatenated sentences with 5 % noise |
| `TED_data_realistic_noise.csv` | DE, FR | TED Talks | Realistic OCR noise (BLDS + SNP) |
| `X-News_data_random_noise.csv` | DE, FR | X-News | Random character noise at 5/10/15 % |

## Column Schema (after normalisation)

Monolingual denoising pairs:

| Column | Description |
|--------|-------------|
| `de` / `fr` | Clean text (ISO 639-1 language code) |
| `de_noisy` / `fr_noisy` | Noisy variant (~5 % character perturbation) |

Original columns are preserved alongside the standardised ones.

Run `python noisy_finetuning_data/normalise_columns.py` to add standardised columns.
