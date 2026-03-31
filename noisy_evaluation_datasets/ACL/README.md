# Noisy Evaluation Datasets — ACL 2025

This directory contains the **noisy CLSD (Cross-Lingual Semantic Discrimination)** evaluation
datasets used in our ACL 2025 paper.

Each CSV is a French–German parallel corpus from WMT (2019 or 2021) with OCR-style noise
applied to both source and target sentences.

## Files

| File | WMT Year | Noise Type |
|------|----------|------------|
| `CLSD_WMT19_BLDS_noise.csv` | 2019 | BlackLetter / Digitised Scanned |
| `CLSD_WMT21_BLDS_noise.csv` | 2021 | BlackLetter / Digitised Scanned |
| `CLSD_WMT19_SNP_noise.csv`  | 2019 | Salt-and-Pepper |
| `CLSD_WMT21_SNP_noise.csv`  | 2021 | Salt-and-Pepper |
| `CLSD_WMT19_MN_noise.csv`   | 2019 | Mixed Noise |
| `CLSD_WMT21_MN_noise.csv`   | 2021 | Mixed Noise |

## Column Schema

**Original columns** (preserved for backward compatibility):

| Column | Description |
|--------|-------------|
| `French` | Clean French sentence |
| `German` | Clean German sentence |
| `de_adv1`–`de_adv4` | Four adversarial (noisy) German variants |
| `fr_adv1`–`fr_adv4` | Four adversarial (noisy) French variants |

**Standardised columns** (added by `normalise_columns.py`):

| Column | Description |
|--------|-------------|
| `id` | Row identifier |
| `src` | Source sentence (= `French`) |
| `tgt` | Target sentence (= `German`) |
| `src_lang` | Source language code (`fra`) |
| `tgt_lang` | Target language code (`deu`) |
| `src_noisy` | Noisy source (= `fr_adv1`) |
| `tgt_noisy` | Noisy target (= `de_adv1`) |

## Normalisation

Run the normalisation script to add the standardised columns:

```bash
python noisy_evaluation_datasets/ACL/normalise_columns.py
```

The script is idempotent — safe to run multiple times.
