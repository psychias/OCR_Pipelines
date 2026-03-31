# Noisy Fine-Tuning Data — LREC 2026

This directory contains the training data for the LREC 2026 paper, including both
monolingual denoising pairs and cross-lingual alignment pairs.

## Files

### Historical Articles (monolingual denoising)

| File | Language | Description |
|------|----------|-------------|
| `de_docs_random_noise.csv` | DE | Historical German newspaper articles with random character noise |
| `fr_docs_random_noise.csv` | FR | Historical French newspaper articles with random character noise |

### MLSum (monolingual denoising)

| File | Language | Description |
|------|----------|-------------|
| `query_doc_dataset_random_noise.csv` | DE | MLSum articles/summaries with random character noise |

### Luxembourgish Cross-Lingual Pairs

| File | Pair | Description |
|------|------|-------------|
| `lb_de_training_set.jsonl` | LB ↔ DE | Historical Luxembourgish–German parallel pairs |
| `lb_en_training_set.jsonl` | LB ↔ EN | Historical Luxembourgish–English parallel pairs |
| `lb_fr_training_set.jsonl` | LB ↔ FR | Historical Luxembourgish–French parallel pairs |

## Column Schema (after normalisation)

**Monolingual denoising pairs:**

| Column | Description |
|--------|-------------|
| `de` / `fr` | Clean text (ISO 639-1 language code) |
| `de_noisy` / `fr_noisy` | Noisy variant |

**Cross-lingual pairs** (generated CSV from JSONL):

| Column | Description |
|--------|-------------|
| `src_lang` | Source language (ISO 639-3: `ltz`) |
| `tgt_lang` | Target language (ISO 639-3: `deu`, `eng`, `fra`) |
| `src` | Source text (Luxembourgish) |
| `tgt` | Target text |

Run `python noisy_finetuning_data/normalise_columns.py` to standardise all columns.
