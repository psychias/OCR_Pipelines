# Noisy Fine-tuning Data — LREC 2026

This folder contains the new training data introduced in:

> **LREC 2026**: *A Recipe for Adapting Multilingual Embedders to OCR-Error Robustness and Historical Texts*

All data in this folder is **new** to the LREC paper — data from the ACL 2025 paper lives exclusively under `../ACL/`.

## Contents

### Historical Articles (monolingual denoising pairs)

| File | Language | Source |
|------|----------|--------|
| `historical_articles_de.csv` | German (de) | Digitised historical newspaper articles |
| `historical_articles_fr.csv` | French (fr) | Digitised historical newspaper articles |

### MLSum Articles (monolingual denoising pairs)

| File | Language |
|------|----------|
| `mlsum_articles_de.csv` | German (de) |
| `mlsum_articles_es.csv` | Spanish (es) |
| `mlsum_articles_fr.csv` | French (fr) |
| `mlsum_articles_ru.csv` | Russian (ru) |
| `mlsum_articles_tr.csv` | Turkish (tr) |

### MLSum Summaries (monolingual denoising pairs)

| File | Language |
|------|----------|
| `mlsum_summaries_de.csv` | German (de) |
| `mlsum_summaries_es.csv` | Spanish (es) |
| `mlsum_summaries_fr.csv` | French (fr) |
| `mlsum_summaries_ru.csv` | Russian (ru) |
| `mlsum_summaries_tr.csv` | Turkish (tr) |

### Luxembourgish Cross-lingual Pairs

| File | Description |
|------|-------------|
| `luxembourgish_historical_crosslingual.csv` | Historical Luxembourgish ↔ DE/FR/EN parallel sentences |
| `luxembourgish_modern_crosslingual.csv` | Modern Luxembourgish ↔ DE/FR/EN parallel sentences |

## Column Schema

### Monolingual denoising pairs

```
lang, lang_04
```

- `lang` — clean text (column named by ISO 639-1 code)
- `lang_04` — same text with ~4% random character-level noise

For Russian (`ru`) files, noise is drawn from the Cyrillic character pool.
For Turkish (`tr`) files, noise is drawn from the Latin character pool.

### Cross-lingual pairs (Luxembourgish files)

```
src_lang, tgt_lang, src, tgt
```

- `src_lang` / `tgt_lang` — ISO 639-3 three-letter codes (e.g. `ltz`, `deu`, `fra`, `eng`)
- `src` — source sentence
- `tgt` — target sentence (parallel translation)

## Regeneration

Monolingual denoising noise was applied using:

```bash
# Example: German historical articles with Latin script noise
python generate_random_character_noise/generate_random_character_noise.py \
    --input_file noisy_finetuning_data/LREC/historical_articles_de.csv \
    --output_file noisy_finetuning_data/LREC/historical_articles_de_noised.csv \
    --target_columns de \
    --script latin \
    --noise_rate 0.05 \
    --seed 42

# Example: Russian MLSum articles with Cyrillic script noise
python generate_random_character_noise/generate_random_character_noise.py \
    --input_file noisy_finetuning_data/LREC/mlsum_articles_ru.csv \
    --output_file noisy_finetuning_data/LREC/mlsum_articles_ru_noised.csv \
    --target_columns ru \
    --script cyrillic \
    --noise_rate 0.05 \
    --seed 42
```

The cross-lingual Luxembourgish files are pre-aligned parallel corpora and do not require noise generation.
