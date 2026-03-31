# Noisy Fine-tuning Data — ACL 2025

This folder contains the noised training data produced for:

> **ACL 2025**: *Cheap Character Noise for OCR-Robust Multilingual Embeddings*
> ([paper](https://aclanthology.org/2025.findings-acl.609/))

## Contents

| File | Language | Source Corpus | Description |
|------|----------|---------------|-------------|
| `TED_de.csv` | German (de) | TED Talks | Denoising pairs from TED parallel corpus |
| `TED_fr.csv` | French (fr) | TED Talks | Denoising pairs from TED parallel corpus |
| `xnews_de_fr.csv` | German (de) | X-News | Denoising pairs from X-News parallel corpus |

## Column Schema

**Monolingual denoising pairs:**
```
lang, lang_04
```

- `lang` — clean text (column named by ISO 639-1 code, e.g. `de`, `fr`)
- `lang_04` — same text with ~4% random character-level noise applied

### Example (German)

```csv
de,de_04
"Die Zukunft der Technologie liegt in der künstlichen Intelligenz.","Die Zukunft der TechnoloXie liegt in der künstlichen Intelligenz."
```

## Regeneration

Noise was applied using:

```bash
python generate_random_character_noise/generate_random_character_noise.py \
    --input_file noisy_finetuning_data/ACL/TED_de.csv \
    --output_file noisy_finetuning_data/ACL/TED_de_noised.csv \
    --target_columns de \
    --script latin \
    --noise_rate 0.05 \
    --seed 42
```
