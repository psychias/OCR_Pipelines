# Noisy Evaluation Datasets — ACL 2025

This folder contains the noised evaluation datasets produced for:

> **ACL 2025**: *Cheap Character Noise for OCR-Robust Multilingual Embeddings*
> ([paper](https://aclanthology.org/2025.findings-acl.609/))

## Contents

| File | Description |
|------|-------------|
| `CLSD_WMT19_blackletter_scanned_noise.csv` | CLSD WMT 2019 with blackletter/scanned character noise |
| `CLSD_WMT19_salt_pepper_noise.csv` | CLSD WMT 2019 with salt-and-pepper character noise |
| `CLSD_WMT21_blackletter_scanned_noise.csv` | CLSD WMT 2021 with blackletter/scanned character noise |
| `CLSD_WMT21_salt_pepper_noise.csv` | CLSD WMT 2021 with salt-and-pepper character noise |

## Column Schema

```
id, src_lang, tgt_lang, src, tgt, src_04, tgt_04, label
```

- `id` — unique row identifier
- `src_lang` / `tgt_lang` — ISO 639-1 language codes (e.g. `de`, `fr`)
- `src` / `tgt` — clean source and target sentences
- `src_04` / `tgt_04` — noised versions at ~4% character error rate
- `label` — `1` for true parallel pair, `0` for distractor

## Languages

- German (`de`) ↔ French (`fr`)

## Source Corpus

Derived from the WMT 2019 and WMT 2021 shared translation tasks using the CLSD (Cross-Lingual Sentence Detection) benchmark format.

## Regeneration

The noised columns were generated using the character noise utilities in
`generate_random_character_noise/` (for random noise) and `ocr_simulator/`
(for realistic OCR noise).
