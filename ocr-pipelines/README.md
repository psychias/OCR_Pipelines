# A Recipe for Adapting Multilingual Embedders to OCR-Error Robustness and Historical Texts

[![LREC 2026](https://img.shields.io/badge/LREC-2026-blue.svg)](https://lrec-coling-2026.org/)
[![ACL 2025](https://img.shields.io/badge/ACL-2025-green.svg)](https://2025.aclweb.org/)
[![License: AGPLV3+](https://img.shields.io/badge/License-AGPLV3+-brightgreen.svg)](LICENSE)

## Overview

This repository accompanies **two** research papers from the [Impresso](https://impresso-project.ch/) project:

1. **LREC 2026**: *A Recipe for Adapting Multilingual Embedders to OCR-Error Robustness and Historical Texts*
2. **ACL 2025**: *Cheap Character Noise for OCR-Robust Multilingual Embeddings*

It provides the **full two-step fine-tuning pipeline**, noisy training and evaluation datasets, and utilities for simulating random character-level OCR noise across multiple writing systems.

The canonical ACL 2025 repository is: [impresso/ocr-robust-multilingual-embeddings](https://github.com/impresso/ocr-robust-multilingual-embeddings). This repo extends it with new datasets, languages (including Luxembourgish historical/modern cross-lingual pairs), and the generalised training recipe from the LREC 2026 paper.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Models](#models)
- [Citation](#citation)
- [License](#license)
- [About Impresso](#about-impresso)

---

## Repository Structure

```
ocr-pipelines/
│
├── noisy_evaluation_datasets/
│   └── ACL/                          # Noised CLSD WMT evaluation datasets (ACL 2025)
│
├── noisy_finetuning_data/
│   ├── ACL/                          # TED and X-News denoising pairs (ACL 2025)
│   └── LREC/                         # Historical articles, MLSum, Luxembourgish (LREC 2026)
│
├── generate_random_character_noise/   # CLI script for script-aware character noise
│   ├── generate_random_character_noise.py
│   └── README.md
│
├── ocr_simulator/                     # OCR simulator library (pointer to ACL repo)
│   └── README.md
│
├── train_and_evaluate.py              # Main training & evaluation script
├── train_and_evaluate.ipynb           # Google Colab-ready notebook version
├── LICENSE                            # AGPL-3.0
└── README.md                          # This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -U transformers==4.46.3
pip install -U "sentence-transformers>=3.0"
pip install -U "peft>=0.7.0"
pip install "datasets>=3.0"
pip install pandas
```

### 2. Generate noisy data

```bash
cd generate_random_character_noise/

python generate_random_character_noise.py \
    --input_file ../noisy_finetuning_data/LREC/historical_articles_de.csv \
    --output_file ../noisy_finetuning_data/LREC/historical_articles_de_noised.csv \
    --target_columns de \
    --script latin \
    --noise_rate 0.05 \
    --seed 42
```

Supported writing systems: `latin`, `cyrillic`, `arabic`, `greek`, `georgian`, `hebrew`.

### 3. Train and evaluate

```bash
# From the repo root
python train_and_evaluate.py
```

Or open `train_and_evaluate.ipynb` in Google Colab for a step-by-step walkthrough.

---

## Datasets

### ACL 2025 vs LREC 2026 split

| Folder | Paper | Contents |
|--------|-------|----------|
| `noisy_evaluation_datasets/ACL/` | ACL 2025 | Noised CLSD WMT19/21 evaluation sets (blackletter-scanned & salt-pepper noise) |
| `noisy_finetuning_data/ACL/` | ACL 2025 | TED and X-News denoising pairs (DE, FR) |
| `noisy_finetuning_data/LREC/` | LREC 2026 | Historical articles (DE, FR), MLSum (DE, ES, FR, RU, TR), Luxembourgish cross-lingual pairs |

Data is **not duplicated** across folders — if a resource belongs to the ACL paper, it lives only under `ACL/`.

### Column schema

**Monolingual denoising pairs** (fine-tuning):
```
lang, lang_04
```
- `lang` — clean text in the given language (ISO 639-1 code as column name)
- `lang_04` — same text with ~4% random character-level noise

**Cross-lingual pairs** (Luxembourgish files):
```
src_lang, tgt_lang, src, tgt
```
- `src_lang` / `tgt_lang` — ISO 639-3 codes (e.g. `ltz`, `deu`, `fra`, `eng`)

**Evaluation data** (noisy CLSD):
```
id, src_lang, tgt_lang, src, tgt, src_04, tgt_04, label
```
- `src_04` / `tgt_04` — noised versions at ~4% character error rate
- `label` — 1 for true parallel pair, 0 for distractor

---

## Models

The following fine-tuned models are available on HuggingFace:

| Model | Paper | Description |
|-------|-------|-------------|
| [`impresso-project/OCR-robust-gte-multilingual-base`](https://huggingface.co/impresso-project/OCR-robust-gte-multilingual-base) | ACL 2025 | OCR-robust GTE base (random noise fine-tuned) |
| [`impresso-project/gte-multilingual-base-ocr-noise-robust`](https://huggingface.co/impresso-project/gte-multilingual-base-ocr-noise-robust) | LREC 2026 | Generalist OCR-noise-robust model |
| [`impresso-project/gte-multilingual-base-histlux-ocr-noise-robust`](https://huggingface.co/impresso-project/gte-multilingual-base-histlux-ocr-noise-robust) | LREC 2026 | Specialist model with historical Luxembourgish adaptation |

---

## Citation

If you use these resources, please cite the relevant paper(s):

### LREC 2026

```bibtex
@inproceedings{michail-etal-2026-recipe,
    title     = "A Recipe for Adapting Multilingual Embedders to {OCR}-Error
                 Robustness and Historical Texts",
    author    = "Michail, Andrianos and Opitz, Juri and Wang, Yining and
                 Meister, Robin and Sennrich, Rico and Clematide, Simon",
    booktitle = "Proceedings of the 2026 Joint International Conference on
                 Computational Linguistics, Language Resources and Evaluation
                 (LREC-COLING 2026)",
    year      = "2026",
}
```

### ACL 2025

```bibtex
@inproceedings{michail-etal-2025-cheap,
    title     = "Cheap Character Noise for {OCR}-Robust Multilingual Embeddings",
    author    = "Michail, Andrianos and Opitz, Juri and Wang, Yining and
                 Meister, Robin and Sennrich, Rico and Clematide, Simon",
    editor    = "Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and
                 Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics:
                 ACL 2025",
    month     = jul,
    year      = "2025",
    address   = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2025.findings-acl.609/",
    pages     = "11705--11716",
    ISBN      = "979-8-89176-256-5",
}
```

---

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) or later.

---

## About Impresso

[Impresso – Media Monitoring of the Past](https://impresso-project.ch/) is an interdisciplinary research project that aims to develop and consolidate tools for processing and exploring large collections of media archives across modalities, time, languages and national borders. The first project (2017–2021) was funded by the Swiss National Science Foundation under grant No. [CRSII5_173719](http://p3.snf.ch/project-173719) and the second project (2023–2027) by the SNSF under grant No. [CRSII5_213585](https://data.snf.ch/grants/grant/213585) and the Luxembourg National Research Fund under grant No. 17498891.

Copyright (C) 2025–2026 The Impresso team.
