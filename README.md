# OCR-Robust Multilingual Embeddings

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/impresso-project)

## Overview

This repository accompanies two papers on making multilingual sentence
embeddings robust to OCR noise. It provides noisy datasets, fine-tuned models,
an OCR-noise simulation toolkit, and a sample training notebook that
demonstrates the two-stage denoising fine-tuning pipeline (cross-lingual
alignment followed by character-noise robustness training).

## Papers

| Venue | Title | Focus |
|-------|-------|-------|
| **ACL 2025 Findings** | [*Cheap Character Noise for OCR-Robust Multilingual Embeddings*](https://aclanthology.org/2025.findings-acl.609/) | CLSD benchmark (French ↔ German) with BLDS / MN / SNP noise |
| **LREC 2026** | *A Recipe for Adapting Multilingual Embedders to OCR-Error Robustness and Historical Texts* | Luxembourgish alignment + historical bitext mining |

## Models

| Model | Base | Description |
|-------|------|-------------|
| [`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | — | Baseline GTE encoder |
| [`impresso-project/histlux-gte-multilingual-base`](https://huggingface.co/impresso-project/histlux-gte-multilingual-base) | GTE | Fine-tuned on historical + Luxembourgish data |

## Repository structure

```
├── ACL/
│   ├── noisy_evaluation_datasets/       # 8 CLSD CSVs (WMT 2019 & 2021)
│   └── noisy_finetuning_data/           # 4 TED / X-News CSVs
├── LREC/
│   ├── noisy_evaluation_datasets/       # 6 HistBIM JSONLs + 4 X-STS17 CSVs
│   ├── noisy_finetuning_data/           # 3 historical-article CSVs
│   └── luxembourgish_dataset/           # 3 parallel JSONLs (≈120 k pairs)
├── generate_random_character_noise/     # Synthetic OCR noise CLI (6 scripts)
├── ocr_simulator/                       # OCR simulation library (impresso)
├── sample_training.ipynb                # End-to-end: eval → train → eval
├── requirements.txt                     # Pinned Python dependencies
└── LICENSE                              # AGPL-3.0
```

## Column naming convention

All data files follow a consistent column scheme:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `{lang3}` | Clean text (ISO 639-3) | `deu`, `fra`, `eng`, `ltz` |
| `{lang3}_04` | Noised text (CER ≈ 0.04) | `deu_04`, `fra_04` |

Additional noise-rate variants (`deu_10`, `deu_15`) and adversarial columns
(`de_adv2`–`de_adv4`) are retained where present.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate OCR noise for a CSV
python generate_random_character_noise/generate_random_character_noise.py \
    data.csv --columns deu fra --script latin --cer 0.04 -o noised.csv

# Run the sample training notebook
jupyter notebook sample_training.ipynb
```

## Citation

If you use these resources, please cite:

```bibtex
@inproceedings{michail-etal-2025-cheap,
    title     = {Cheap Character Noise for {OCR}-Robust Multilingual Embeddings},
    author    = {Michail, Andrianos and Opitz, Juri and Wang, Yining and
                 Meister, Robin and Sennrich, Rico and Clematide, Simon},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
    year      = {2025},
    url       = {https://aclanthology.org/2025.findings-acl.609/},
    pages     = {11705--11716}
}

@inproceedings{michail-etal-2026-recipe,
    title     = {A Recipe for Adapting Multilingual Embedders to {OCR}-Error
                 Robustness and Historical Texts},
    author    = {Michail, Andrianos and Opitz, Juri and Racl{\'e}, Corina and
                 Sennrich, Rico and Clematide, Simon},
    booktitle = {Proceedings of the 15th Language Resources and Evaluation
                 Conference (LREC 2026)},
    year      = {2026},
    publisher = {European Language Resources Association}
}
```

## License

AGPL-3.0. See [LICENSE](LICENSE).

Copyright (C) 2025 The [Impresso](https://impresso-project.ch/) team.
