# OCR M-GTE: Multilingual Embeddings for OCR-Robust and Historical Text Retrieval

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/impresso-project)

## Overview

Modern multilingual text embedding models perform well on clean contemporary text but degrade significantly on digitised historical documents, where OCR-induced character errors (substitutions, insertions, deletions) and archaic orthography distort the input signal. This degradation is especially pronounced for underrepresented languages such as Luxembourgish, where historical materials combine evolving spelling conventions with OCR artifacts absent from standard training data.

This repository accompanies two papers that address this problem. Our **ACL 2025** paper (*Cheap Character Noise for OCR-Robust Multilingual Embeddings*) showed that fine-tuning with simple random character noise already yields measurable gains on cross-lingual semantic discrimination under OCR degradation. Our **LREC 2026** paper (*A Recipe for Adapting Multilingual Embedders to OCR-Error Robustness and Historical Texts*) extends this approach with a two-step procedure — first acquiring historical Luxembourgish through cross-lingual alignment (Task A), then adapting to OCR noise across six European languages (Task B) — and demonstrates state-of-the-art results on both OCR-robust retrieval and historical bitext mining.

## Repository Structure

```
OCR_Pipelines/
│
├── noisy_evaluation_datasets/
│   └── ACL/                              # Noisy CLSD evaluation CSVs (WMT19/21, 3 noise types)
│
├── noisy_finetuning_data/
│   ├── ACL/                              # TED and X-News denoising pairs (DE, FR)
│   └── LREC/                             # Historical Articles, MLSum, Luxembourgish cross-lingual
│
├── generate_random_character_noise/
│   ├── generate_random_character_noise.py # Standalone multi-script OCR noise CLI
│   └── README.md
│
├── src/
│   ├── evaluation_sets/                  # Clean evaluation data + bitext mining JSONLs
│   ├── evaluations_scripts/              # Per-model/noise-type evaluation scripts (see README.md inside)
│   ├── finetuning/                       # Core training script + legacy noise generation
│   ├── finetuning_data/                  # Working training data directory
│   └── prepared_bitext_mining_format/    # Bitext mining format JSONLs
│
├── train_and_evaluate.py                 # Canonical LREC 2026 two-step training pipeline
├── train_and_evaluate.ipynb              # Colab notebook version of the above
├── train_mix_model.ipynb                 # Original Colab training notebook (ACL 2025)
├── requirements.txt
├── requirements_ubuntu.txt
├── run_all_scripts.sh
├── run_evaluations_both_models.sh
├── run_experiments.sh
├── LICENSE                               # AGPL-3.0
└── README.md
```

## Models

| Model | HuggingFace ID | Description |
|-------|---------------|-------------|
| OCR-robust generalist (ACL) | [`impresso-project/OCR-robust-gte-multilingual-base`](https://huggingface.co/impresso-project/OCR-robust-gte-multilingual-base) | Task B only — denoising fine-tuning on TED + X-News |
| OCR-robust generalist (LREC) | [`impresso-project/gte-multilingual-base-ocr-noise-robust`](https://huggingface.co/impresso-project/gte-multilingual-base-ocr-noise-robust) | Task A then B — full two-step procedure |
| LUX specialist (LREC) | [`impresso-project/gte-multilingual-base-histlux-ocr-noise-robust`](https://huggingface.co/impresso-project/gte-multilingual-base-histlux-ocr-noise-robust) | Task A then B, optimised for historical Luxembourgish |

All models are based on [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base).

## Datasets

Training and evaluation data is split across two directories reflecting the two papers:

### ACL 2025 (`noisy_evaluation_datasets/ACL/` and `noisy_finetuning_data/ACL/`)

- **Evaluation**: CLSD (Cross-Lingual Semantic Discrimination) test sets derived from WMT 2019 and 2021, with three noise types (BlackLetter/Scanned, Salt-and-Pepper, Mixed).
- **Training**: TED talk transcripts and X-News articles in German and French, paired with random character noise at ~5 %.

### LREC 2026 (`noisy_finetuning_data/LREC/`)

- **Historical Articles**: German and French newspaper articles from digitised 19th–20th century archives.
- **MLSum**: Multilingual article–summary pairs (DE, ES, FR, RU, TR) with character noise.
- **Luxembourgish cross-lingual**: Historical and modern Luxembourgish paired with German, French, and English translations.

### Column Schema

**Monolingual denoising pairs** (TED, X-News, Historical Articles, MLSum):

| Column | Description |
|--------|-------------|
| `<lang>` | Clean text (ISO 639-1 code, e.g. `de`, `fr`) |
| `<lang>_noisy` | Same text with ~5 % random character noise |

**Cross-lingual pairs** (Luxembourgish files):

| Column | Description |
|--------|-------------|
| `src_lang` | Source language (ISO 639-3, e.g. `ltz`) |
| `tgt_lang` | Target language (ISO 639-3, e.g. `deu`, `fra`, `eng`) |
| `src` | Source text |
| `tgt` | Target text |

**Evaluation datasets** (CLSD):

| Column | Description |
|--------|-------------|
| `French` / `German` | Clean parallel sentences |
| `fr_adv1`–`fr_adv4` / `de_adv1`–`de_adv4` | Noisy adversarial variants |
| `src` / `tgt` / `src_noisy` / `tgt_noisy` | Standardised aliases (added by `normalise_columns.py`) |

Run normalisation scripts to add standardised columns:

```bash
python noisy_evaluation_datasets/ACL/normalise_columns.py
python noisy_finetuning_data/normalise_columns.py
```

## Quick Start

### Generate OCR Noise

```bash
python generate_random_character_noise/generate_random_character_noise.py \
    --input_file  noisy_finetuning_data/ACL/TED_data_random_noise.csv \
    --output_file /tmp/ted_noised.csv \
    --target_columns german french \
    --script latin \
    --noise_rate 0.05
```

See [`generate_random_character_noise/README.md`](generate_random_character_noise/README.md) for full CLI reference and examples for all supported writing systems.

### Run the Sample Training Pipeline

```bash
# Two-step fine-tuning (Task A: Luxembourgish, Task B: OCR noise)
python train_and_evaluate.py
```

Or open [`train_and_evaluate.ipynb`](train_and_evaluate.ipynb) in Google Colab.

### Evaluation Scripts

Individual evaluation scripts for each model/noise-type combination are in [`src/evaluations_scripts/`](src/evaluations_scripts/README.md).

```bash
# Example: CLSD evaluation with BlackLetter noise on GTE mono model
python src/evaluations_scripts/CLSD_wmt_evaluation_gte_mono_BL.py
```

## Citation

If you use this code, datasets, or models, please cite:

```bibtex
@inproceedings{michail2026recipe,
    title={A Recipe for Adapting Multilingual Embedders to {OCR}-Error Robustness and Historical Texts},
    author={Michail, Andrianos and Opitz, Juri and Racl{\'e}, Corina and Sennrich, Rico and Clematide, Simon},
    booktitle={Proceedings of the 15th Language Resources and Evaluation Conference (LREC 2026)},
    year={2026},
    publisher={European Language Resources Association}
}

@inproceedings{michail2025cheap,
    title={Cheap Character Noise for {OCR}-Robust Multilingual Embeddings},
    author={Michail, Andrianos and Opitz, Juri and Wang, Yining and Meister, Robin and Sennrich, Rico and Clematide, Simon},
    booktitle={Findings of ACL 2025},
    year={2025}
}
```

## License

AGPL-3.0. See [LICENSE](LICENSE).

## Acknowledgements

- [Impresso — Media Monitoring of the Past II](https://impresso-project.ch/) (Swiss National Science Foundation, Sinergia grant CRSII5_213585)
- [Department of Computational Linguistics, University of Zurich](https://www.cl.uzh.ch/)
- Base model: [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) and [mGTE](https://arxiv.org/abs/2407.19669)
- Training framework: [Sentence-Transformers](https://www.sbert.net/)
