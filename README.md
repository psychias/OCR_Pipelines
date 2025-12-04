# OCR M-GTE: Multilingual Embeddings for OCR-Robust and Historical Text Retrieval

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/impresso-project)

This repository contains the code, datasets, and training scripts for **OCR M-GTE**, a family of multilingual embedding models adapted for robust semantic search in OCR-affected and historical texts.

## Overview

Modern multilingual text embedding models excel at semantic search on contemporary text but degrade measurably on digitized historical documents. This degradation is especially pronounced for underrepresented languages such as Luxembourgish, where historical materials combine evolving spelling conventions with OCR artifacts absent from standard training data.

We introduce a simple multi-step training procedure that adapts multilingual embedders to handle:
- **OCR-induced noise** (character substitutions, insertions, deletions)
- **Historical text variants** (evolving orthography, archaic spelling)
- **Cross-lingual retrieval** in heterogeneous European digitized corpora

## Released Models

We release two models on Hugging Face:

| Model | Description | Best For |
|-------|-------------|----------|
| [`halloween_workshop_ocr_robust_preview`](https://huggingface.co/impresso-project/halloween_workshop_ocr_robust_preview) | Denoising-trained generalist | Multilingual retrieval with OCR robustness |
| [`halloween_workshop_ocr_robust_with_lux_preview`](https://huggingface.co/impresso-project/halloween_workshop_ocr_robust_with_lux_preview) | Historical Luxembourgish specialist | OCR-rich archives, historical European texts |

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous_github/OCR_Pipelines.git
cd OCR_Pipelines

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
sentence-transformers>=2.2.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

## Quick Start

### Loading the Models

```python
from sentence_transformers import SentenceTransformer

# Load the OCR-robust generalist model
model = SentenceTransformer('impresso-project/halloween_workshop_ocr_robust_preview')

# Or load the Historical Luxembourgish specialist (requires trust_remote_code=True)
model = SentenceTransformer('impresso-project/halloween_workshop_ocr_robust_with_lux_preview', trust_remote_code=True)
```

### Encoding Text

```python
# Encode sentences (handles both clean and OCR-affected text)
sentences = [
    "The quick brown fox jumps over the lazy dog.",  # Clean text
    "Tlie quicK br0wn fox jurnps over tbe lazy d0g.",  # OCR-affected text
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (2, 768)
```

### Semantic Search Example

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('impresso-project/halloween_workshop_ocr_robust_preview')

# Query (clean modern text)
query = "historical newspaper archives"

# Corpus with OCR noise (simulating digitized documents)
corpus = [
    "Historische Zeitungsarchive aus dem 19. Jahrhundert",  # German
    "Hist0rische Zeitungsarchive aus dern 19. Jalirhundert",  # German with OCR errors
    "Archives de journaux historiques du XIXe siècle",  # French
    "Arcliives de journaux liistoriques du XlXe siècle",  # French with OCR errors
]

# Encode
query_embedding = model.encode(query, convert_to_tensor=True)
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Compute similarity
similarities = util.cos_sim(query_embedding, corpus_embeddings)
print(similarities)
```

## Training

Our adaptation follows a two-step fine-tuning procedure using the scripts in `src/finetuning/`.

### Step 1: Historical Luxembourgish Adaptation (Task A)

Cross-lingual training with historical Luxembourgish paired with modern German, French, and English translations.

```bash
python src/finetuning/finetuning_no_sampler_args.py \
    --base_model "Alibaba-NLP/gte-multilingual-base" \
    --train_data src/finetuning_data/ \
    --output_dir models/ocr_robust_with_lux/ \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 2e-5
```

### Step 2: OCR-Error Noise Adaptation (Task B)

Monolingual denoising training across six European languages. Use `src/finetuning/generate_random_noise/` to create noisy training pairs.

```bash
python src/finetuning/finetuning_no_sampler_args.py \
    --base_model models/ocr_robust_with_lux/ \
    --train_data src/finetuning_data/ \
    --output_dir models/ocr_robust/ \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 2e-5
```

### Full Training Pipeline

```bash
# Run complete two-step training
python src/finetuning/finetuning_no_sampler_args.py \
    --config configs/full_training.yaml \
    --seed 42
```

## Dataset Preparation

### Training Data Overview

| Source | Type | Avg. Length | Languages | Pairs |
|--------|------|-------------|-----------|-------|
| Historical Luxembourgish | Cross-lingual | 13 tokens | LB ↔ DE/FR/EN | 60,000 |
| Modern Luxembourgish | Cross-lingual | 18 tokens | LB ↔ FR/EN | 60,000 |
| TED | Monolingual | 16 tokens | DE, FR | 20,000 |
| Historical Articles | Monolingual | 315 tokens | DE, FR | 20,000 |
| MLSum Articles | Monolingual | 550 tokens | DE, ES, FR, RU, TR | 10,000 |
| MLSum Summaries | Monolingual | 20 tokens | DE, ES, FR, RU, TR | 10,000 |

### Generating OCR Noise

We apply random character-level perturbations to 5% of characters. Noise generation utilities are in `src/finetuning/generate_random_noise/`:

```python
from src.finetuning.generate_random_noise import apply_ocr_noise

clean_text = "The historical document was digitized in 2020."
noisy_text = apply_ocr_noise(clean_text, noise_rate=0.05)
print(noisy_text)
# Example output: "Tlie liistorical docurnent was digitized in 2O2O."
```

### Preparing Training Pairs

Training data should be placed in `src/finetuning_data/`. The prepared bitext mining format data is stored in `src/prepared_bitext_mining_format/`.

```python
from src.finetuning.generate_random_noise import create_denoising_pairs

# Create noisy-clean pairs for training
pairs = create_denoising_pairs(
    corpus_path="data/clean_corpus.txt",
    noise_rate=0.05,
    output_path="src/finetuning_data/denoising_pairs.jsonl"
)
```

## Evaluation

### Running Evaluations

The evaluation scripts are located in `src/evaluations_scripts/`.

```bash
# CLSD evaluation on clean data
python src/evaluations_scripts/CLSD_wmt_evaluation.py

# CLSD evaluation with OCR noise (BlackLetter)
python src/evaluations_scripts/CLSD_wmt_evaluation_gte_mono_BL.py

# CLSD evaluation with Salt & Pepper noise
python src/evaluations_scripts/CLSD_wmt_evaluation_gte_mono_SnP.py

# Historical Bitext Mining evaluation
python src/evaluations_scripts/HistBIM_evaluation.py

# Cross-lingual STS evaluation
python src/evaluations_scripts/X-STS17_evaluation.py

# PARALUX evaluation
python src/evaluations_scripts/PARALUX_evaluation.py
```

### Evaluation Datasets

Evaluation sets are located in `src/evaluation_sets/`:

| File | Description |
|------|-------------|
| `CLSD_wmt2019_adversarial_dataset.csv` | Clean CLSD WMT19 test set |
| `CLSD_wmt2021_adversarial_dataset.csv` | Clean CLSD WMT21 test set |
| `CLSD_WMT19_BLDS_noise.csv` | WMT19 with BlackLetter/ScannedNoise |
| `CLSD_WMT21_BLDS_noise.csv` | WMT21 with BlackLetter/ScannedNoise |
| `CLSD_WMT19_SNP_noise.csv` | WMT19 with Salt-and-Pepper noise |
| `CLSD_WMT21_SNP_noise.csv` | WMT21 with Salt-and-Pepper noise |
| `CLSD_WMT19_MN_noise.csv` | WMT19 with mixed noise |
| `CLSD_WMT21_MN_noise.csv` | WMT21 with mixed noise |

### Evaluation Benchmarks

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| **CLSD** | Cross-Lingual Semantic Discrimination (DE↔FR) | Precision@1 |
| **CLSD-OCR** | CLSD with simulated OCR degradation | Precision@1 |
| **X-STS** | Cross-lingual Semantic Textual Similarity | Spearman ρ |
| **HistLUX** | Historical Luxembourgish Bitext Mining | Accuracy |

### OCR Noise Types

- **BlackLetter/ScannedNoise (BL/SN)**: Distortions from historical typefaces and low-quality scans
- **Salt-and-Pepper (SnP)**: Artifacts from paper degradation and aged-document backgrounds

## Results

Average results across five fine-tuning seeds:

| Model | HistLUX | CLSD-BL/SN | CLSD-SnP | Clean CLSD | X-STS |
|-------|---------|------------|----------|------------|-------|
| M-GTE Base | 83.78 | 78.15 | 81.62 | 90.50 | 79.78 |
| `halloween_workshop_ocr_robust_preview` | 87.43 | 81.29 | 84.03 | 93.26 | 79.17 |
| `halloween_workshop_ocr_robust_with_lux_preview` | **97.59** | **81.53** | 76.74 | 92.50 | 71.16 |

## Repository Structure

```
.
├── results/                              # Evaluation results
├── src/
│   ├── evaluation_sets/                  # Evaluation datasets
│   │   ├── CLSD_WMT19_BLDS_noise.csv
│   │   ├── CLSD_WMT19_MN_noise.csv
│   │   ├── CLSD_WMT19_SNP_noise.csv
│   │   ├── CLSD_WMT21_BLDS_noise.csv
│   │   ├── CLSD_WMT21_MN_noise.csv
│   │   ├── CLSD_WMT21_SNP_noise.csv
│   │   ├── CLSD_wmt2019_adversarial_dataset.csv
│   │   ├── CLSD_wmt2021_adversarial_dataset.csv
│   │   └── taken_evaluation_sets/
│   ├── evaluations_scripts/              # Evaluation scripts
│   │   ├── CLSD_wmt_evaluation_e5_base.py
│   │   ├── CLSD_wmt_evaluation_e5_cross-clean.py
│   │   ├── CLSD_wmt_evaluation_e5_cross.py
│   │   ├── CLSD_wmt_evaluation_e5_mono.py
│   │   ├── CLSD_wmt_evaluation_e5.py
│   │   ├── CLSD_wmt_evaluation_gte_base.py
│   │   ├── CLSD_wmt_evaluation_gte_cross-clean.py
│   │   ├── CLSD_wmt_evaluation_gte_cross.py
│   │   ├── CLSD_wmt_evaluation_gte_mono_BL.py
│   │   ├── CLSD_wmt_evaluation_gte_mono_SnP.py
│   │   ├── CLSD_wmt_evaluation_gte_mono.py
│   │   ├── CLSD_wmt_evaluation_gte_x_mono.py
│   │   ├── CLSD_wmt_evaluation_mgte.py
│   │   ├── CLSD_wmt_evaluation.py
│   │   ├── HistBIM_evaluation.py         # Historical Bitext Mining evaluation
│   │   ├── PARALUX_evaluation.py
│   │   └── X-STS17_evaluation.py         # Cross-lingual STS evaluation
│   ├── finetuning/                       # Training scripts
│   │   ├── generate_random_noise/        # OCR noise generation utilities
│   │   └── finetuning_no_sampler_args.py
│   ├── finetuning_data/                  # Training datasets
│   └── prepared_bitext_mining_format/    # Preprocessed bitext data
├── evaluation.py
├── .gitattributes
├── LICENSE
└── README.md
```

## Citation

If you use this code or our models, please cite our paper:

```bibtex
@inproceedings{anonymous2026ocrmgte,
    title={A Recipe for Adapting Multilingual Embedders to OCR-Error Robustness and Historical Texts},
    author={Anonymous},
    booktitle={Proceedings of the 15th Language Resources and Evaluation Conference (LREC 2026)},
    year={2026},
    publisher={European Language Resources Association}
}
```

### Related Work

This work builds upon:

```bibtex
@inproceedings{michail2025cheap,
    title={Cheap Character Noise for OCR-Robust Multilingual Embeddings},
    author={Michail, Andrianos and Opitz, Juri and Wang, Yining and Meister, Robin and Sennrich, Rico and Clematide, Simon},
    booktitle={Findings of ACL 2025},
    year={2025}
}

@inproceedings{michail2025adapting,
    title={Adapting Multilingual Embedding Models to Historical Luxembourgish},
    author={Michail, Andrianos and Raclé, Corina and Opitz, Juri and Clematide, Simon},
    booktitle={Proceedings of LaTeCH-CLfL 2025},
    year={2025}
}

@inproceedings{zhang2024mgte,
    title={mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval},
    author={Zhang, Xin and others},
    booktitle={EMNLP 2024 Industry Track},
    year={2024}
}
```

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Base model: [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- Training framework: [Sentence-Transformers](https://www.sbert.net/)
- Models hosted by: [Impresso Project](https://huggingface.co/impresso-project)

## Contact

For questions or issues, please open a GitHub issue or contact the authors.