# OCR-Robust Multilingual Embeddings

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/impresso-project)

Data, training, and evaluation code for two papers on making multilingual
sentence embeddings resistant to OCR noise.

## Papers

| Venue | Title | Focus |
|-------|-------|-------|
| **ACL 2025** | *Cross-Lingual Semantic Divergence under OCR Noise* | CLSD benchmark (French ↔ German) with BLDS / MN / SNP noise |
| **LREC 2026** | *Extending OCR-Robust Embeddings to Low-Resource Languages* | Luxembourgish alignment + historical bitext mining |

## Models

| Model | Base | Description |
|-------|------|-------------|
| [`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | — | Baseline GTE encoder |
| [`impresso-project/histlux-gte-multilingual-base`](https://huggingface.co/impresso-project/histlux-gte-multilingual-base) | GTE | Fine-tuned on historical + Luxembourgish data |

## Repository structure

```
├── ACL/
│   ├── noisy_evaluation_datasets/   # 8 CLSD CSVs (WMT 2019 & 2021)
│   └── noisy_finetuning_data/       # 4 TED / X-News CSVs
├── LREC/
│   ├── noisy_evaluation_datasets/   # 6 HistBIM JSONLs + 4 X-STS17 CSVs
│   ├── noisy_finetuning_data/       # 3 historical-article CSVs
│   └── luxembourgish_dataset/       # 3 parallel JSONLs (≈120k pairs)
├── generate_random_character_noise/ # Synthetic OCR noise with confusable tables
├── ocr_simulator/                   # OCR simulation pipeline (from impresso)
├── src/
│   ├── evaluations_scripts/         # 17 evaluation scripts (annotated per paper)
│   ├── finetuning/                  # Training code (InputExample + MNRL)
│   ├── evaluation_sets/             # (legacy) moved to ACL/ and LREC/
│   └── finetuning_data/             # (legacy) moved to ACL/ and LREC/
├── scripts/                         # Shell helpers
├── _tools/                          # normalise_columns.py
├── sample_training.ipynb            # Minimal 2-stage training notebook
├── train_mix_model.ipynb            # Full experiment notebook
├── requirements.txt                 # Pinned Python dependencies
└── LICENSE                          # AGPL-3.0
```

## Column naming convention

All data files follow a consistent column scheme:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `{lang3}` | Clean text in ISO 639-3 | `deu`, `fra`, `eng` |
| `{lang3}_04` | Noised text (CER ≈ 0.04) | `deu_04`, `fra_04` |

Additional noise-rate variants (e.g. `*_random10`, `*_random15`) and
adversarial columns (`de_adv2`–`de_adv4`) are retained where present.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate OCR noise for a CSV
python generate_random_character_noise/generate_random_character_noise.py \
    data.csv --columns deu fra --script latin --cer 0.04 -o noised.csv

# Run a CLSD evaluation
python src/evaluations_scripts/CLSD_wmt_evaluation_gte_base.py

# Minimal training (see sample_training.ipynb for the full walkthrough)
jupyter notebook sample_training.ipynb
```

## Citation

```bibtex
@inproceedings{psychias2025acl,
  title     = {Cross-Lingual Semantic Divergence under {OCR} Noise},
  author    = {Psychias, Alexandros},
  booktitle = {Proceedings of ACL 2025},
  year      = {2025}
}

@inproceedings{psychias2026lrec,
  title     = {Extending {OCR}-Robust Embeddings to Low-Resource Languages},
  author    = {Psychias, Alexandros},
  booktitle = {Proceedings of LREC 2026},
  year      = {2026}
}
```

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

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
