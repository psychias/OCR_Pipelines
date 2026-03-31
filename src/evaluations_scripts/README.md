# Evaluation Scripts

This directory contains evaluation scripts used in the ACL 2025 and LREC 2026 papers.
Each script evaluates a specific model / noise-type / task combination. They are kept
as individual files to preserve the experimental history.

## Script Index

| Script | Model Family | Noise Type | Task |
|--------|-------------|------------|------|
| `CLSD_wmt_evaluation.py` | E5 (multilingual-e5-base) | Simple, BlackLetter, Salt-and-Pepper | CLSD baseline |
| `CLSD_wmt_evaluation_e5.py` | E5 (cross-clean fine-tuned) | Simple, BlackLetter, Salt-and-Pepper | CLSD cross-clean |
| `CLSD_wmt_evaluation_e5_base.py` | E5 (multilingual-e5-base) | Simple, BlackLetter, Salt-and-Pepper | CLSD base |
| `CLSD_wmt_evaluation_e5_cross-clean.py` | E5 (cross-clean fine-tuned) | Simple, BlackLetter, Salt-and-Pepper | CLSD cross-clean |
| `CLSD_wmt_evaluation_e5_cross.py` | E5 (cross fine-tuned) | Simple, BlackLetter, Salt-and-Pepper | CLSD cross |
| `CLSD_wmt_evaluation_e5_mono.py` | E5 (mono fine-tuned) | Simple, BlackLetter, Salt-and-Pepper | CLSD mono |
| `CLSD_wmt_evaluation_gte_base.py` | GTE (gte-multilingual-base) | Simple, BlackLetter, Salt-and-Pepper | CLSD base |
| `CLSD_wmt_evaluation_gte_cross-clean.py` | GTE (cross-clean fine-tuned) | Simple, BlackLetter, Salt-and-Pepper | CLSD cross-clean |
| `CLSD_wmt_evaluation_gte_cross.py` | GTE (cross fine-tuned) | Simple, BlackLetter, Salt-and-Pepper | CLSD cross |
| `CLSD_wmt_evaluation_gte_mono.py` | GTE (mono fine-tuned) | Simple, BlackLetter, Salt-and-Pepper | CLSD mono |
| `CLSD_wmt_evaluation_gte_mono_BL.py` | GTE (mono BlackLetter fine-tuned) | Simple, BlackLetter, Salt-and-Pepper | CLSD mono-BL |
| `CLSD_wmt_evaluation_gte_mono_SnP.py` | GTE (mono Salt-and-Pepper fine-tuned) | Simple, BlackLetter, Salt-and-Pepper | CLSD mono-SnP |
| `CLSD_wmt_evaluation_gte_x_mono.py` | GTE (cross-mono fine-tuned) | Simple, BlackLetter, Salt-and-Pepper | CLSD x-mono |
| `CLSD_wmt_evaluation_mgte.py` | Multiple GTE variants | Various | HistBIM (bitext mining) |
| `HistBIM_evaluation.py` | Multiple GTE models | N/A | Historical Bitext Mining (LB↔DE/FR/EN) |
| `PARALUX_evaluation.py` | Multiple GTE models | N/A | PARALUX paraphrase robustness |
| `X-STS17_evaluation.py` | GTE (mono-BL fine-tuned) | N/A | Cross-lingual STS (AR-EN, ES-EN, TR-EN) |

## Path Conventions

- **CLSD scripts** reference evaluation CSVs via `./evaluation/` relative paths
  (e.g., `./evaluation/wmt2019_adversarial_dataset.csv`). These are intended to be
  run from a working directory that contains an `evaluation/` symlink or copy.
- **HistBIM scripts** reference JSONL files via `evaluation/bitext_mining_task_*.jsonl`.
- **X-STS17** references `./evaluation_sets/taken_evaluation_sets/sts17_*.csv`.
- The canonical noisy CLSD evaluation datasets now live in
  `noisy_evaluation_datasets/ACL/CLSD_WMT{19,21}_{BLDS,SNP,MN}_noise.csv`.
