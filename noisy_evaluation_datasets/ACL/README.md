# Noisy Evaluation Datasets – ACL

Cross-Lingual Semantic-Divergence (CLSD) benchmark sets derived from
WMT 2019 and WMT 2021 adversarial test suites (French ↔ German),
with synthetic OCR noise applied.

## Column convention

| Column | Meaning |
|--------|---------|
| `fra` | Clean French sentence |
| `deu` | Clean German sentence |
| `fra_04` | French sentence with OCR noise (CER ≈ 0.04) |
| `deu_04` | German sentence with OCR noise (CER ≈ 0.04) |
| `de_adv2 … de_adv4` | Additional German adversarial variants |
| `fr_adv2 … fr_adv4` | Additional French adversarial variants |

## Files

| File | Noise type | Source |
|------|-----------|--------|
| `CLSD_WMT19_BLDS_noise.csv` | BLDS (Bavarian-Latin-Digit Substitution) | WMT 2019 |
| `CLSD_WMT19_MN_noise.csv` | MN (Mixed Noise) | WMT 2019 |
| `CLSD_WMT19_SNP_noise.csv` | SNP (Salt-and-Pepper) | WMT 2019 |
| `CLSD_WMT21_BLDS_noise.csv` | BLDS | WMT 2021 |
| `CLSD_WMT21_MN_noise.csv` | MN | WMT 2021 |
| `CLSD_WMT21_SNP_noise.csv` | SNP | WMT 2021 |

## HistBIM – Bitext Mining (LTZ↔DE/FR/EN)

Six JSONL files for the Historical Bitext Mining evaluation task.
Each line has `source_sentence` and `candidates` keys.

| File | Direction |
|------|----------|
| `bitext_mining_task_lb_to_de.jsonl` | Luxembourgish → German |
| `bitext_mining_task_lb_to_en.jsonl` | Luxembourgish → English |
| `bitext_mining_task_lb_to_fr.jsonl` | Luxembourgish → French |
| `bitext_mining_task_de_to_lb.jsonl` | German → Luxembourgish |
| `bitext_mining_task_en_to_lb.jsonl` | English → Luxembourgish |
| `bitext_mining_task_fr_to_lb.jsonl` | French → Luxembourgish |
