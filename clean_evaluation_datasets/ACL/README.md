# Clean Evaluation Datasets – ACL

Clean (no synthetic noise) evaluation data used across both papers.

## CLSD adversarial test sets

| File | Source |
|------|--------|
| `CLSD_wmt2019_adversarial_dataset.csv` | WMT 2019 |
| `CLSD_wmt2021_adversarial_dataset.csv` | WMT 2021 |

Columns: `fra`, `deu`, `fra_04`, `deu_04`, `de_adv2`–`de_adv4`,
`fr_adv2`–`fr_adv4`.


## X-STS17 – Semantic Textual Similarity

Cross-lingual STS benchmark subsets. Columns use ISO 639-3 codes
(`eng`, `ara`, `spa`, `tur`) plus `similarity_score`.

| File | Languages |
|------|-----------|
| `sts17_ar-en.csv` | Arabic – English |
| `sts17_en-es.csv` | English – Spanish |
| `sts17_es-en.csv` | Spanish – English |
| `sts17_tr-en.csv` | Turkish – English |
