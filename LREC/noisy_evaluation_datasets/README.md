# LREC – Noisy Evaluation Datasets

## HistBIM (Bitext Mining)

Six JSONL files for the **Historical Bitext Mining** evaluation task.
Each line is a JSON object with a `custom_id` and a `translation` array
containing a Luxembourgish sentence paired with one target language.

| File | Direction |
|------|-----------|
| `bitext_mining_task_lb_to_de.jsonl` | Luxembourgish → German |
| `bitext_mining_task_lb_to_en.jsonl` | Luxembourgish → English |
| `bitext_mining_task_lb_to_fr.jsonl` | Luxembourgish → French |
| `bitext_mining_task_de_to_lb.jsonl` | German → Luxembourgish |
| `bitext_mining_task_en_to_lb.jsonl` | English → Luxembourgish |
| `bitext_mining_task_fr_to_lb.jsonl` | French → Luxembourgish |

## X-STS17 (Semantic Textual Similarity)

Cross-lingual STS benchmark subsets stored in `taken_evaluation_sets/`.
Columns use ISO 639-3 codes (`eng`, `ara`, `spa`, `tur`) plus
`similarity_score`.

| File | Languages |
|------|-----------|
| `sts17_ar-en.csv` | Arabic – English |
| `sts17_en-es.csv` | English – Spanish |
| `sts17_es-en.csv` | Spanish – English |
| `sts17_tr-en.csv` | Turkish – English |
