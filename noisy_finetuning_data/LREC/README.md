# Noisy Fine-tuning Data – LREC

Historical newspaper articles, query-document pairs, and Luxembourgish
parallel sentences used for fine-tuning.

## Historical articles

| File | Description |
|------|-------------|
| `de_docs_random_noise.csv` | German historical newspaper articles. Key columns: `deu`, `deu_04`, plus metadata (`ci_id`, `ocrqa`, `lg`, `newspaper`, `date`). |
| `fr_docs_random_noise.csv` | French historical newspaper articles. Key columns: `fra`, `fra_04`, plus metadata. |
| `query_doc_dataset_random_noise.csv` | Multilingual query-document pairs. Columns: `text`, `summary`, `query` (clean) and `text_noised`, `summary_noised`, `query_noised` (noisy). |

## Luxembourgish parallel dataset

≈ 120 000 Luxembourgish sentence pairs for **Task A** (cross-lingual
alignment without noise). Each JSONL line contains:

```json
{"custom_id": "…", "translation": [{"lb": "…", "de": "…"}]}
```

| File | Pair |
|------|------|
| `lb_de_training_set.jsonl` | Luxembourgish – German |
| `lb_en_training_set.jsonl` | Luxembourgish – English |
| `lb_fr_training_set.jsonl` | Luxembourgish – French |
