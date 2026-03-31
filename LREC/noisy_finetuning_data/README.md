# LREC – Noisy Fine-tuning Data

Historical newspaper articles and query-document pairs with synthetic OCR
noise, used for **Task B** fine-tuning.

## Column convention

| Column | Meaning |
|--------|---------|
| `deu` / `fra` | Clean text (language depends on file) |
| `deu_04` / `fra_04` | OCR-noised variant (CER ≈ 0.04) |

## Files

| File | Description |
|------|-------------|
| `de_docs_random_noise.csv` | German historical newspaper articles. Key columns: `deu`, `deu_04`, plus metadata (`ci_id`, `ocrqa`, `lg`, `newspaper`, `date`). |
| `fr_docs_random_noise.csv` | French historical newspaper articles. Key columns: `fra`, `fra_04`, plus metadata. |
| `query_doc_dataset_random_noise.csv` | Multilingual query-document pairs. Columns: `text`, `summary`, `query` (clean) and `text_04`, `summary_04`, `query_04` (noisy). |
