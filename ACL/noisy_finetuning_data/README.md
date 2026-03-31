# ACL – Noisy Fine-tuning Data

Parallel clean / OCR-noised sentence pairs used for **Task B** fine-tuning
(noise-robust cross-lingual embedding alignment).

## Column convention

| Column | Meaning |
|--------|---------|
| `deu` | Clean German text |
| `fra` | Clean French text |
| `deu_04` | German text with synthetic OCR noise (CER ≈ 0.04) |
| `fra_04` | French text with synthetic OCR noise (CER ≈ 0.04) |

Additional noise-rate variants (`deu_10`, `deu_15`, `fra_10`, `fra_15`) and
realistic-noise columns (`deu_snp`, `fra_snp`) are retained where present.

## Files

| File | Description |
|------|-------------|
| `TED_data_random_noise.csv` | TED talk transcripts with random character noise |
| `TED_data_random_noise_concat.csv` | German-only concatenated variant |
| `TED_data_realistic_noise.csv` | TED transcripts with BLDS-style realistic noise |
| `X-News_data_random_noise.csv` | X-News corpus with random character noise |
