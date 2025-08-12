# OCR_Pipelines

---

##  Historical Luxembourgish

* **Goal:** Improve multilingual sentence embeddings for **historical Luxembourgish (LB)** texts.
* **Method:**

  * Use **real historical LB corpora** (1841–1948 newspapers).
  * Align LB with other languages (DE, FR, EN) using bitext mining.
  * Fine-tune a pretrained multilingual model (mBERT / LaBSE style) on *authentic* historical text, so embeddings work well even with old spelling/orthography.
* **Key point:** Noise here is **naturally occurring** from historical spelling drift, not synthetic OCR.

---

## character-level noise
* **Goal:** Make multilingual embeddings more robust to **OCR noise** (misrecognized characters, diacritics, glyph confusions).
* **Method:**

  * Inject **synthetic OCR-like character noise** into *modern* multilingual text.
  * Fine-tune embeddings so they align noisy text with its clean counterpart (denoising objective).
  * Evaluate in OCR settings across multiple languages.
* **Key point:** Noise here is **artificially generated** but targeted to OCR patterns.

---

## Common points between the two

* Both **start with a multilingual embedding model** (e.g., LaBSE, mBERT, gte-multilingual).
* Both **fine-tune to adapt embeddings** to a specific robustness domain:

  * Paper 1 → historical orthography drift
  * Paper 2 → OCR character noise
* Both **use contrastive / alignment losses** so the model learns to map “difficult” text to the same vector space as its normalized/translated equivalent.

---

## Extension

**Data sources:**

1. **Historical Greek (EL)** —

   * *Old → Modern Greek* (orthography normalization pairs).
   * *Old Greek ↔ Other languages* (parallel corpora).
2. Possibly **Luxembourgish historical data** as well, to test multilingual generalization.

**Objectives:**

1. **Denoising fine-tuning** —  add *synthetic OCR noise* to both historical and modern Greek.
2. **Historical adaptation** —  train on authentic historical Greek text to handle orthography drift.
3. **Cross-lingual alignment** — align old Greek with EN/DE/FR and modern Greek.
4. **New retrieval objectives** —

   * **Query → Document**
   * **Document ↔ Document**
     (beyond just sentence-to-sentence)

**Goal:**
Produce **generalizable historical embeddings** that:

* Handle **both** orthographic variation and OCR corruption.
* Work **multilingually** (Greek ↔ modern Greek ↔ other languages).
* Are **retrieval-friendly** at multiple granularity levels (sentence, query, document).

---

* Phase 1 → noise & normalization
* Phase 2 → historical & cross-lingual
* Phase 3 → retrieval (q→d, d↔d)


