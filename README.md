# OCR_Pipelines

--- 
## High-Level Goal

create a generalizable historical embedding model that can:

Handle OCR noise.

Generalize across languages (FR/DE) and domains.

Learn to embed both queries and documents in the same space.

Learn to rank or retrieve correctly under real-world noisy conditions.

To do this, you mix:

q → q (query ↔ noisy version of the same sentence)

q → d (query ↔ document match)

d ↔ d (clean ↔ noisy version of full documents)


### Evaluation 
The evaluation script tests the following:

1. **Retrieval performance** — how well the model retrieves relevant documents given a query.
2. **Robustness to noise** — how well the model handles OCR errors and historical spelling variations.
3. **Cross-lingual capabilities** — how well the model generalizes across different languages and dialects.
