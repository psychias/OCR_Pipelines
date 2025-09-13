# OCR_Pipelines

--- 
## High-Level Goal

create a generalizable OCR robust embedding model that can:

Handle OCR noise.

Generalize across languages  and domains.

Learn to embed both queries and documents in the same space.

Learn to rank or retrieve correctly under real-world noisy conditions.

To do this, you mix:

  [1] summary -> doc
  [2] doc -> doc (German noisy)
  [3] doc -> doc (French noisy)
  [4] summary -> summary_noise
  [5] query -> doc
  [6] query -> summary
  [7] TED q -> q (de <-> fr)
  [8] LUX q -> q (lb <-> {de,fr,en})

### Evaluation 
The evaluation script tests the following:




1. **Retrieval performance** — how well the model retrieves relevant documents given a query.
2. **Robustness to noise** — how well the model handles OCR errors and historical spelling variations.
3. **Cross-lingual capabilities** — how well the model generalizes across different languages and dialects.




######################################
Evalutation:
LUX
STSb
BM25


sts

mix different granualities 
retrieval evaluation 

BM25 down, performance up. -> more robust

multilingual information retrieval evaluation:

https://huggingface.co/datasets/Shitao/MLDR (note to context length den ftanei gia ta documents, mallon apla trunc)


Gia source code se evaluation iparxei edo:

https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/Retrieval/multilingual/MultiLongDocRetrieval.py

OCR:
https://github.com/impresso/ocr-robust-multilingual-embeddings/tree/main/ocr_simulator