# eval_historic_ocr_xl_plus.py
import os, json, random, math
from itertools import islice
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi  # BM25 baseline

# ----------------------------
# Device / seeds
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
RNG_SEED = 42
random.seed(RNG_SEED); np.random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)

# ----------------------------
# Eval config
# ----------------------------
EVAL_N = 1500                 # cap per language / dataset
MAX_CHUNK_LEN = 256           # clamped to model.max_seq_length
CHUNK_STRIDE = 128
BATCH_ENCODE = 16
TOPK = [1, 5, 10]
MIN_DOC_CHARS = 400
TITLE_MIN_CHARS = 10
OCR_BINS = 4                  # number of quantile bins for Europeana mean_ocr
MIN_BIN_SIZE = 150            # skip bins with fewer examples than this

# ----------------------------
# Helpers
# ----------------------------
def find_model_directories(prefix: str) -> List[str]:
    base = os.path.join(os.getcwd(), "trained_models")
    if not os.path.exists(base): return []
    return [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and d.startswith(prefix)]

def tag(text: str, model_name: str, kind: str) -> str:
    s = "" if text is None else str(text)
    low = model_name.lower()
    if "multilingual-e5" in low or "/e5" in low:
        return ("query: " if kind == "query" else "passage: ") + s
    return s

def _ensure_title(text: str, title: Optional[str]) -> str:
    if title and len(title.strip()) >= TITLE_MIN_CHARS:
        return title.strip()
    toks = (text or "").split()
    return " ".join(toks[:12]) if toks else "untitled"

# ----------------------------
# Chunking (no hard truncation)
# ----------------------------
def _word_chunks(words: List[str], chunk_len: int, stride: int) -> List[str]:
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_len]))
        if i + chunk_len >= len(words): break
        i += max(1, stride)
    return out

def chunk_text_for_model(model: SentenceTransformer, text: str,
                         desired_len: int = MAX_CHUNK_LEN,
                         stride: int = CHUNK_STRIDE) -> List[str]:
    max_len = getattr(model, "max_seq_length", desired_len)
    chunk_len = min(desired_len, max_len - 2) if max_len and max_len > 2 else desired_len
    words = (text or "").split()
    if not words: return []
    return _word_chunks(words, chunk_len=chunk_len, stride=stride)

def encode_chunks_aggregate(model: SentenceTransformer, docs: List[str]) -> torch.Tensor:
    pooled = []
    for doc in tqdm(docs, desc="Encode docs (sliding windows)"):
        chunks = chunk_text_for_model(model, doc)
        if not chunks: chunks = [doc]
        embs = model.encode(chunks, batch_size=BATCH_ENCODE, convert_to_tensor=True,
                            device=device, show_progress_bar=False)
        pooled.append(embs.mean(dim=0))
    return torch.stack(pooled, dim=0) if pooled else torch.empty(0)

def encode_queries(model: SentenceTransformer, queries: List[str], model_name: str) -> torch.Tensor:
    q = [tag(q, model_name, "query") for q in queries]
    return model.encode(q, batch_size=BATCH_ENCODE, convert_to_tensor=True,
                        device=device, show_progress_bar=False)

# ----------------------------
# Dataset-specific loaders (schema-tailored)
# ----------------------------
def load_europeana_lang(lang_config: str, limit: int) -> List[Dict]:
    """
    Returns list of dicts: {id, lang, title, text, date, mean_ocr}
    """
    ds = load_dataset("biglam/europeana_newspapers", lang_config, split="train", streaming=True)
    rows, out = 0, []
    for ex in ds:
        text = ex.get("text") or ""
        if len(text) < MIN_DOC_CHARS: continue
        title = _ensure_title(text, ex.get("title"))
        mean_ocr = ex.get("mean_ocr")
        try:
            mean_ocr = float(mean_ocr) if mean_ocr is not None else None
        except Exception:
            mean_ocr = None
        out.append({
            "id": ex.get("id") or f"{lang_config}-{rows}",
            "lang": lang_config,
            "title": title,
            "text": text,
            "date": ex.get("date"),
            "mean_ocr": mean_ocr
        })
        rows += 1
        if rows >= limit: break
    return out

def load_americanstories(limit: int) -> List[Dict]:
    ds = load_dataset("dell-research-harvard/AmericanStories", "subset_years", split="train", streaming=True)
    rows, out = 0, []
    for ex in ds:
        text = ex.get("article") or ""
        if len(text) < MIN_DOC_CHARS: continue
        title = _ensure_title(text, ex.get("headline"))
        out.append({
            "id": ex.get("article_id") or f"as-{rows}",
            "lang": "en",
            "title": title,
            "text": text,
            "date": ex.get("date")
        })
        rows += 1
        if rows >= limit: break
    return out

def load_kubhist2(limit_docs: int, sentences_per_doc: int = 300) -> List[Dict]:
    ds = load_dataset("iguanodon-ai/kubhist2", split="train", streaming=True)
    out, cur_sent, cur_buf, doc_id = [], 0, [], 0
    for ex in ds:
        t = ex.get("text") or ""
        if not t: continue
        cur_buf.append(t.strip())
        cur_sent += 1
        if cur_sent >= sentences_per_doc:
            text = " ".join(cur_buf)
            if len(text) >= MIN_DOC_CHARS:
                out.append({
                    "id": f"kub-{doc_id}",
                    "lang": "sv",
                    "title": _ensure_title(text, None),
                    "text": text
                })
                doc_id += 1
                if doc_id >= limit_docs: break
            cur_buf, cur_sent = [], 0
    return out

# ----------------------------
# Pools & metrics
# ----------------------------
def build_qd_pool(examples: List[Dict]) -> Tuple[List[str], List[str], np.ndarray]:
    queries = [e["title"] for e in examples]
    docs = [e["text"] for e in examples]
    gold = np.arange(len(examples))
    return queries, docs, gold

def recall_at_k(scores: torch.Tensor, gold: np.ndarray, ks=TOPK) -> Dict[str, float]:
    N, M = scores.shape
    topk = torch.topk(scores, k=max(ks), dim=1).indices.cpu().numpy()
    out = {}
    for k in ks:
        hits = sum(gold[i] in topk[i, :k] for i in range(N))
        out[f"R@{k}"] = 100.0 * hits / N if N else 0.0
    out["P@1"] = out["R@1"]
    return out

# ----------------------------
# BM25 baseline (q→d)
# ----------------------------
def _tok(s: str) -> List[str]:
    return [w for w in (s or "").lower().split() if w]

def bm25_eval_qd(queries: List[str], docs: List[str], gold: np.ndarray) -> Dict[str, float]:
    tokenized_docs = [_tok(d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    topk = max(TOPK)
    hits_at_k = {k: 0 for k in TOPK}
    for i, q in enumerate(queries):
        scores = bm25.get_scores(_tok(q))
        # argsort descending
        idx = np.argsort(-np.array(scores))
        for k in TOPK:
            if gold[i] in idx[:k]:
                hits_at_k[k] += 1
    out = {f"R@{k}": 100.0 * hits_at_k[k] / len(queries) for k in TOPK} if queries else {f"R@{k}": 0.0 for k in TOPK}
    out["P@1"] = out["R@1"]
    return out

# ----------------------------
# Embedder evals
# ----------------------------
def eval_qd_embedder(model: SentenceTransformer, model_name: str, examples: List[Dict]) -> Dict[str, float]:
    q, d, gold = build_qd_pool(examples)
    qe = encode_queries(model, q, model_name)
    de = encode_chunks_aggregate(model, d)
    scores = util.cos_sim(qe, de)
    return recall_at_k(scores, gold)

def eval_dd_embedder(model: SentenceTransformer, examples: List[Dict]) -> Dict[str, float]:
    # split each doc in two halves; A retrieves B
    A, B = [], []
    for e in examples:
        words = e["text"].split()
        if len(words) < 200: continue
        mid = len(words)//2
        A.append(" ".join(words[:mid]))
        B.append(" ".join(words[mid:]))
    gold = np.arange(len(A))
    if not A: 
        return {"P@1": 0.0, "R@5": 0.0, "R@10": 0.0}
    Ae = encode_chunks_aggregate(model, A)
    Be = encode_chunks_aggregate(model, B)
    scores = util.cos_sim(Ae, Be)
    return recall_at_k(scores, gold)

# ----------------------------
# Europeana cross-lingual (mixed pool)
# ----------------------------
def eval_europeana_xl_mixed(model: SentenceTransformer, model_name: str,
                            fr: List[Dict], de: List[Dict]) -> Dict[str, float]:
    fr_q = [e["title"] for e in fr]
    frd = [e["text"] for e in fr]
    ded = [e["text"] for e in de]
    pool = frd + ded
    gold_fr = np.arange(len(fr))  # FR gold at same index in pool
    qe_fr = encode_queries(model, fr_q, model_name)
    de_pool = encode_chunks_aggregate(model, pool)
    scores_fr = util.cos_sim(qe_fr, de_pool)
    fr_metrics = {f"FR→(FR+DE) {k}": v for k, v in recall_at_k(scores_fr, gold_fr).items()}

    de_q = [e["title"] for e in de]
    gold_de = np.arange(len(de)) + len(frd)  # DE docs offset
    qe_de = encode_queries(model, de_q, model_name)
    scores_de = util.cos_sim(qe_de, de_pool)
    de_metrics = {f"DE→(FR+DE) {k}": v for k, v in recall_at_k(scores_de, gold_de).items()}

    return {**fr_metrics, **de_metrics, "PoolSize": len(pool)}

# ----------------------------
# OCR-confidence stratification
# ----------------------------
def ocr_quantile_bins(examples: List[Dict], q_bins: int = OCR_BINS) -> List[Tuple[float, float]]:
    vals = [e["mean_ocr"] for e in examples if isinstance(e.get("mean_ocr"), (int, float))]
    if len(vals) < max(20, q_bins):   # too few to bin; return single bin
        return [(float("-inf"), float("inf"))]
    qs = np.quantile(vals, np.linspace(0, 1, q_bins+1))
    bins = []
    for i in range(q_bins):
        lo, hi = float(qs[i]), float(qs[i+1] + 1e-9)
        bins.append((lo, hi))
    return bins

def filter_by_ocr_bin(examples: List[Dict], lo: float, hi: float) -> List[Dict]:
    out = []
    for e in examples:
        v = e.get("mean_ocr")
        if isinstance(v, (int, float)) and lo <= v <= hi:
            out.append(e)
    return out

# ----------------------------
# Orchestrator
# ----------------------------
def run_all(prefixes: List[str]):
    os.makedirs("./results", exist_ok=True)

    # Preload datasets
    eu_de = load_europeana_lang("de", EVAL_N)
    eu_fr = load_europeana_lang("fr", EVAL_N)
    eu_mix = (eu_de[:EVAL_N//2] + eu_fr[:EVAL_N//2])
    amer  = load_americanstories(EVAL_N)
    kub   = load_kubhist2(EVAL_N)

    # Precompute OCR bins for Europeana
    de_bins = ocr_quantile_bins(eu_de, OCR_BINS)
    fr_bins = ocr_quantile_bins(eu_fr, OCR_BINS)

    all_rows = []
    for pref in prefixes:
        mdirs = find_model_directories(pref)
        if not mdirs:
            print(f"No models found for prefix '{pref}'"); 
            continue

        for mdir in mdirs:
            try:
                model = SentenceTransformer(mdir, trust_remote_code=True).to(device)
                model_name = os.path.basename(mdir)
            except Exception as e:
                print("Load failed:", mdir, e); 
                continue

            # ---------------- Europeana (mono, full) ----------------
            def eval_qd_pair(examples, dataset_tag):
                q, d, gold = build_qd_pool(examples)
                # Embedder
                qd_emb = eval_qd_embedder(model, model_name, examples)
                # BM25
                qd_bm  = bm25_eval_qd(q, d, gold)
                # Δ
                delta_p1 = qd_emb["P@1"] - qd_bm["P@1"]
                return {
                    f"{dataset_tag} P@1(qd)": qd_emb["P@1"], f"{dataset_tag} R@5(qd)": qd_emb["R@5"], f"{dataset_tag} R@10(qd)": qd_emb["R@10"],
                    f"{dataset_tag} P@1(BM25)": qd_bm["P@1"], f"{dataset_tag} R@5(BM25)": qd_bm["R@5"], f"{dataset_tag} R@10(BM25)": qd_bm["R@10"],
                    f"{dataset_tag} ΔP@1": delta_p1
                }

            qd_de = eval_qd_pair(eu_de, "EU-DE")
            dd_de = eval_dd_embedder(model, eu_de)

            qd_fr = eval_qd_pair(eu_fr, "EU-FR")
            dd_fr = eval_dd_embedder(model, eu_fr)

            qd_mix = eval_qd_pair(eu_mix, "EU-MIX")
            dd_mix = eval_dd_embedder(model, eu_mix)

            # ---------------- Europeana (OCR bins) ----------------
            bin_rows = []
            # DE bins
            for i, (lo, hi) in enumerate(de_bins, 1):
                subset = filter_by_ocr_bin(eu_de, lo, hi)
                if len(subset) < MIN_BIN_SIZE: 
                    continue
                metrics = eval_qd_pair(subset, f"EU-DE[mean_ocr∈{lo:.3f},{hi:.3f}]")
                metrics["SubsetSize"] = len(subset)
                metrics["Bin"] = f"DE_bin{i}"
                bin_rows.append(metrics)
            # FR bins
            for i, (lo, hi) in enumerate(fr_bins, 1):
                subset = filter_by_ocr_bin(eu_fr, lo, hi)
                if len(subset) < MIN_BIN_SIZE: 
                    continue
                metrics = eval_qd_pair(subset, f"EU-FR[mean_ocr∈{lo:.3f},{hi:.3f}]")
                metrics["SubsetSize"] = len(subset)
                metrics["Bin"] = f"FR_bin{i}"
                bin_rows.append(metrics)

            # ---------------- Europeana cross-lingual (mixed pool, embedder only) ----------------
            xl = eval_europeana_xl_mixed(model, model_name, eu_fr, eu_de)

            # ---------------- AmericanStories (EN) ----------------
            as_qd = eval_qd_pair(amer, "AS")
            as_dd = eval_dd_embedder(model, amer)

            # ---------------- Kubhist2 (SV) ----------------
            kb_qd = eval_qd_pair(kub, "KUB")
            kb_dd = eval_dd_embedder(model, kub)

            row = {
                "ModelDir": mdir,

                # Europeana full
                **qd_de, "EU-DE P@1(dd)": dd_de["P@1"], "EU-DE R@5(dd)": dd_de["R@5"], "EU-DE R@10(dd)": dd_de["R@10"],
                **qd_fr, "EU-FR P@1(dd)": dd_fr["P@1"], "EU-FR R@5(dd)": dd_fr["R@5"], "EU-FR R@10(dd)": dd_fr["R@10"],
                **qd_mix, "EU-MIX P@1(dd)": dd_mix["P@1"], "EU-MIX R@5(dd)": dd_mix["R@5"], "EU-MIX R@10(dd)": dd_mix["R@10"],

                # XL mixed (embedder only)
                **xl,

                # AmericanStories
                **as_qd, "AS P@1(dd)": as_dd["P@1"], "AS R@5(dd)": as_dd["R@5"], "AS R@10(dd)": as_dd["R@10"],

                # Kubhist2
                **kb_qd, "KUB P@1(dd)": kb_dd["P@1"], "KUB R@5(dd)": kb_dd["R@5"], "KUB R@10(dd)": kb_dd["R@10"],
            }
            all_rows.append(row)

            # Save OCR-bin rows per model
            if bin_rows:
                bins_df = pd.DataFrame(bin_rows)
                bins_out = f"./results/ocr_bins_{os.path.basename(mdir)}.csv"
                bins_df.to_csv(bins_out, index=False)
                print("Saved OCR-bin results:", bins_out)

    # Save main table
    if all_rows:
        df = pd.DataFrame(all_rows)
        out = "./results/eval_historic_ocr_xl_plus.csv"
        df.to_csv(out, index=False)
        print("Saved:", out)
    else:
        print("No results to save.")

if __name__ == "__main__":
    MODEL_PREFIXES = [
        "Alibaba-NLP_gte-multilingual-base-doc_mix_training-40000",
        # add more prefixes here if needed
    ]
    run_all(MODEL_PREFIXES)
