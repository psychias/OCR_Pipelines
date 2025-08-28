# align_news_with_querygen_fast.py
# FR MLSUM → (query, translation) builder
# - Fixed batching (100) for both query-gen and translation
# - Faithful translations: deterministic beams, no output "cleanup"
# - Dynamic max_new_tokens sized from batch source length
# - Sentence packing for long paragraphs
# - Safe tokenizer/model length clamps; cautious CUDA moves

import os
import json
import re
import math
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from time import perf_counter

# Better error localization if a CUDA assert ever happens
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ----------------------------
# Config
# ----------------------------
OUT_DIR = Path("./out_news_querygen")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS_MLSUM_FR = 10_000
BATCH_ROWS = 2_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16_OK = (DEVICE == "cuda" and torch.cuda.is_bf16_supported())

# Fixed batch size
FIXED_BATCH_SIZE = 100

# Models
MODEL_NAME_QUERY = "BeIR/query-gen-msmarco-t5-base-v1"
MODEL_NAMES_TRANSL = {
    ("de", "fr"): "Helsinki-NLP/opus-mt-de-fr",
    ("fr", "de"): "Helsinki-NLP/opus-mt-fr-de",
}
USE_NLLB_FALLBACK = False  # keep False for speed; unused below

# Query gen (short)
MAX_NEW_TOKENS_QUERY = 24
QUERY_ENC_MAXLEN = 384

# Translation packing + budgets
SRC_CHUNK_TOKENS = 300            # encoder chunk size (<= ~500)
MAX_INPUT_LEN_TRANSL = 768        # pre-clamp cap (will clamp to model max)
# Dynamic per-batch target sizing
DYN_TARGET_RATIO = 1.3            # target ≈ ratio * max(src_len_in_batch)
DYN_TARGET_BONUS = 20             # extra safety tokens
DECODER_SAFETY = 8                # headroom vs decoder limit

# ----------------------------
# IO utils
# ----------------------------
def write_jsonl(path: Path, it: Iterable[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for rec in it:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    cols = sorted({k for r in rows for k in r.keys()})
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def is_blank(x: Any) -> bool:
    return not isinstance(x, str) or not x.strip()

# ----------------------------
# Safe tokenizer lengths
# ----------------------------
_BIG_SENTINEL = 10_000
def effective_model_max_len(tok: AutoTokenizer, safety: int = 16, fallback: int = 512) -> int:
    m = getattr(tok, "model_max_length", fallback) or fallback
    if m > _BIG_SENTINEL:  # unknown sentinel
        m = fallback
    return max(8, int(m) - safety)

def decoder_max_len_from_config(mdl) -> int:
    cfg = getattr(mdl, "config", object())
    for attr in ("max_position_embeddings", "n_positions", "max_length", "max_target_positions"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return max(8, int(v) - DECODER_SAFETY)
    return 512 - DECODER_SAFETY  # sensible default

# ----------------------------
# Sentence split & packing
# ----------------------------
def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    # Keep sentence delimiters; preserve newlines as separators
    parts = re.split(r'([.!?…]+["»”\)\]]*\s+|\n+)', text)
    if len(parts) == 1:
        return [text.strip()]
    sents = []
    for i in range(0, len(parts), 2):
        s = parts[i]
        tail = parts[i + 1] if i + 1 < len(parts) else ""
        cand = (s + tail).strip()
        if cand:
            sents.append(cand)
    return sents

def _pack_docs_into_chunks_marian(
    docs: List[str], tok: AutoTokenizer, max_src_tokens: int
) -> Tuple[List[Tuple[int, str]], List[int]]:
    """
    Returns:
      work:   [(doc_idx, chunk_text), ...]
      counts: number of chunks per doc
    """
    work: List[Tuple[int, str]] = []
    counts: List[int] = [0] * len(docs)

    for di, doc in enumerate(docs):
        if not doc or not doc.strip():
            continue
        sents = split_sentences(doc) or [doc.strip()]
        enc = tok(sents, add_special_tokens=False, return_length=True)
        lens = enc["length"]

        cur: List[str] = []
        cur_len = 0
        for s, n in zip(sents, lens):
            if n > max_src_tokens:
                # very long single sentence → cut by words proportionally
                words = s.split()
                step = max(1, int(len(words) * max_src_tokens / max(n, 1)))
                for i in range(0, len(words), step):
                    sub = " ".join(words[i:i + step]).strip()
                    if sub:
                        work.append((di, sub))
                        counts[di] += 1
                cur, cur_len = [], 0
                continue

            if cur_len + n > max_src_tokens and cur:
                work.append((di, " ".join(cur)))
                counts[di] += 1
                cur, cur_len = [s], n
            else:
                cur.append(s)
                cur_len += n

        if cur:
            work.append((di, " ".join(cur)))
            counts[di] += 1

    return work, counts

# ----------------------------
# Minimal helpers
# ----------------------------
def ensure_pad_token(tok: AutoTokenizer):
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

def safe_move_to_device(mdl: AutoModelForSeq2SeqLM, prefer_bf16: bool = True) -> AutoModelForSeq2SeqLM:
    global DEVICE
    if DEVICE != "cuda":
        return mdl.to("cpu")
    try:
        mdl = mdl.to("cuda", non_blocking=True)   # move fp32 first
        if prefer_bf16 and BF16_OK:
            try:
                mdl = mdl.to(dtype=torch.bfloat16)
            except Exception as e:
                print("bf16 cast failed; using fp32:", repr(e))
        return mdl
    except Exception as e:
        print("GPU move failed; falling back to CPU.\n", repr(e))
        DEVICE = "cpu"
        return mdl.to("cpu")

# ----------------------------
# Load models
# ----------------------------
print("Device:", DEVICE)
if DEVICE == "cuda":
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("GPU name unavailable:", repr(e))
    print("BF16 supported:", BF16_OK)

# Query model
print(f"Loading query-gen: {MODEL_NAME_QUERY}")
tok_query = AutoTokenizer.from_pretrained(MODEL_NAME_QUERY, use_fast=True)
tok_query.padding_side = "right"; tok_query.truncation_side = "right"; ensure_pad_token(tok_query)
mdl_query = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_QUERY, low_cpu_mem_usage=True).eval()
mdl_query = safe_move_to_device(mdl_query)
if getattr(mdl_query.config, "pad_token_id", None) is None:
    mdl_query.config.pad_token_id = tok_query.pad_token_id

# Marian caches
tok_cache: Dict[str, AutoTokenizer] = {}
mdl_cache: Dict[str, AutoModelForSeq2SeqLM] = {}

def _load_marian(mname: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    tok = tok_cache.get(mname)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(mname, use_fast=True)
        tok.padding_side = "right"; tok.truncation_side = "right"; ensure_pad_token(tok)
        tok_cache[mname] = tok
    mdl = mdl_cache.get(mname)
    if mdl is None:
        mdl = AutoModelForSeq2SeqLM.from_pretrained(mname, low_cpu_mem_usage=True).eval()
        mdl = safe_move_to_device(mdl)
        if getattr(mdl.config, "pad_token_id", None) is None:
            mdl.config.pad_token_id = tok.pad_token_id
        mdl_cache[mname] = mdl
    return tok, mdl

# ----------------------------
# Generators (fixed-size batching)
# ----------------------------
@torch.no_grad()
def generate_queries(texts: List[str]) -> List[str]:
    prompts = ["generate query: " + (t or "") for t in texts]
    outs = [""] * len(prompts)
    q_enc_max = min(QUERY_ENC_MAXLEN, effective_model_max_len(tok_query, safety=8, fallback=512))

    t0 = perf_counter()
    total_tokens = 0
    for start in range(0, len(prompts), FIXED_BATCH_SIZE):
        end = start + FIXED_BATCH_SIZE
        batch = prompts[start:end]
        enc = tok_query(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=q_enc_max,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        total_tokens += int(enc["attention_mask"].sum().item())

        gen = mdl_query.generate(
            **enc,
            num_beams=1,          # greedy, deterministic
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS_QUERY,
            use_cache=True,
        )
        dec = tok_query.batch_decode(gen, skip_special_tokens=True)
        outs[start:start+len(dec)] = dec

    print(f"[queries] {len(prompts)} items | ~{total_tokens} src toks | {perf_counter()-t0:.1f}s")
    return outs

@torch.no_grad()
def batch_translate_marian(texts: List[str], src_lang: str) -> List[str]:
    if not texts:
        return []
    tgt_lang = "fr" if src_lang == "de" else "de"
    mname = MODEL_NAMES_TRANSL[(src_lang, tgt_lang)]
    tok, mdl = _load_marian(mname)

    # Encoder caps
    tok_max = effective_model_max_len(tok, safety=8, fallback=512)  # ~504 for Marian
    src_budget = min(SRC_CHUNK_TOKENS, tok_max)
    enc_max_len = min(MAX_INPUT_LEN_TRANSL, tok_max)

    # Decoder cap (true model limit, minus safety)
    dec_cap = decoder_max_len_from_config(mdl)

    # Pack long docs into chunks ≤ src_budget
    work, counts = _pack_docs_into_chunks_marian(texts, tok, src_budget)
    if not work:
        return [""] * len(texts)

    outs_per_doc: List[List[str]] = [[] for _ in texts]

    t0 = perf_counter()
    total_src_tokens = 0

    # Fixed-size batching over work items (doc_idx, chunk_text)
    for start in range(0, len(work), FIXED_BATCH_SIZE):
        end = start + FIXED_BATCH_SIZE
        batch = work[start:end]
        batch_texts = [t for (_, t) in batch]

        enc = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=enc_max_len,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        total_src_tokens += int(enc["attention_mask"].sum().item())

        # Faithful decoding: deterministic beams, dynamic target length
        batch_src_max = int(enc["attention_mask"].sum(dim=1).max().item())  # max non-pad
        est_target = int(math.ceil(batch_src_max * DYN_TARGET_RATIO) + DYN_TARGET_BONUS)
        batch_max_new = min(dec_cap, est_target)

        gen = mdl.generate(
            **enc,
            num_beams=2,                 # ↑ quality; set 1 for speed
            do_sample=False,             # deterministic
            max_new_tokens=batch_max_new,
            length_penalty=1.0,
            use_cache=True,
            early_stopping=True,
        )
        dec = tok.batch_decode(gen, skip_special_tokens=True)

        for (di, _), txt in zip(batch, dec):
            # No cleaning: keep raw model output for exactness
            outs_per_doc[di].append(txt)

    print(f"[translate] {len(texts)} docs → {len(work)} chunks | ~{total_src_tokens} src toks | {perf_counter()-t0:.1f}s")
    # Reassemble per document
    return [" ".join(chunks) for chunks in outs_per_doc]

@torch.no_grad()
def translate_batch(texts: List[str], src_lang: str) -> List[str]:
    return batch_translate_marian(texts, src_lang)

# ----------------------------
# Record builder
# ----------------------------
def make_records(rows: List[Dict[str, Any]], lang: str, split: str, prefix: str) -> List[Dict[str, Any]]:
    ids, queries_in, texts_in = [], [], []
    for i, row in enumerate(tqdm(rows, desc="collect input", leave=False)):
        rid = f"{prefix}-{lang}-{split}-{i}"
        ids.append(rid)
        q_src = row.get("target") or row.get("summary") or row.get("title") or ""
        queries_in.append("" if is_blank(q_src) else q_src)
        texts_in.append("" if is_blank(row.get("text")) else row.get("text"))

    tq0 = perf_counter()
    queries_out = generate_queries(queries_in)
    print(f"Queries took {perf_counter()-tq0:.1f}s")

    tt0 = perf_counter()
    translations_out = translate_batch(texts_in, lang)
    print(f"Translations took {perf_counter()-tt0:.1f}s")

    recs: List[Dict[str, Any]] = []
    for i, row in enumerate(tqdm(rows, desc="make_records", leave=False)):
        recs.append({
            "id": ids[i],
            "source": prefix,
            "lang": lang,
            "split": split,
            "title": row.get("title"),
            "topic": row.get("topic"),
            "doc_text": row.get("text"),
            "summary": row.get("summary"),
            "query": queries_out[i],
            "translation": translations_out[i],
        })
    return recs

# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading MLSUM (FR) ...")
    mlsum_fr = load_dataset("reciTAL/mlsum", "fr")
    per_lang: Dict[Tuple[str, str], List[Dict[str, Any]]] = {("fr", "mlsum"): []}

    for split in ("train",):
        ds = mlsum_fr[split]
        limit = len(ds) if MAX_ROWS_MLSUM_FR is None else min(MAX_ROWS_MLSUM_FR, len(ds))
        for start in range(0, limit, BATCH_ROWS):
            end = min(start + BATCH_ROWS, limit)
            rows = list(ds.select(range(start, end)))
            if not rows:
                continue

            print(f"\nFR {split}: rows {start}:{end} (size={len(rows)})")
            recs = make_records(rows, lang="fr", split=split, prefix="mlsum")
            per_lang[("fr", "mlsum")].extend(recs)

    write_jsonl(OUT_DIR / "mlsum_fr.jsonl", per_lang[("fr", "mlsum")])
    write_csv(OUT_DIR / "mlsum_fr.csv", per_lang[("fr", "mlsum")])
    print("Done.")


main()