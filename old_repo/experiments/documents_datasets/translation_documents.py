# align_news_with_querygen_improved.py
import json
import re
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, List, Tuple

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
OUT_DIR = Path("./out_news_querygen")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS_MLSUM_FR = 10000
MAX_ROWS_MLSUM_DE = 10000

CHUNK_SIZE = 2500
LIMIT_FR = MAX_ROWS_MLSUM_FR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models
MODEL_NAME_QUERY = "BeIR/query-gen-msmarco-t5-base-v1"

# Primary MT (Marian) + optional fallback (NLLB)
MODEL_NAMES_TRANSL = {
    ("de", "fr"): "Helsinki-NLP/opus-mt-de-fr",
    ("fr", "de"): "Helsinki-NLP/opus-mt-fr-de",
}
USE_NLLB_FALLBACK = True
NLLB_NAME = "facebook/nllb-200-distilled-600M"
# NLLB language codes
NLLB_LANG = {"fr": "fra_Latn", "de": "deu_Latn"}

# Batch sizes / limits
BATCH_SIZE_QUERY = 16
MAX_NEW_TOKENS_QUERY = 128

# Chunking budgets (source-side tokens)
SRC_CHUNK_TOKENS = 300
MAX_INPUT_LEN = 1100  # hard cap for tokenizer truncation

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

def chunk_iter(items: List[Any], bs: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), bs):
        yield items[i:i+bs]

def is_blank(x: Any) -> bool:
    return not isinstance(x, str) or not x.strip()

# ----------------------------
# Split / packing for MT
# ----------------------------
def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter that preserves delimiters."""
    if not text or not text.strip():
        return []
    parts = re.split(r'([.!?…]+["»”\)\]]*\s+)', text)
    if len(parts) == 1:
        return [text.strip()]
    sents = []
    for i in range(0, len(parts), 2):
        s = parts[i]
        tail = parts[i+1] if i+1 < len(parts) else ""
        cand = (s + tail).strip()
        if cand:
            sents.append(cand)
    return sents

def chunk_by_tokens(sents: List[str], tok, max_src_tokens: int = SRC_CHUNK_TOKENS) -> List[List[str]]:
    """Pack sentences into chunks under a token budget."""
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        if n > max_src_tokens:
            # soft break an overly long sentence by words
            words = s.split()
            # proportional step to respect token budget
            step_words = max(1, int(len(words) * max_src_tokens / max(n, 1)))
            for i in range(0, len(words), step_words):
                sub = " ".join(words[i:i+step_words]).strip()
                if sub:
                    chunks.append([sub])
            cur, cur_len = [], 0
            continue
        if cur_len + n > max_src_tokens and cur:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += n
    if cur:
        chunks.append(cur)
    return chunks

# ----------------------------
# Repeat detection / cleaning
# ----------------------------
def looks_repetitive(txt: str) -> bool:
    if not txt:
        return True
    toks = txt.split()
    if not toks:
        return True
    uniq_ratio = len(set(toks)) / max(1, len(toks))
    if uniq_ratio < 0.35:
        return True
    # long repeated n-grams
    for n in (3, 4, 5):
        ngrams = [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
        counts = {}
        for g in ngrams:
            counts[g] = counts.get(g, 0) + 1
        if any(c >= 3 for c in counts.values()):
            return True
    # same token repeated 5+ times
    if re.search(r'(\b\w+\b)(?:\s+\1){4,}', txt, flags=re.IGNORECASE):
        return True
    return False

def clean_minor_repeats(txt: str) -> str:
    txt = re.sub(r'(\b\w+\b)(\s+\1){2,}', r'\1 \1', txt, flags=re.IGNORECASE)
    txt = re.sub(r'([.!?,;:])\1{1,}', r'\1', txt)
    return txt.strip()

def translation_is_acceptable(src: str, tgt: str) -> bool:
    if looks_repetitive(tgt):
        return False
    src_len, tgt_len = len(src.split()), len(tgt.split())
    if tgt_len < 5 or tgt_len > 3.5 * max(5, src_len):
        return False
    return True

# ----------------------------
# Load models
# ----------------------------
print(f"Loading query-gen model: {MODEL_NAME_QUERY}")
tok_query = AutoTokenizer.from_pretrained(MODEL_NAME_QUERY)
mdl_query = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_QUERY).to(DEVICE).eval()
if DEVICE == "cuda":
    mdl_query.half()

tok_cache: Dict[str, AutoTokenizer] = {}
mdl_cache: Dict[str, AutoModelForSeq2SeqLM] = {}

# NLLB cache (optional)
tok_nllb = None
mdl_nllb = None
if USE_NLLB_FALLBACK:
    print(f"Preparing optional fallback MT: {NLLB_NAME}")
    tok_nllb = AutoTokenizer.from_pretrained(NLLB_NAME)
    mdl_nllb = AutoModelForSeq2SeqLM.from_pretrained(NLLB_NAME).to(DEVICE).eval()
    if DEVICE == "cuda":
        mdl_nllb.half()

# ----------------------------
# Generators
# ----------------------------
@torch.no_grad()
def generate_queries(texts: List[str]) -> List[str]:
    prompts = ["generate query: " + (t or "") for t in texts]
    outs: List[str] = []
    for batch in chunk_iter(prompts, BATCH_SIZE_QUERY):
        inputs = tok_query(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LEN).to(DEVICE)
        gen = mdl_query.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_QUERY,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        outs.extend(tok_query.batch_decode(gen, skip_special_tokens=True))
    return outs

@torch.no_grad()
def _translate_chunk_marian(src: str, tok, mdl) -> str:
    inputs = tok([src], return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LEN).to(DEVICE)

    # pass 1: beam, anti-repeat
    gen = mdl.generate(
        **inputs,
        num_beams=5,
        max_new_tokens=256,
        early_stopping=True,
        no_repeat_ngram_size=4,
        encoder_no_repeat_ngram_size=4,
        repetition_penalty=1.15,
        length_penalty=0.9,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id,
    )
    out = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
    out = clean_minor_repeats(out)

    if looks_repetitive(out):
        # pass 2: harder constraints
        gen = mdl.generate(
            **inputs,
            num_beams=6,
            max_new_tokens=220,
            early_stopping=True,
            no_repeat_ngram_size=5,
            encoder_no_repeat_ngram_size=5,
            repetition_penalty=1.3,
            length_penalty=0.8,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id,
        )
        out = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
        out = clean_minor_repeats(out)

    if looks_repetitive(out):
        # pass 3: conservative sampling to escape loops
        gen = mdl.generate(
            **inputs,
            do_sample=True,
            top_p=0.8,
            temperature=0.8,
            max_new_tokens=200,
            no_repeat_ngram_size=4,
            repetition_penalty=1.25,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id,
        )
        out = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
        out = clean_minor_repeats(out)

    return out

@torch.no_grad()
def translate_one_doc_marian(doc: str, src_lang: str) -> str:
    if not doc or not doc.strip():
        return ""
    tgt_lang = "fr" if src_lang == "de" else "de"
    mname = MODEL_NAMES_TRANSL[(src_lang, tgt_lang)]
    tok = tok_cache.setdefault(mname, AutoTokenizer.from_pretrained(mname))
    mdl = mdl_cache.setdefault(mname, AutoModelForSeq2SeqLM.from_pretrained(mname).to(DEVICE).eval())
    if DEVICE == "cuda":
        mdl.half()

    sents = split_sentences(doc)
    if not sents:
        sents = [doc.strip()]
    chunks = chunk_by_tokens(sents, tok, max_src_tokens=SRC_CHUNK_TOKENS)

    out_chunks: List[str] = []
    for chunk in chunks:
        src = " ".join(chunk).strip()
        out = _translate_chunk_marian(src, tok, mdl)
        out_chunks.append(out)
    return " ".join(out_chunks).strip()

@torch.no_grad()
def translate_one_doc_nllb(doc: str, src_lang: str) -> str:
    """Optional fallback translation with NLLB-200."""
    if not USE_NLLB_FALLBACK or not doc or not doc.strip():
        return ""
    assert tok_nllb is not None and mdl_nllb is not None
    tgt_lang = "fr" if src_lang == "de" else "de"
    src_code = NLLB_LANG[src_lang]
    tgt_code = NLLB_LANG[tgt_lang]

    sents = split_sentences(doc)
    if not sents:
        sents = [doc.strip()]
    # NLLB can handle a bit longer chunks; still keep them reasonable
    chunks = chunk_by_tokens(sents, tok_nllb, max_src_tokens=SRC_CHUNK_TOKENS + 80)

    outs = []
    for chunk in chunks:
        src = " ".join(chunk).strip()
        inputs = tok_nllb(
            [src],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
        ).to(DEVICE)
        # set language codes
        tok_nllb.src_lang = src_code
        forced_bos_id = tok_nllb.convert_tokens_to_ids(tgt_code)

        gen = mdl_nllb.generate(
            **inputs,
            forced_bos_token_id=forced_bos_id,
            num_beams=5,
            max_new_tokens=256,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15,
            length_penalty=0.9,
        )
        out = tok_nllb.batch_decode(gen, skip_special_tokens=True)[0].strip()
        out = clean_minor_repeats(out)
        # quick local retry with sampling if still bad
        if looks_repetitive(out):
            gen = mdl_nllb.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                do_sample=True,
                top_p=0.8,
                temperature=0.8,
                max_new_tokens=220,
                no_repeat_ngram_size=4,
                repetition_penalty=1.25,
            )
            out = tok_nllb.batch_decode(gen, skip_special_tokens=True)[0].strip()
            out = clean_minor_repeats(out)
        outs.append(out)
    return " ".join(outs).strip()

@torch.no_grad()
def translate_one_doc(doc: str, src_lang: str) -> str:
    """Primary -> validate -> optional fallback."""
    out = translate_one_doc_marian(doc, src_lang)
    if not translation_is_acceptable(doc, out) and USE_NLLB_FALLBACK:
        out_fb = translate_one_doc_nllb(doc, src_lang)
        if translation_is_acceptable(doc, out_fb):
            return out_fb
    return out

@torch.no_grad()
def translate_batch(texts: List[str], src_lang: str) -> List[str]:
    return [translate_one_doc(t, src_lang) for t in texts]

# ----------------------------
# Record builder
# ----------------------------
def make_records(rows: List[Dict[str, Any]], lang: str, split: str, prefix: str) -> List[Dict[str, Any]]:
    ids, queries_in, texts_in = [], [], []
    for i, row in enumerate(tqdm(rows, desc="collect input")):
        rid = f"{prefix}-{lang}-{split}-{i}"
        ids.append(rid)
        q_src = row.get("target") or row.get("summary") or row.get("title") or ""
        queries_in.append("" if is_blank(q_src) else q_src)
        texts_in.append("" if is_blank(row.get("text")) else row.get("text"))

    queries_out = generate_queries(queries_in)
    translations_out = translate_batch(texts_in, lang)

    recs: List[Dict[str, Any]] = []
    for i, row in enumerate(tqdm(rows, desc="make_records")):
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


def iter_split(ds_dict, split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
        take = ds_dict[split]
        if limit is not None:
            take = take.select(range(min(limit, len(take))))
        return list(take)



def main():
        
    # ----------------------------
    # 1) MLSUM FR & DE
    # ----------------------------
    print("Loading MLSUM (FR/DE) ...")
    mlsum_fr = load_dataset("reciTAL/mlsum", "fr")
    mlsum_de = load_dataset("reciTAL/mlsum", "de")

    
    all_recs: List[Dict[str, Any]] = []
    per_lang: Dict[Tuple[str, str], List[Dict[str, Any]]] = {("fr", "mlsum"): [], ("de", "mlsum"): []}
    BATCH = 1500
    NUM_CHUNKS = (len(mlsum_fr["train"]) + BATCH - 1) // BATCH  # ceiling

    for split in ("train",):
        ds_fr = mlsum_fr[split]
        ds_de = mlsum_de[split]
        limit_fr = min(MAX_ROWS_MLSUM_FR, len(ds_fr))  # <= 10k total
        limit_de = min(MAX_ROWS_MLSUM_DE, len(ds_de))  # <= 10k total

        chunk_idx = 0
        for i, start in enumerate(range(0, len(ds_fr), BATCH)):
            end = min(start + BATCH, limit_fr)
            # select the slice and convert to list[dict]
            fr_rows: List[Dict[str, Any]] = list(ds_fr.select(range(start, end)))
            de_rows: List[Dict[str, Any]] = list(ds_de.select(range(start, min(start + BATCH, limit_de))))

            if fr_rows:
                print(f"Generating FR {split} — {len(fr_rows)} rows")
                fr_recs = make_records(fr_rows, lang="fr", split=split, prefix="mlsum")
                per_lang[("fr", "mlsum")].extend(fr_recs)

            if de_rows:
                print(f"Generating DE {split} — {len(de_rows)} rows")
                de_recs = make_records(de_rows, lang="de", split=split, prefix="mlsum")
                per_lang[("de", "mlsum")].extend(de_recs)
                all_recs.extend(de_recs)




        write_jsonl(OUT_DIR / "mlsum_fr.jsonl", per_lang[("fr", "mlsum")])
        write_csv(OUT_DIR / "mlsum_fr.csv", per_lang[("fr", "mlsum")])

        write_csv(OUT_DIR / "mlsum_de.csv", per_lang[("de", "mlsum")])
        write_jsonl(OUT_DIR / "mlsum_de.jsonl", per_lang[("de", "mlsum")])

        write_jsonl(OUT_DIR / "mlsum_all.jsonl", all_recs)
        write_csv(OUT_DIR / "mlsum_all.csv", all_recs)


    # ----------------------------
    # Combined outputs (optional)
    # ----------------------------
    # write_jsonl(OUT_DIR / "mlsum_all.jsonl", all_recs)
    # write_csv(OUT_DIR / "mlsum_all.csv", all_recs)

    print("Done.")