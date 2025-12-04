# Standard library imports
import json
import ast

# Third-party imports
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Local imports
from generate_random_character_noise_latin_alphabet import apply_noise_to_dataframe



mlsum_de = load_dataset("reciTAL/mlsum", "de")  # splits: train/validation/test
mlsum_fr = load_dataset("reciTAL/mlsum", "fr")  # splits: train/validation/test
mlsum_es = load_dataset("reciTAL/mlsum", "es")  # splits: train/validation/test
mlsum_en = load_dataset("reciTAL/mlsum", "en")  # splits: train/validation/test
mlsum_ru = load_dataset("reciTAL/mlsum", "ru")  # splits: train/validation/test
mlsum_tu = load_dataset("reciTAL/mlsum", "tu")  # splits: train/validation/test



model_name = "BeIR/query-gen-msmarco-t5-base-v1"
tok = AutoTokenizer.from_pretrained(model_name)
mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
mdl.eval()



def generate_queries(summary: str, n=3, max_new_tokens=25):
    """Generate queries from a summary using T5 model.
    
    Args:
        summary: Input text summary
        n: Number of queries to generate
        max_new_tokens: Maximum tokens per query
        
    Returns:
        List of unique generated queries
    """
    if not isinstance(summary, str) or not summary.strip():
        return []
    
    inp = tok(summary, return_tensors="pt", truncation=True).to(mdl.device)
    out = mdl.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=10,
        num_return_sequences=n,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    
    qs = [tok.decode(o, skip_special_tokens=True).strip() for o in out]
    seen, uniq = set(), []
    for q in qs:
        k = q.lower().rstrip(" ?!.")
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    
    return uniq


def process_language_dataset(dataset, language_code, sample_size=2000):
    """Process a language dataset by sampling and generating queries.
    
    Args:
        dataset: HuggingFace dataset
        language_code: Two-letter language code
        sample_size: Number of samples to process
        
    Returns:
        Processed DataFrame with queries
    """
    df = dataset["train"].to_pandas().sample(sample_size)
    df["language"] = language_code
    df["queries"] = df["summary"].apply(lambda s: generate_queries(s, n=3))
    return df


def last_query(x):
    """Extract the last query from various input formats.
    
    Args:
        x: Input that could be list, string, or None
        
    Returns:
        Last query string or NaN
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return np.nan
    
    if isinstance(x, list):
        vals = [str(v).strip() for v in x if str(v).strip()]
        return vals[-1] if vals else np.nan

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return np.nan
        
        # Try parsing as JSON
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                vals = [str(v).strip() for v in parsed if str(v).strip()]
                return vals[-1] if vals else np.nan
        except Exception:
            pass
        
        # Try parsing as Python literal
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                vals = [str(v).strip() for v in parsed if str(v).strip()]
                return vals[-1] if vals else np.nan
        except Exception:
            pass
        
        # Try splitting by comma
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return parts[-1] if parts else np.nan
        
        return s

    return str(x).strip() or np.nan



print("Processing German dataset...")
mlsum_de_train = process_language_dataset(mlsum_de, "de")
mlsum_de_train.to_csv("mlsum_de_train.csv", index=False)

print("Processing French dataset...")
mlsum_fr_train = process_language_dataset(mlsum_fr, "fr")
mlsum_fr_train.to_csv("mlsum_fr_train.csv", index=False)

print("Processing Spanish dataset...")
mlsum_es_train = process_language_dataset(mlsum_es, "es")
mlsum_es_train.to_csv("mlsum_es_train.csv", index=False)

print("Processing Russian dataset...")
mlsum_ru_train = process_language_dataset(mlsum_ru, "ru")
mlsum_ru_train.to_csv("mlsum_ru_train.csv", index=False)

print("Processing Turkish dataset...")
mlsum_tu_train = process_language_dataset(mlsum_tu, "tu")
mlsum_tu_train.to_csv("mlsum_tu_train.csv", index=False)

print("Merging all datasets...")
merged_df = pd.concat([
    mlsum_de_train,
    mlsum_fr_train,
    mlsum_es_train,
    mlsum_ru_train,
    mlsum_tu_train
], ignore_index=True)

print("Processing queries...")
merged_df['queries'] = merged_df['queries'].apply(last_query)

if 'queries_noised' in merged_df.columns:
    merged_df['queries_noised'] = merged_df['queries_noised'].apply(last_query)
    merged_df = merged_df.rename(columns={"queries": "query", "queries_noised": "query_noised"})
else:
    merged_df = merged_df.rename(columns={"queries": "query"})

print("Saving merged dataset...")
merged_df.to_csv("q_to_summary.csv", index=False)



print("Applying noise to dataset...")
sample_dataset_corrupted = apply_noise_to_dataframe(merged_df)

print("Saving final corrupted dataset...")
sample_dataset_corrupted.to_csv('query_doc_dataset_random_noise.csv', index=False)

print("Processing complete!")
