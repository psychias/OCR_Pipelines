# Paper: LREC 2026
# Evaluates: ParaLux paraphrase evaluation (Luxembourgish)
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import os, re, gc, shutil, traceback, difflib
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer

_ts_re = re.compile(r'(\d{8}-\d{6})(?!.*\d)')

EVAL_ROOT = os.path.join(os.getcwd(), "trained_models")

# Load the ParaLux test dataset
paralux_test = load_dataset("fredxlpy/ParaLux", split="test")

# Now you can work with the paralux_test dataset
paralux_data = {}
paralux_data['anchor'] = paralux_test['anchor']
paralux_data['positive'] = paralux_test['paraphrase']
paralux_data['negative'] = paralux_test['not_paraphrase']

def find_model_directories(prefix):
    # Get the models directory inside the current working directory
    model_base_dir = os.path.join(os.getcwd(), "trained_models")

    # Check if the models directory exists
    if not os.path.exists(model_base_dir):
        return []

    # Return directories in "models/" that start with the given prefix
    return [
        os.path.join(model_base_dir, d)
        for d in os.listdir(model_base_dir)
        if os.path.isdir(os.path.join(model_base_dir, d)) and d.startswith(prefix)
    ]


def evaluate_paralux_performance(anchor, positive, negative, model):
    """
    Measures the frequency with which the model's embeddings yield a higher cosine similarity
    between anchor and positive than between anchor and negative examples.

    Parameters:
        anchor (list): List of anchor text strings.
        positive (list): List of positive/paraphrase text strings.
        negative (list): List of negative/not paraphrase text strings.
        model: A model with an `encode` method to generate text embeddings.

    Returns:
        float: The frequency of anchor-positive similarity being higher than anchor-negative similarity.
    """
    # Encode the texts
    anchor_embeddings = model.encode(anchor)
    positive_embeddings = model.encode(positive)
    negative_embeddings = model.encode(negative)

    # Compute cosine similarities
    anchor_positive_sim = cosine_similarity(anchor_embeddings, positive_embeddings).diagonal()
    anchor_negative_sim = cosine_similarity(anchor_embeddings, negative_embeddings).diagonal()

    # Calculate the percentage of samples where positive similarity > negative similarity
    higher_similarity_percentage = round(np.mean(anchor_positive_sim > anchor_negative_sim) * 100, 2)

    return higher_similarity_percentage


def run_lux_embeddings_evaluations(
    model,
    model_name,
    similarity_data,
    run_paralux_clf=True,
):
    """
    Combines template-based model evaluation and similarity frequency measurement.

    Parameters:
        model: A model object with an `.encode()` method for generating embeddings.
        model_name (str): Name of the model for display in the results.
        dataset: The dataset containing examples with 'text' and 'category' fields (required for evaluation).
        class_to_templates: A dictionary mapping categories to their list of templates (required for evaluation).
        similarity_data (dict): A dictionary containing 'anchor', 'positive', and 'negative' lists for similarity check.
        run_evaluation (bool): Whether to run the template-based evaluation.
        run_similarity_check (bool): Whether to run the similarity frequency check.

    Returns:
        dict: A dictionary containing results for evaluation and similarity checks.
    """
    results = {}

    if run_paralux_clf:
        if not (similarity_data and
                "anchor" in similarity_data and
                "positive" in similarity_data and
                "negative" in similarity_data):
            raise ValueError("A dictionary containing 'anchor', 'positive', and 'negative' lists must be provided for similarity check.")

        paralux_performance = evaluate_paralux_performance(
            similarity_data["anchor"],
            similarity_data["positive"],
            similarity_data["negative"],
            model
        )
        results["PARALux Accuracy"] = paralux_performance

    return results



def _boundary_prefix_match(name: str, prefix: str) -> bool:
    """
    True if `name` == prefix OR `name` starts with `prefix` followed by a dash.
    This prevents '...+TED' from matching '...+TED_Impresso'.
    """
    if not name.startswith(prefix):
        return False
    if len(name) == len(prefix):
        return True
    return name[len(prefix)] == '-'  # only accept "-steps..."


def _extract_ts(name: str):
    m = _ts_re.search(name)
    if not m: return None
    try: return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
    except: return None

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def _ensure_unzipped(zip_path: str, unzip_root: str) -> str:
    """Unpack a zip if needed and return the model dir containing modules.json."""
    base = os.path.splitext(os.path.basename(zip_path))[0]
    target_root = os.path.join(unzip_root, base)
    os.makedirs(target_root, exist_ok=True)
    # Only unpack if empty
    if not os.listdir(target_root):
        shutil.unpack_archive(zip_path, target_root)
    # If the archive created a subfolder, find the one with modules.json
    if os.path.exists(os.path.join(target_root, "modules.json")):
        return target_root
    for entry in os.listdir(target_root):
        p = os.path.join(target_root, entry)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "modules.json")):
            return p
    # Fallback: return root (SentenceTransformer will error if not a model folder)
    return target_root

def collect_model_targets(prefixes,
                          root,
                          include_hf_ids=True,
                          latest_per_prefix=True,
                          strict_prefix=True,      # <-- default to True now
                          include_zips=False,      # <-- all models already unzipped
                          debug=True,
                          suggest_topk=5):
    if not os.path.isdir(root):
        print(f"[warn] root does not exist or is not a dir: {root}")
        return []

    if debug:
        print("scanning root:", root)

    # gather only real dirs (ignore macOS ghost dirs)
    all_dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.startswith("._"):         # ignore macOS ghost entries
                continue
            all_dirs.append((d, os.path.join(dirpath, d)))

    matches = {}  # prefix -> list[(disp, path, ts, mtime, kind)]
    for p in prefixes:
        hits = []
        for d, full in all_dirs:
            ok = (_boundary_prefix_match(d, p) if strict_prefix
                  else (_norm(p) in _norm(d) or _norm(d).startswith(_norm(p))))
            if ok:
                ts = _extract_ts(d)
                try:
                    mtime = os.path.getmtime(full)
                except Exception:
                    mtime = 0
                hits.append((d, full, ts, mtime, "dir"))

        if hits:
            matches[p] = hits

    targets = []
    for p, items in matches.items():
        items_sorted = sorted(items, key=lambda x: (x[2] is not None, x[2], x[3]), reverse=True)
        if latest_per_prefix:
            disp, path, _, _, kind = items_sorted[0]
            targets.append((disp, path, kind))
        else:
            for disp, path, _, _, kind in items_sorted:
                targets.append((disp, path, kind))

    if include_hf_ids:
        for p in prefixes:
            if "/" in p:
                targets.append((p, p, "hf"))

    # de-dup keep order
    seen, final = set(), []
    for disp, path, kind in targets:
        key = (disp, path)
        if key not in seen:
            seen.add(key)
            final.append((disp, path, kind))

    if debug:
        print("\n[debug] matched targets:")
        for disp, path, kind in final:
            print(f" • {disp} ({kind}) -> {path}")

    return final


def evaluate_all_models(prefixes,
                        similarity_data,
                        root,
                        include_hf_ids=True,
                        latest_per_prefix=True,
                        strict_prefix=False,
                        include_zips=True,
                        unzip_root=None,
                        device=None,
                        save_csv_path=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    unzip_root = unzip_root or os.path.join(root, "_unzipped")
    os.makedirs(unzip_root, exist_ok=True)

    rows = []
    targets = collect_model_targets(prefixes,
                                    root=root,
                                    include_hf_ids=include_hf_ids,
                                    latest_per_prefix=latest_per_prefix,
                                    strict_prefix=strict_prefix,
                                    include_zips=include_zips,
                                    debug=True)

    if not targets:
        print("[warn] No model targets found.")
        return pd.DataFrame()

    print(f"\n[info] Will evaluate {len(targets)} models")
    for i, (disp, path_or_id, kind) in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] Loading: {path_or_id}")
        try:
            load_path = path_or_id
            if kind == "zip":
                load_path = _ensure_unzipped(path_or_id, unzip_root)

            model = SentenceTransformer(load_path, trust_remote_code=True).to(device)
            model.eval()
            with torch.no_grad():
                results = run_lux_embeddings_evaluations(
                    model=model,
                    model_name=disp,
                    similarity_data=similarity_data,
                    run_paralux_clf=True,
                )

            row = {
                "model_name": disp,
                "model_path_or_id": path_or_id,
                "load_kind": kind
            }
            row.update(results)
            rows.append(row)

        except Exception as e:
            print(f"[ERROR] {disp}: {e}")
            traceback.print_exc()
        finally:
            try: del model
            except: pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("model_name", "model_path_or_id", "load_kind")]
    if metric_cols:
        df = df[["model_name", "model_path_or_id", "load_kind"] + metric_cols]
        if "PARALux Accuracy" in df.columns:
            df = df.sort_values("PARALux Accuracy", ascending=False)

    print("\n=== Evaluation Summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.to_string(index=False))

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        print(f"[saved] -> {save_csv_path}")

    return df

ALL_PREFIXES = [
    # Alibaba family (exact)
    "Alibaba-NLP_gte-multilingual-base+TED",
    "Alibaba-NLP_gte-multilingual-base+TED_Impresso",
    "Alibaba-NLP_gte-multilingual-base+TED_Impresso_SUMDOCS",
    "Alibaba-NLP_gte-multilingual-base+TED_Impresso_SUMDOCS_SUMMARY",

    # impresso-project family (exact)
    "impresso-project_histlux-gte-multilingual-base+TED",
    "impresso-project_histlux-gte-multilingual-base+TED_Impresso",
    "impresso-project_histlux-gte-multilingual-base+TED_Impresso_SUMDOCS",
    "impresso-project_histlux-gte-multilingual-base+TED_Impresso_SUMDOCS_SUMMARY",


    # ---------- Hugging Face model IDs ----------
    "gte-multilingual-base",
    "histlux-gte-multilingual-base",

]


df_results = evaluate_all_models(
    prefixes=ALL_PREFIXES,
    similarity_data=paralux_data,
    root=EVAL_ROOT,
    include_hf_ids=True,     # keep if you also want HF IDs (those with '/')
    latest_per_prefix=True,  # one run per intended prefix
    strict_prefix=True,      # crucial to avoid prefix bleed
    include_zips=False,      # you said everything's unzipped
    save_csv_path=f"./results/paralux_eval.csv",
)
