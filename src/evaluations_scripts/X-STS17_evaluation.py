# Paper: LREC 2026
# Evaluates: Cross-lingual STS17 (ar-en, en-es, es-en, tr-en)
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from scipy.stats import spearmanr

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def find_model_directories(prefix):
    model_base_dir = "./trained_models"
    return [os.path.join(model_base_dir, d) for d in os.listdir(model_base_dir) if d.startswith(prefix)]

def evaluate_sts_task(model, sts_es_en, sts_ar_en, sts_tr_en, device, use_query_prefix=False):
    def add_prefix(texts, prefix):
        return [f"{prefix}{text}" for text in texts]

    # Process each dataset with prefixes if applicable
    def process_sts_data(sentences1, sentences2, scores, prefix=""):
        if prefix:
            sentences1 = add_prefix(sentences1, prefix)
            sentences2 = add_prefix(sentences2, prefix)
        embeddings1 = model.encode(sentences1, convert_to_tensor=True, device=device)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True, device=device)
        cosine_scores = util.cos_sim(embeddings1, embeddings2).diag().cpu().numpy()
        return spearmanr(cosine_scores, scores).correlation * 100

    # Evaluate and calculate Spearman correlation for each dataset
    spearman_corr_es_en = process_sts_data(
        sts_es_en['es'].tolist(), sts_es_en['en'].tolist(), sts_es_en['similarity_score'].tolist(), "query: " if use_query_prefix else ""
    )
    spearman_corr_ar_en = process_sts_data(
        sts_ar_en['ar'].tolist(), sts_ar_en['en'].tolist(), sts_ar_en['similarity_score'].tolist(), "query: " if use_query_prefix else ""
    )
    spearman_corr_tr_en = process_sts_data(
        sts_tr_en['tr'].tolist(), sts_tr_en['en'].tolist(), sts_tr_en['similarity_score'].tolist(), "query: " if use_query_prefix else ""
    )

    avg_spearman_corr = (spearman_corr_es_en + spearman_corr_ar_en + spearman_corr_tr_en) / 3
    return spearman_corr_es_en, spearman_corr_ar_en, spearman_corr_tr_en, avg_spearman_corr

def process_and_save_results(tasks, sts_es_en, sts_ar_en, sts_tr_en):
    for task in tasks:
        results = []
        task_name = task['name']
        prefixes = task['prefixes']

        print("Processing Task Name: " + task_name)

        save_name = task_name.replace(' ', '-') + "-sts-evaluations.csv"

        for model_prefix in tqdm(prefixes, desc=f"Processing Task {task_name}"):
            model_dirs = find_model_directories(model_prefix)
            if not model_dirs:
                print(f"No models found for prefix '{model_prefix}'")
                continue
            prefix_results = []

            for model_dir in model_dirs:
                try:
                    model = SentenceTransformer(model_dir, trust_remote_code=True)
                    model.to(device)
                except ValueError as e:
                    print(f"Error loading model '{model_dir}': {e}")
                    continue

                # Evaluate the model on the STS tasks
                spearman_es_en, spearman_ar_en, spearman_tr_en, avg_spearman = evaluate_sts_task(
                    model, sts_es_en, sts_ar_en, sts_tr_en, device, use_query_prefix=True
                )

                prefix_results.append({
                    'Model': model_dir,
                    'Task': task_name,
                    'Spearman ES-EN': spearman_es_en,
                    'Spearman AR-EN': spearman_ar_en,
                    'Spearman TR-EN': spearman_tr_en,
                    'Average Spearman': avg_spearman
                })

            # Average results for models under the same prefix
            prefix_results_df = pd.DataFrame(prefix_results)
            if not prefix_results_df.empty:
                avg_results = prefix_results_df.groupby(['Task'])[
                    ['Spearman ES-EN', 'Spearman AR-EN', 'Spearman TR-EN', 'Average Spearman']
                ].mean().reset_index()
                avg_results['Model Prefix'] = model_prefix
                results.extend(avg_results.to_dict('records'))

        # Save averaged results for this task
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_name, index=False)
        print(f"Results for task '{task_name}' and model prefix saved to {save_name}")

if __name__ == '__main__':
    # Define STS evaluation data
    sts_es_en = pd.read_csv("./evaluation_sets/taken_evaluation_sets/sts17_en-es.csv")
    sts_ar_en = pd.read_csv('./evaluation_sets/taken_evaluation_sets/sts17_ar-en.csv')
    sts_tr_en = pd.read_csv('./evaluation_sets/taken_evaluation_sets/sts17_tr-en.csv')

    # Define tasks here
    tasks = [
        {
            'name': 'x-sts-mgte_mono_real',
            'prefixes': [
                'Alibaba-NLP_gte-multilingual-base-mono_bl_real-10000-samples',
            ]
        },
    ]

    # Run the evaluation process
    process_and_save_results(tasks, sts_es_en, sts_ar_en, sts_tr_en)