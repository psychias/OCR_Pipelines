# Paper: LREC 2026
# Evaluates: Historical Bitext Mining (lb-de/en/fr)
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.distance import cdist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def find_model_directories(prefix):
    model_base_dir = "trained_models"
    return [os.path.join(model_base_dir, d) for d in os.listdir(model_base_dir) if d.startswith(prefix)]

def numpy_cosine_similarity(v1, v2):
    v1 = np.squeeze(v1)
    v2 = np.squeeze(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def evaluate_historical_bitext_mining_task(bitext_data, model, sentence_embeddings=None):
    if sentence_embeddings is None:
        unique_sentences = set(
            sentence for entry in bitext_data for sentence in [entry["source_sentence"]] + entry["candidates"]
        )
        sentence_embeddings = {sentence: model.encode(sentence) for sentence in unique_sentences}

    correct_count = 0

    for entry in bitext_data:
        source_embedding = sentence_embeddings[entry["source_sentence"]]
        parallel_embedding = sentence_embeddings[entry["candidates"][0]]
        parallel_similarity = numpy_cosine_similarity([source_embedding], [parallel_embedding])

        candidate_embeddings = [sentence_embeddings[candidate] for candidate in entry["candidates"][1:]]

        similarities = 1 - cdist(candidate_embeddings, source_embedding.reshape(1, -1), metric="cosine").flatten()

        if np.max(similarities) < parallel_similarity:
            correct_count += 1

    accuracy = round(correct_count / len(bitext_data) * 100, 2)
    return accuracy, sentence_embeddings

def load_dataset(file_path):
    """
    Loads a dataset (list of dictionaries) from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: List of dictionaries loaded from the file.
    """
    dataset = []
    with open(file_path, "r") as file:
        for line in file:
            dataset.append(json.loads(line))
    return dataset

def process_and_save_results(tasks, bitext_files):
    results = []

    for task in tasks:
        task_name = task['name']
        prefixes = task['prefixes']
        print("Processing Task Name: " + task_name)
        save_name = f"{task_name.replace(' ', '-')}-bitext-evaluations_x_mono.csv"

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

                # Evaluate the model on each bitext dataset
                accuracy_results = {}
                for bitext_name, bitext_data in bitext_files.items():
                    accuracy, _ = evaluate_historical_bitext_mining_task(
                        bitext_data, model
                    )
                    accuracy_results[f'{bitext_name} Accuracy'] = accuracy

                prefix_results.append({
                    'Model': model_dir,
                    'Task': task_name,
                    **accuracy_results
                })

            prefix_results_df = pd.DataFrame(prefix_results)
            if not prefix_results_df.empty:
                accuracy_columns = [col for col in prefix_results_df.columns if 'Accuracy' in col]
                avg_results = prefix_results_df.groupby(['Task'])[accuracy_columns].mean().reset_index()
                avg_results['Model Prefix'] = model_prefix
                results.extend(avg_results.to_dict('records'))

        # Save averaged results for this task
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_name, index=False)
        print(f"Results for task '{task_name}' and model prefix saved to {save_name}")

if __name__ == '__main__':
    # Define tasks here
    tasks = [
        {
            'name': 'lux-bitext-mgte_snp_real',
            'prefixes': [
                'Alibaba-NLP_gte-multilingual-base-mono_snp_real-10000-samples-',
            ]
        },
        {

	    'name': 'lux-bitext-mgte_bl_real',
            'prefixes': [
                'Alibaba-NLP_gte-multilingual-base-mono_bl_real-10000-samples-',
            ]
        },
        {
	    'name': 'lux-bitext-mgte_x_mono',
            'prefixes': [
                'Alibaba-NLP_gte-multilingual-base-x_mono-10000-samples-',
            ]
        },
        {
	    'name': 'lux-bitext-mgte_cross',
            'prefixes': [
                'Alibaba-NLP_gte-multilingual-base-cross-10000-samples-',
            ]
        },
        {
	    'name': 'lux-bitext-mgte_cross_clean',
            'prefixes': [
                'Alibaba-NLP_gte-multilingual-base-cross_clean-10000-samples-',
            ]
        },
        {
	    'name': 'lux-bitext-mgte_mono',
            'prefixes': [
                'Alibaba-NLP_gte-multilingual-base-mono-10000-samples-',
            ]
        },
        {
	    'name': 'lux-bitext-mgte_mono_batches',
            'prefixes': [
                'Alibaba-NLP_gte-multilingual-base-mono_batches-10000-samples-',
            ]
        }
    ]

    # Define bitext data files for different evaluations
    bitext_files = {
        'Hist BIM DE -> LB' : load_dataset('evaluation/bitext_mining_task_de_to_lb.jsonl'),
        'Hist BIM LB -> DE' : load_dataset('evaluation/bitext_mining_task_lb_to_de.jsonl'),
        'Hist BIM FR -> LB' : load_dataset('evaluation/bitext_mining_task_fr_to_lb.jsonl'),
        'Hist BIM LB -> FR' : load_dataset('evaluation/bitext_mining_task_lb_to_fr.jsonl'),
	    'Hist BIM EN -> LB' : load_dataset('evaluation/bitext_mining_task_en_to_lb.jsonl'),
        'Hist BIM LB -> EN' : load_dataset('evaluation/bitext_mining_task_lb_to_en.jsonl'),
    }

    # Run the evaluation process
    process_and_save_results(tasks, bitext_files)
