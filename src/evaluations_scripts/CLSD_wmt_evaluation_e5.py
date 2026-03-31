# Paper: ACL 2025
# Evaluates: CLSD evaluation with E5 model variants
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from itertools import product

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def find_model_directories(prefix):
    # Get the models directory inside the current working directory
    model_base_dir = os.path.join(os.getcwd(), "trained_models")
    
    # Check if the models directory exists
    if not os.path.exists(model_base_dir):
        return []
    
    # Return directories in "models/" that start with the given prefix
    return [os.path.join(model_base_dir, d) for d in os.listdir(model_base_dir) if os.path.isdir(os.path.join(model_base_dir, d)) and d.startswith(prefix)]

def encode_texts(model, texts, prefix=""):
    return model.encode([prefix + t for t in texts], convert_to_tensor=True).to(device)

def compare_languages(model, left_df, right_df, main_col, versions, prefix=""):
    main_embeddings = encode_texts(model, left_df[main_col].tolist(), prefix)
    results = pd.DataFrame()

    for version in versions:
        version_embeddings = encode_texts(model, right_df[version].tolist(), prefix)
        similarity_scores = util.cos_sim(main_embeddings, version_embeddings).diag().cpu().numpy()
        results[version] = similarity_scores

    max_scores = results.idxmax(axis=1)
    max_values = results.max(axis=1)
    is_max = results.eq(max_values, axis=0)
    max_counts = is_max.sum(axis=1)
    unique_max = max_counts == 1
    correct_version = versions[0]
    correct = (max_scores == correct_version) & unique_max

    return correct.mean() * 100

def process_and_save_results(tasks):
    for task in tasks:
        results = []  # Reset results for each task
        task_name = task['name']
        prefixes = task['prefixes']
        levels = task['levels']
        prefix = task.get('prefix', "")
        versions_dict = task['versions_dict']
        eval_dataset = task['eval_dataset']

        print("Processing Task Name: " + task_name)

        save_name = "cross_clean_multilingual-e5-base" + "-".join(levels.keys()) + eval_dataset + f"-{task_name.replace(' ', '-')}-evaluations.csv"

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

                comparisons = [
                    {
                        'name': f'{left.capitalize()} to {right.capitalize()}',
                        'left_levels': [left],
                        'right_levels': [right],
                    }
                    for left, right in product(levels.keys(), repeat=2)
                ]

                for comparison in comparisons:
                    for left_level in comparison['left_levels']:
                        for right_level in comparison['right_levels']:
                            left_df = pd.read_csv(levels[left_level])
                            right_df = pd.read_csv(levels[right_level])

                            for compare_col in versions_dict:
                                versions = versions_dict[compare_col]
                                if not all(v in right_df.columns for v in versions):
                                    print(f"Missing columns for {versions} in {levels[right_level]}")
                                    continue

                                percentage = compare_languages(
                                    model, left_df, right_df, compare_col, versions, prefix
                                )

                                left_lang = 'DE' if compare_col == 'German' else 'FR'
                                right_lang = 'FR' if compare_col == 'German' else 'DE'
                                language_direction = f"{left_lang}_{left_level}->{right_lang}_{right_level}"

                                prefix_results.append({
                                    'Model': model_dir,
                                    'Comparison': comparison['name'],
                                    'Corruption Level': f"{left_level} to {right_level}",
                                    'Language Direction': language_direction,
                                    'Accuracy': percentage
                                })

            # Average results for models under the same prefix
            prefix_results_df = pd.DataFrame(prefix_results)
            if not prefix_results_df.empty:
                avg_results = prefix_results_df.groupby(
                    ['Comparison', 'Corruption Level', 'Language Direction']
                )['Accuracy'].mean().reset_index()
                avg_results['Model Prefix'] = model_prefix
                results.extend(avg_results.to_dict('records'))

        # Save averaged results for this task
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_name, index=False)
        print(f"Results for task '{task_name}' and model prefix saved to {save_name}")

if __name__ == '__main__':
    tasks = [
        {
            'name': 'Simple Noise',
            'prefixes': [
                'intfloat_multilingual-e5-base-cross_clean-10000-samples-',
            ],
            'levels': {
                'clean': './evaluation/wmt2019_adversarial_dataset.csv',
                'simple': './evaluation/simple_19.csv'
            },
            'versions_dict': {
                'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
                'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
            },
            'prefix': "query: ",
            'eval_dataset': "CLSD-wmt19-"
        },
        {
            'name': 'Simple Noise',
            'prefixes': [
                'intfloat_multilingual-e5-base-cross_clean-10000-samples-',
            ],
            'levels': {
                'clean': './evaluation/wmt2021_adversarial_dataset.csv',
                'simple': './evaluation/simple_21.csv'
            },
            'versions_dict': {
                'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
                'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
            },
            'prefix': "query: ",
            'eval_dataset': "CLSD-wmt21-"
        },
        {
            'name': 'Blackletter-Scanned Noise',
            'prefixes': [
                'intfloat_multilingual-e5-base-cross_clean-10000-samples-',
            ],
            'levels': {
                'clean': './evaluation/wmt2019_adversarial_dataset.csv',
                'bl-distorted': './evaluation/blackletter+distort_19.csv'
            },
            'versions_dict': {
                'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
                'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
            },
            'prefix': "query: ",
            'eval_dataset': "CLSD-wmt19-"
        },
        {
            'name': 'Blackletter-Scanned Noise',
            'prefixes': [
                'intfloat_multilingual-e5-base-cross_clean-10000-samples-',
            ],
            'levels': {
                'clean': './evaluation/wmt2021_adversarial_dataset.csv',
                'bl-distorted': './evaluation/blackletter+distort_21.csv'
            },
            'versions_dict': {
                'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
                'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
            },
            'prefix': "query: ",
            'eval_dataset': "CLSD-wmt21-"
        },
        {
            'name': 'SaltnPepper Noise',
            'prefixes': [
                'intfloat_multilingual-e5-base-cross_clean-10000-samples-',
            ],
            'levels': {
                'clean': './evaluation/wmt2019_adversarial_dataset.csv',
                'SaltnPepper': './evaluation/noise_19.csv'
            },
            'versions_dict': {
                'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
                'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
            },
            'prefix': "query: ",
            'eval_dataset': "CLSD-wmt19-"
        },
        {
            'name': 'SaltnPepper Noise',
            'prefixes': [
                'intfloat_multilingual-e5-base-cross_clean-10000-samples-',
            ],
            'levels': {
                'clean': './evaluation/wmt2021_adversarial_dataset.csv',
                'SaltnPepper': './evaluation/noise_21.csv'
            },
            'versions_dict': {
                'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
                'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
            },
            'prefix': "query: ",
            'eval_dataset': "CLSD-wmt21-"
        }
    ]

    process_and_save_results(tasks)
