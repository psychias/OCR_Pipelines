
from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import Dataset, load_dataset, concatenate_datasets
import json
import random
import os
import re
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer


# Specify the languages you want to use
languages = languages = ["fra_Latn", "eng_Latn", "deu_Latn", "zho_Hans", "rus_Cyrl", "ltz_Latn"]
def _flatten(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary. 
    Example:
        {'a': {'b': 1}} -> {'a_b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
# Load language-specific splits
train_datasets = {}
test_datasets = {}

for lang in languages:
  try:
    lang_dataset = load_dataset("mteb/sib200", lang)
    train_datasets[lang] = lang_dataset["train"]
    test_datasets[lang] = lang_dataset["test"]
  except Exception as e:
    print(f'there was an error: {e} in lang {lang}')
    continue

sib_train_dataset = concatenate_datasets([train_datasets[lang] for lang in languages if lang!='ltz_Latn'])
sib_test_dataset = test_datasets['ltz_Latn']

zero_shot_template_labels = ["An dësem Beispill geet et em ", "D'Thema vun dësem Text ass ", "Dëst Dokument beschäftegt sech mat "] # just add label at the end.

category_to_lux_map = {
    "science/technology": "Technologie.",
    "travel": "Reesen.",
    "politics": "Politik.",
    "health": "Gesondheet.",
    "entertainment": "Ennerhalung.",
    "geography": "Geographie.",
    "sports": "Sport."
}

class_to_templates = {}
for category in list(set(sib_train_dataset['category'])):
  if category == None:continue
  class_to_templates[category] = []
  for template in zero_shot_template_labels:
    class_to_templates[category].append(template + category_to_lux_map[category])
  class_to_templates[category].append(category_to_lux_map[category][:-1]) # one template that is just the label, no full stop
  class_to_templates[category].append("Hei gëtt iwwer " + category_to_lux_map[category][:-1] + " geschwat.") # one template that label is inbetween
  
  def load_dataset_local(file_path):
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


bitext_mining_data_fr_to_lb = load_dataset_local("src/prepared_bitext_mining_format/bitext_mining_task_fr_to_lb.jsonl")
bitext_mining_data_lb_to_fr = load_dataset_local("src/prepared_bitext_mining_format/bitext_mining_task_lb_to_fr.jsonl")

bitext_mining_data_en_to_lb = load_dataset_local("src/prepared_bitext_mining_format/bitext_mining_task_en_to_lb.jsonl")
bitext_mining_data_lb_to_en = load_dataset_local("src/prepared_bitext_mining_format/bitext_mining_task_lb_to_en.jsonl")

bitext_mining_data_de_to_lb = load_dataset_local("src/prepared_bitext_mining_format/bitext_mining_task_de_to_lb.jsonl")
bitext_mining_data_lb_to_de = load_dataset_local("src/prepared_bitext_mining_format/bitext_mining_task_lb_to_de.jsonl")


# Load the ParaLux test dataset
paralux_test = load_dataset("fredxlpy/ParaLux", split="test")

# Now you can work with the paralux_test dataset
paralux_data = {}
paralux_data['anchor'] = paralux_test['anchor']
paralux_data['positive'] = paralux_test['paraphrase']
paralux_data['negative'] = paralux_test['not_paraphrase']


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

def numpy_cosine_similarity(v1, v2):
    """Compute cosine similarity using NumPy."""
    v1 = np.squeeze(v1)
    v2 = np.squeeze(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_sib_topic_classification_with_templates(
    model,
    dataset,                       # HF Dataset or list of dicts with 'text' and either 'category' or 'label'
    class_to_templates: dict,
    text_field: str = "text",
    category_field: str = "category",
    label_field: str = "label",
):
    categories = list(class_to_templates.keys())
    id2category = {
    5: "sports",
    6: "travel",
    1: "geography",
    4: "science/technology",
    0: "entertainment",
    2: "health",
    3: "politics",}


    # Precompute template embeddings per category (shape per cat: [num_templates, d])
    precomputed_template_embeddings = {
        cat: model.encode(class_to_templates[cat]) for cat in categories
    }
    num_templates = len(next(iter(class_to_templates.values())))

    if hasattr(dataset, "column_names"):
        texts = dataset[text_field]
        # Prefer string category if present and not None; else map labels
        has_cat = category_field in dataset.column_names and \
                  any(x is not None for x in dataset[category_field])
        if has_cat:
            true_cats = dataset[category_field]
        else:
            assert id2category is not None, "Provide id2category when using integer labels."
            label_ids = dataset[label_field]
            true_cats = [id2category[int(i)] for i in label_ids]
    else:
        # generic iterable of dicts
        rows = list(dataset)
        texts = [r[text_field] for r in rows]
        if all(("category" in r and r["category"] is not None) for r in rows):
            true_cats = [r["category"] for r in rows]
        else:
            assert id2category is not None, "Provide id2category when using integer labels."
            true_cats = [id2category[int(r[label_field])] for r in rows]

    text_embs = model.encode(texts)  # (N, d)

    template_performance = []
    for t_idx in range(num_templates):
        # Stack the t_idx-th template for every category -> (C, d)
        template_stack = np.stack(
            [precomputed_template_embeddings[cat][t_idx] for cat in categories],
            axis=0
        )
        # Cosine similarities for all texts vs (C templates) -> (N, C)
        sims = cosine_similarity(text_embs, template_stack)

        pred_idx = sims.argmax(axis=1)
        pred_cats = [categories[i] for i in pred_idx]

        correct = sum(1 for gt, pr in zip(true_cats, pred_cats) if gt == pr)
        template_performance.append(correct / len(true_cats))

    average_performance = float(np.mean(template_performance))
    return template_performance, average_performance


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

def evaluate_historical_bitext_mining_task(bitext_data, model, sentence_embeddings = None):
    """
    Evaluates an embedding model on the bitext-mining task.

    Args:
        bitext_data (list): A list of dictionaries containing the bitext-mining task data.
        model (object): A model with a `.encode` method to compute sentence embeddings.

    Returns:
        float: The accuracy of the model on the bitext-mining task.
    """
    # Precompute embeddings for all unique sentences
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

        # Prepare the embeddings for all candidates
        candidate_embeddings = [sentence_embeddings[candidate] for candidate in entry["candidates"][1:]]

        # Use cdist for one-to-many cosine similarity computation
        similarities = 1 - cdist(candidate_embeddings, source_embedding.reshape(1, -1), metric="cosine").flatten()

        # Check if any similarity exceeds max_similarity
        if np.max(similarities) < parallel_similarity:
            correct_count += 1

    accuracy = round(correct_count / len(bitext_data) * 100,2)
    return accuracy, sentence_embeddings

def run_lux_embeddings_evaluations(
    model,
    model_name,
    sib200_dataset,
    sib200_class_to_templates,
    similarity_data,
    bitext_mining_datasets_de_lb,
    bitext_mining_datasets_fr_lb=None,
    bitext_mining_datasets_en_lb=None,
    run_sib_clf=True,
    run_paralux_clf=True,
    run_bitext_mining=True
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

    if run_sib_clf:
        template_performance, average_performance = evaluate_sib_topic_classification_with_templates(
            model, sib200_dataset, class_to_templates
        )
        results["Zero Shot SIB"] = round(average_performance * 100,2)
        # print(f"Zero Shot SIB (7 classes) Accuracy for {model_name}: {average_performance*100:.2f}%")

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
        # print(f"PARALux (300 samples) Accuracy for : {model_name}': {paralux_performance:.2f}%")
    if run_bitext_mining:
        de_lb_accuracy, sentence_embeddings = evaluate_historical_bitext_mining_task(bitext_mining_datasets_de_lb[0], model)
        results["Historical Bitext Mining DE -> LB"] = de_lb_accuracy
        # print(f"DE - > LB Historical Bitext Mining Accuracy (2170 Sentences) for {model_name}: " + str(de_lb_accuracy) + "%")
        lb_de_accuracy, sentence_embeddings = evaluate_historical_bitext_mining_task(bitext_mining_datasets_de_lb[1], model, sentence_embeddings)
        results["Historical Bitext Mining LB -> DE"] = lb_de_accuracy
        # print(f"LB - > DE Historical Bitext Mining Accuracy (2170 Sentences) for {model_name}: "  + str(lb_de_accuracy) + "%")

        fr_lb_accuracy, sentence_embeddings = evaluate_historical_bitext_mining_task(bitext_mining_datasets_fr_lb[0], model)
        results["Historical Bitext Mining FR -> LB"] = fr_lb_accuracy
        # print(f"DE - > LB Historical Bitext Mining Accuracy (471 Sentences) for {model_name}: " + str(fr_lb_accuracy) + "%")
        lb_fr_accuracy, sentence_embeddings = evaluate_historical_bitext_mining_task(bitext_mining_datasets_fr_lb[1], model, sentence_embeddings)
        results["Historical Bitext Mining LB -> FR"] = lb_fr_accuracy

        en_lb_accuracy, sentence_embeddings = evaluate_historical_bitext_mining_task(bitext_mining_datasets_en_lb[0], model)
        results["Historical Bitext Mining EN -> LB"] = en_lb_accuracy
        # print(f"DE - > LB Historical Bitext Mining Accuracy (471 Sentences) for {model_name}: " + str(fr_lb_accuracy) + "%")
        lb_en_accuracy, sentence_embeddings = evaluate_historical_bitext_mining_task(bitext_mining_datasets_en_lb[1], model, sentence_embeddings)
        results["Historical Bitext Mining LB -> EN"] = lb_en_accuracy
        # print(f"LB - > DE Historical Bitext Mining Accuracy (471 Sentences) for {model_name}: "  + str(lb_fr_accuracy) + "%")

    return results



class CustomEvaluator:
    def __init__(
        self,
        model_name,
        sib_test_dataset,
        class_to_templates,
        similarity_data,
        bitext_mining_datasets_de_lb,
        bitext_mining_datasets_fr_lb=None,
        bitext_mining_datasets_en_lb=None,
        log_file="evaluation_logs.jsonl"
    ):
        self.model_name = model_name
        self.sib_test_dataset = sib_test_dataset
        self.class_to_templates = class_to_templates
        self.similarity_data = similarity_data
        self.bitext_mining_datasets_de_lb = bitext_mining_datasets_de_lb
        self.bitext_mining_datasets_fr_lb = bitext_mining_datasets_fr_lb
        self.bitext_mining_datasets_en_lb = bitext_mining_datasets_en_lb

        # Get the directory of the log file
        log_dir = os.path.dirname(log_file)
        if log_dir:  # Only create directory if it exists
            os.makedirs(log_dir, exist_ok=True)

        self.log_file = log_file

        # Initialize the log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                pass  # Create an empty file


    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        # Run evaluations
        results = run_lux_embeddings_evaluations(
            model=model,
            model_name=self.model_name,
            sib200_dataset=self.sib_test_dataset,
            sib200_class_to_templates=self.class_to_templates,
            similarity_data=self.similarity_data,
            bitext_mining_datasets_de_lb=self.bitext_mining_datasets_de_lb,
            bitext_mining_datasets_fr_lb=self.bitext_mining_datasets_fr_lb,
            bitext_mining_datasets_en_lb=self.bitext_mining_datasets_en_lb,
            run_sib_clf=True,
            run_paralux_clf=True,
            run_bitext_mining=True
        )

        # Create a log entry
        log_entry = {
            "epoch": epoch,
            "steps": steps,
            "results": results
        }

        # Append the log entry to the JSONL file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")  # Write as a single line

        if output_path:
          os.makedirs(output_path, exist_ok=True)
          json_out = os.path.join(output_path, f"metrics_epoch{epoch}_step{steps}.json")
          with open(json_out, "w", encoding="utf-8") as jf:
              json.dump(log_entry, jf, ensure_ascii=False, indent=2)
          flat = {"epoch": epoch, "steps": steps, **_flatten(results)}
          csv_out = os.path.join(output_path, "metrics_summary.csv")
          write_header = not os.path.exists(csv_out)
          import csv
          with open(csv_out, "a", newline="", encoding="utf-8") as cf:
              w = csv.DictWriter(cf, fieldnames=list(flat.keys()))
              if write_header:
                  w.writeheader()
              w.writerow(flat)


        print(f"Logged evaluation at epoch {epoch}, step {steps} to {self.log_file}")
        return results["Historical Bitext Mining DE -> LB"]

def prepare_training_samples(train_samples_tuple, sample_size=5120, random_sampling=False, sorted_descending=False):
    """
    Prepares training samples for finetuning the embedding models.

    Args:
        train_samples_tuple (list): A list of tuples, each containing a pair of sentences and a custom_id.
        sample_size (int, optional): The number of samples to prepare. Defaults to 3000.
        random_sampling (bool, optional): If True, samples are randomly selected; otherwise, the first `sample_size` samples are taken. Defaults to False.
        sorted_descending (bool, optional): If True, samples are sorted in descending order of (joint) character length. Defaults to False.

    Returns:
        list: A list of `InputExample` objects containing the training samples.
    """
    if sorted_descending:
        train_samples_tuple = sorted(train_samples_tuple, key=lambda pair: len(pair[0]) + len(pair[1]), reverse=True)
    if random_sampling:
        train_samples_tuple = random.sample(train_samples_tuple, min(sample_size, len(train_samples_tuple))) # is also shuffled
    else:
        train_samples_tuple = train_samples_tuple[:sample_size]

    train_samples = [InputExample(texts=[sample[0], sample[1]], label=1.0) for sample in train_samples_tuple]
    return train_samples


def clean_text_from_punctuation(text):
        """
        Cleans text by removing random punctuation and extra whitespace.

        Args:
            text (str): The text to clean.

        Returns:
            str: Cleaned text containing only alphanumeric characters and preserving spaces.
        """
        return re.sub(r'[^a-zA-Z0-9\s]+', '', text).strip().lower()

def extract_parallel_sentences(lines, src_col, tgt_col):
    """
    Extracts parallel sentence pairs (lb, de) and preserves the custom_id from the JSONL lines.

    Args:
        lines (list): A list of dictionaries, where each dictionary represents a line from the file.

    Returns:
        list: A list of tuples containing sentence pairs (lb_sentence, de_sentence, custom_id).
    """
    parallel_sentences = []
    for data in lines:
        custom_id = data.get('custom_id', None)
        translations = data.get('translation', {})
        for sentence_pair in translations:
            if isinstance(sentence_pair, dict):
                src_sentence = sentence_pair.get(src_col)
                tgt_sentence = sentence_pair.get(tgt_col)
                if type(src_sentence) == list or type(tgt_sentence) == list: # to handle a special case of ['']
                    continue
                if (src_sentence and tgt_sentence)  and (len(clean_text_from_punctuation(src_sentence)) >= 5 and len(clean_text_from_punctuation(tgt_sentence)) >= 5): # must contain atleast 5 alphnum characters
                    parallel_sentences.append((src_sentence, tgt_sentence, custom_id))
    return parallel_sentences


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import json, csv, os

RESULTS_DIR = "./results/"
in_path = os.path.join(RESULTS_DIR, "mgte_evaluation_logs_en_mixed.jsonl")
out_path = "src/results/evaluation_summary.csv"

def flatten(d, parent="", sep="."):
    out = {}
    for k, v in d.items():
        kk = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten(v, kk, sep))
        else:
            out[kk] = v
    return out

rows = []
with open(in_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        rec = json.loads(line)
        flat = {"epoch": rec.get("epoch"), "steps": rec.get("steps")}
        flat.update(flatten(rec.get("results", {})))
        rows.append(flat)

fields = sorted({k for r in rows for k in r.keys()})
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print("Wrote:", os.path.abspath(out_path))


import os
import pandas as pd
from sentence_transformers import util
import torch
from tqdm import tqdm
from itertools import product


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def find_model_directories(prefix):
    model_base_dir = os.path.join(os.getcwd(), "trained_models")

    if not os.path.exists(model_base_dir):
        return []

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
        results = []  #
        task_name = task['name']
        prefixes = task['prefixes']
        levels = task['levels']
        prefix = task.get('prefix', "")
        versions_dict = task['versions_dict']
        eval_dataset = task['eval_dataset']

        print("Processing Task Name: " + task_name)

        save_name = "Alibaba-NLP_gte-multilingual-base-mixed8-seed" + "-".join(levels.keys()) + eval_dataset + f"-{task_name.replace(' ', '-')}-evaluations.csv"

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

            prefix_results_df = pd.DataFrame(prefix_results)
            if not prefix_results_df.empty:
                avg_results = prefix_results_df.groupby(
                    ['Comparison', 'Corruption Level', 'Language Direction']
                )['Accuracy'].mean().reset_index()
                avg_results['Model Prefix'] = model_prefix
                results.extend(avg_results.to_dict('records'))

        results_df = pd.DataFrame(results)
        results_df.to_csv("src/results/"+save_name, index=False)
        print(f"Results for task '{task_name}' and model prefix saved to {save_name}")

tasks = [
    {
        'name': 'Simple Noise',
        'prefixes': [
            "Alibaba-NLP_gte-multilingual-base-mixed8-seed",
        ],
        'levels': {
            'clean': 'src/evaluation_sets/CLSD_wmt2019_adversarial_dataset.csv',
            'simple': 'src/evaluation_sets/CLSD_WMT19_MN_noise.csv'
        },
        'versions_dict': {
            'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
            'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
        },
        'prefix': "",
        'eval_dataset': "CLSD-wmt19-"
    },
    {
        'name': 'Simple Noise',
        'prefixes': [
            "Alibaba-NLP_gte-multilingual-base-mixed8-seed",
        ],
        'levels': {
            'clean': 'src/evaluation_sets/CLSD_wmt2021_adversarial_dataset.csv',
            'simple': 'src/evaluation_sets/CLSD_WMT21_MN_noise.csv'
        },
        'versions_dict': {
            'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
            'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
        },
        'prefix': "",
        'eval_dataset': "CLSD-wmt21-"
    },
    {
        'name': 'Blackletter-Scanned Noise',
        'prefixes': [
            "Alibaba-NLP_gte-multilingual-base-mixed8-seed",
        ],
        'levels': {
            'clean': 'src/evaluation_sets/CLSD_wmt2019_adversarial_dataset.csv',
            'bl-distorted': 'src/evaluation_sets/CLSD_WMT19_BLDS_noise.csv'
        },
        'versions_dict': {
            'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
            'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
        },
        'prefix': "",
        'eval_dataset': "CLSD-wmt19-"
    },
    {
        'name': 'Blackletter-Scanned Noise',
        'prefixes': [
            "Alibaba-NLP_gte-multilingual-base-mixed8-seed",
        ],
        'levels': {
            'clean': 'src/evaluation_sets/CLSD_wmt2021_adversarial_dataset.csv',
            'bl-distorted': 'src/evaluation_sets/CLSD_WMT21_BLDS_noise.csv'
        },
        'versions_dict': {
            'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
            'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
        },
        'prefix': "",
        'eval_dataset': "CLSD-wmt21-"
    },
    {
        'name': 'SaltnPepper Noise',
        'prefixes': [
            "Alibaba-NLP_gte-multilingual-base-mixed8-seed",
        ],
        'levels': {
            'clean': 'src/evaluation_sets/CLSD_wmt2019_adversarial_dataset.csv',
            'SaltnPepper': 'src/evaluation_sets/CLSD_WMT19_SNP_noise.csv'
        },
        'versions_dict': {
            'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
            'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
        },
        'prefix': "",
        'eval_dataset': "CLSD-wmt19-"
    },
    {
        'name': 'SaltnPepper Noise',
        'prefixes': [
            "Alibaba-NLP_gte-multilingual-base-mixed8-seed",
        ],
        'levels': {
            'clean': 'src/evaluation_sets/CLSD_wmt2021_adversarial_dataset.csv',
            'SaltnPepper': 'src/evaluation_sets/CLSD_WMT21_SNP_noise.csv'
        },
        'versions_dict': {
            'German': ['French', 'fr_adv1', 'fr_adv2', 'fr_adv3', 'fr_adv4'],
            'French': ['German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4']
        },
        'prefix': "",
        'eval_dataset': "CLSD-wmt21-"
    }
]

process_and_save_results(tasks)

import os
import csv
import numpy as np
import scipy.stats as stats
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


model_dir_prefix = "Alibaba-NLP_gte-multilingual-base-mixed8-seed"
models_root = "./trained_models/"
pairs = [("en", "tr"), ("en", "es"), ("en", "de"),("es", "en"),("fr", "en"),("en", "de")]
sts_dataset_name = "mteb/sts17-crosslingual-sts"
batch_size = 64
max_seq_length = 512
out_csv = f"./results/xsts_results.csv"

# -----------------------
# Helpers
# -----------------------
def _standardize_row(row):
    """Map dataset row to (s1, s2, score) regardless of schema."""
    # Common schemas: 'sentence1'/'sentence2'/'score' OR 'text_1'/'text_2'/'labels'
    if "sentence1" in row and "sentence2" in row and "score" in row:
        return row["sentence1"], row["sentence2"], float(row["score"])
    if "text_1" in row and "text_2" in row and "labels" in row:
        return row["text_1"], row["text_2"], float(row["labels"])
    # Last-ditch: try generic keys
    keys = list(row.keys())
    s1 = row.get("s1") or row.get("src") or row.get("text1") or row.get(keys[0])
    s2 = row.get("s2") or row.get("tgt") or row.get("text2") or row.get(keys[1])
    sc = row.get("score") or row.get("labels") or row.get("label") or row.get(keys[2])
    return s1, s2, float(sc)

def load_xsts_split(dataset_name, src, tgt):
    """Load X-STS for a pair, trying src-tgt then tgt-src; fall back to local TSV."""
    # Try HF dataset in both directions (some repos only host one orientation)
    for cfg in (f"{src}-{tgt}", f"{tgt}-{src}"):
        try:
            ds = load_dataset(dataset_name, cfg, split="test")
            orientation = "forward" if cfg == f"{src}-{tgt}" else "reversed"
            return ds, orientation
        except Exception:
            pass

    # Fallback to local TSV: STS_en_tr.tsv (tab: sent1, sent2, score)
    local_path_1 = f"STS_{src}_{tgt}.tsv"
    local_path_2 = f"STS_{tgt}_{src}.tsv"
    for pth, orientation in ((local_path_1, "forward"), (local_path_2, "reversed")):
        if os.path.exists(pth):
            rows = []
            with open(pth, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 3:
                        rows.append({"sentence1": parts[0], "sentence2": parts[1], "score": float(parts[2])})
            return rows, orientation

    raise FileNotFoundError(
        f"Could not load X-STS for {src}-{tgt}. Checked HF ({dataset_name}) and local TSVs."
    )

def eval_model_on_pairs(model: SentenceTransformer, pairs):
    model.eval()
    correlations = []

    with torch.inference_mode():
        for (src, tgt) in pairs:
            data, orientation = load_xsts_split(sts_dataset_name, src, tgt)

            # Extract fields
            if isinstance(data, list):
                # local list of dicts
                sents1 = [d["sentence1"] for d in data]
                sents2 = [d["sentence2"] for d in data]
                gold_scores = np.array([float(d["score"]) for d in data], dtype=np.float32)
            else:
                # HF Dataset
                sents1, sents2, gold_scores = [], [], []
                for row in data:
                    a, b, sc = _standardize_row(row)
                    sents1.append(a)
                    sents2.append(b)
                    gold_scores.append(sc)
                gold_scores = np.array(gold_scores, dtype=np.float32)

            # If we loaded reversed orientation but pair is src->tgt, we still just compute correlation;
            # Spearman is invariant to monotonic transforms, so orientation won’t matter for ranking.
            emb1 = model.encode(
                sents1, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            emb2 = model.encode(
                sents2, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )

            assert emb1.shape == emb2.shape == (len(gold_scores), emb1.shape[1]), "Embedding length mismatch."

            sims = np.sum(emb1 * emb2, axis=1)  # cosine since we normalized
            # Guard against pathological NaNs
            mask = ~np.isnan(sims)
            spear = stats.spearmanr(sims[mask], gold_scores[mask]).correlation
            correlations.append((f"{src}-{tgt}", float(spear)))
    return correlations


def main():
    # discover models
    model_dirs = [
        os.path.join(models_root, d)
        for d in os.listdir(models_root)
        if d.startswith(model_dir_prefix) and os.path.isdir(os.path.join(models_root, d))
    ]
    model_dirs.sort()

    if not model_dirs:
        raise RuntimeError(f"No model directories found in '{models_root}' with prefix '{model_dir_prefix}'.")

    # prepare CSV
    header = ["Model"] + [f"Spearman_{src}-{tgt}" for (src, tgt) in pairs] + ["Spearman_Avg"]
    rows = []

    for model_path in model_dirs:
        model_name = os.path.basename(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n=== Evaluating {model_name} on {device} ===")
        model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        model.max_seq_length = max_seq_length

        corrs = eval_model_on_pairs(model, pairs)

        # Log per-pair and average
        pair_to_score = {p: s for p, s in corrs}
        for p, s in corrs:
            print(f"[{model_name}] {p} Spearman: {s:.4f}")

        scores = [pair_to_score[f"{src}-{tgt}"] for (src, tgt) in pairs]
        avg_score = float(np.mean(scores))
        rows.append([model_name] + [f"{s:.4f}" for s in scores] + [f"{avg_score:.4f}"])

    # write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nX-STS evaluation completed. Results saved to {out_csv}.")

main()


import os, math
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Parameters
model_dir_prefix = "Alibaba-NLP_gte-multilingual-base-mixed8-seed"
languages = ["en", "de", "fr","tu","ru","es","it"]
max_length = 512

models_root = "./trained_models"
model_dirs = [os.path.join(models_root, d) for d in os.listdir(models_root)
              if d.startswith(model_dir_prefix) and os.path.isdir(os.path.join(models_root, d))]

header = "Model"
for lang in languages:
    header += f", {lang}_NDCG@10, {lang}_MRR"
out_lines = [header]

# Helper: compute NDCG@10 for a list of ranks (1-indexed)
def ndcg_at_10(ranks):
    # If rank > 10 (not retrieved in top10), contribution is 0
    # NDCG@10 for a single query with a single relevant item at 'rank':
    if ranks > 10:
        return 0.0
    # relevance = 1 for the one relevant document
    return 1.0 / math.log2(ranks + 1)

for model_path in model_dirs + ["BM25_BASELINE"]:
    model_name = os.path.basename(model_path) if model_path != "BM25_BASELINE" else "BM25_baseline"
    print(f"Evaluating model: {model_name}")
    metrics = {lang: {"ndcg10": 0.0, "mrr": 0.0} for lang in languages}
    total_queries = {lang: 0 for lang in languages}

    # If model_path is a real model, load it; if it's BM25 baseline, we will handle separately
    model = None
    if model_path != "BM25_BASELINE":
        model = SentenceTransformer(model_path, device=('cuda' if torch.cuda.is_available() else 'cpu'))
        model.max_seq_length = max_length  # set max length for encoding

    for lang in languages:
        # Load MLDR data for this language
        data = load_dataset("Shitao/MLDR", lang)
        # The dataset has 'train', 'dev', 'test' splits and also a 'corpus' subset:
        test_queries = data['test']       # list of {'query_id', 'query', 'positive_passages': [...], 'negative_passages': [...]}
        corpus = data[f"corpus-{lang}"]   # list of {'docid', 'text'} for all documents in the corpus

        # Prepare corpus documents and a lookup for docid -> index
        corpus_texts = [doc["text"] for doc in corpus]
        docid_to_index = {doc["docid"]: idx for idx, doc in enumerate(corpus)}

        # Encode corpus documents with embedding model or prepare BM25 index
        corpus_embeddings = None
        bm25 = None
        if model is not None:
            # Embed all documents in corpus (potentially large, so we do in batches)
            corpus_embeddings = model.encode(corpus_texts, batch_size=16, convert_to_numpy=True)
            # Normalize embeddings for cosine similarity
            corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        else:
            # BM25 baseline: initialize inverted index
            tokenized_docs = [text.split() for text in corpus_texts]  # simple whitespace tokenization
            bm25 = BM25Okapi(tokenized_docs)

        # Evaluate each query
        ndcg_sum = 0.0
        mrr_sum = 0.0
        for query in test_queries:
            q_text = query["query"]
            # The positive passage list contains the relevant doc (one entry in test set)
            pos_docid = query["positive_passages"][0]["docid"]
            relevant_index = docid_to_index[pos_docid]

            # Get similarity scores for all docs
            if model is not None:
                q_emb = model.encode(q_text, convert_to_numpy=True)
                q_emb = q_emb / np.linalg.norm(q_emb)  # normalize
                # Compute cosine similarities between query and all corpus embeddings
                scores = np.dot(corpus_embeddings, q_emb)  # cosine similarity since both normalized
            else:
                # BM25: get relevance scores for query
                scores = bm25.get_scores(q_text.split())
            # Rank documents by score (higher is more similar/relevant)
            ranked_indices = np.argsort(-scores)  # indices sorted by descending score
            rank_of_relevant = int(np.where(ranked_indices == relevant_index)[0][0]) + 1  # 1-indexed rank

            # Compute metrics
            total_queries[lang] += 1
            # NDCG@10: only one relevant doc, calculate based on its rank
            ndcg_val = 0.0
            if rank_of_relevant <= 10:
                ndcg_val = 1.0 / math.log2(rank_of_relevant + 1)
            ndcg_sum += ndcg_val
            # Reciprocal rank
            mrr_sum += 1.0 / rank_of_relevant

        # Average metrics for this language
        if total_queries[lang] > 0:
            metrics[lang]["ndcg10"] = ndcg_sum / total_queries[lang]
            metrics[lang]["mrr"] = mrr_sum / total_queries[lang]
        print(f"  [{lang}] NDCG@10 = {metrics[lang]['ndcg10']:.4f}, MRR = {metrics[lang]['mrr']:.4f}")

    # Append to results
    line = model_name
    for lang in languages:
        line += f", {metrics[lang]['ndcg10']:.4f}, {metrics[lang]['mrr']:.4f}"
    out_lines.append(line)

# Save to CSV
with open("src/results/mldr_results.csv", "w", encoding="utf-8") as fout:
    fout.write("\n".join(out_lines))

print("MLDR evaluation completed. Results in mldr_results.csv.")

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
    model_base_dir = "./trained_models"
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
        save_name = f"./results/{task_name.replace(' ', '-')}-bitext-evaluations_x_mono.csv"

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
	    'name': 'lux-bitext-mgte_mix_training',
            'prefixes': [
                'Alibaba-NLP_gte-multilingual-base-mixed8-seed',
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
        'Hist BIM DE -> LB' : load_dataset('src/evaluation_sets/bitext_mining_task_de_to_lb.jsonl'),
        'Hist BIM LB -> DE' : load_dataset('src/evaluation_sets/evaluation_sets/bitext_mining_task_lb_to_de.jsonl'),
        'Hist BIM FR -> LB' : load_dataset('src/evaluation_sets/evaluation_sets/bitext_mining_task_fr_to_lb.jsonl'),
        'Hist BIM LB -> FR' : load_dataset('src/evaluation_sets/evaluation_sets/bitext_mining_task_lb_to_fr.jsonl'),
	      'Hist BIM EN -> LB' : load_dataset('src/evaluation_sets/evaluation_sets/bitext_mining_task_en_to_lb.jsonl'),
        'Hist BIM LB -> EN' : load_dataset('src/evaluation_sets/evaluation_sets/bitext_mining_task_lb_to_en.jsonl'),
    }

    # Run the evaluation process
    process_and_save_results(tasks, bitext_files)
