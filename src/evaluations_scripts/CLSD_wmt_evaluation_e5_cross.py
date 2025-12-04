import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from tqdm import tqdm
from itertools import product

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def pandas_row_to_string_key(row):
    return (
        row["RUN_ID"]
        + " $ "
        + row["Comparison"]
        + " $ "
        + row["Corruption Level"]
        + " $ "
        + row["Language Direction"]
    )


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


def encode_texts(overall_embeddings, model, texts, prefix=""):
    prefixed_texts = [prefix + t for t in texts]
    new_texts = [t for t in prefixed_texts if t not in overall_embeddings]

    # Encode only new texts
    if new_texts:
        new_embeddings = model.encode(new_texts, convert_to_tensor=True).to(device)
        for text, emb in zip(new_texts, new_embeddings):
            overall_embeddings[text] = emb

    # Reconstruct the tensor in the original order
    embeddings = [overall_embeddings[t] for t in prefixed_texts]
    return torch.stack(embeddings), overall_embeddings


def compare_languages(
    overall_embeddings, model, left_df, right_df, main_col, versions, prefix=""
):
    main_embeddings, overall_embeddings = encode_texts(
        overall_embeddings, model, left_df[main_col].tolist(), prefix
    )
    results = pd.DataFrame()

    for version in versions:
        version_embeddings, overall_embeddings = encode_texts(
            overall_embeddings, model, right_df[version].tolist(), prefix
        )
        similarity_scores = (
            util.cos_sim(main_embeddings, version_embeddings).diag().cpu().numpy()
        )
        results[version] = similarity_scores

    max_scores = results.idxmax(axis=1)
    max_values = results.max(axis=1)
    is_max = results.eq(max_values, axis=0)
    max_counts = is_max.sum(axis=1)
    unique_max = max_counts == 1
    correct_version = versions[0]
    correct = (max_scores == correct_version) & unique_max
    return correct.mean() * 100, correct, overall_embeddings


def process_and_save_results(tasks):
    overall_embeddings = {}
    predictions_to_save = []
    for task in tasks:
        results = []  # Reset results for each task
        task_name = task["name"]
        prefixes = task["prefixes"]
        levels = task["levels"]
        prefix = task.get("prefix", "")
        versions_dict = task["versions_dict"]
        eval_dataset = task["eval_dataset"]

        print("Processing Task Name: " + task_name)

        save_name = (
            "CROSS_E5-multilingual-base"
            + "-".join(levels.keys())
            + eval_dataset
            + f"-{task_name.replace(' ', '-')}-evaluations.csv"
        )

        for model_prefix in tqdm(prefixes, desc=f"Processing Task {task_name}"):
            model_dirs = find_model_directories(model_prefix)
            # model_dirs = [model_prefix]
            if not model_dirs:
                print(f"No models found for prefix '{model_prefix}'")
                continue

            prefix_results = []

            for model_dir in model_dirs:
                if model_dir not in overall_embeddings:
                    overall_embeddings[model_dir] = {}
                try:
                    model = SentenceTransformer(model_dir, trust_remote_code=True)
                    model.to(device)
                except ValueError as e:
                    print(f"Error loading model '{model_dir}': {e}")
                    continue

                comparisons = [
                    {
                        "name": f"{left.capitalize()} to {right.capitalize()}",
                        "left_levels": [left],
                        "right_levels": [right],
                    }
                    for left, right in product(levels.keys(), repeat=2)
                ]

                for comparison in comparisons:
                    for left_level in comparison["left_levels"]:
                        for right_level in comparison["right_levels"]:
                            left_df = pd.read_csv(levels[left_level])
                            right_df = pd.read_csv(levels[right_level])

                            for compare_col in versions_dict:
                                versions = versions_dict[compare_col]
                                if not all(v in right_df.columns for v in versions):
                                    print(
                                        f"Missing columns for {versions} in {levels[right_level]}"
                                    )
                                    continue

                                (
                                    percentage,
                                    all_predictions,
                                    overall_embeddings[model_dir],
                                ) = compare_languages(
                                    overall_embeddings[model_dir],
                                    model,
                                    left_df,
                                    right_df,
                                    compare_col,
                                    versions,
                                    prefix,
                                )

                                # if (
                                #     left_level + "_" + right_level
                                #     not in to_save_predictions
                                # ):
                                #     to_save_predictions[
                                #         left_level + "_" + right_level
                                #     ] = []
                                # to_save_predictions[
                                #     left_level + "_" + right_level
                                # ].extend(all_predictions)

                                left_lang = "DE" if compare_col == "German" else "FR"
                                right_lang = "FR" if compare_col == "German" else "DE"
                                language_direction = f"{left_lang}_{left_level}->{right_lang}_{right_level}"

                                prefix_results.append(
                                    {
                                        "Model": model_dir,
                                        "Comparison": comparison["name"],
                                        "Corruption Level": f"{left_level} to {right_level}",
                                        "Language Direction": language_direction,
                                        "Accuracy": percentage,
                                        "All Predictions": all_predictions,
                                    }
                                )

            # Average results for models under the same prefix
            prefix_results_df = pd.DataFrame(prefix_results)
            if not prefix_results_df.empty:
                avg_results = (
                    prefix_results_df.groupby(
                        ["Comparison", "Corruption Level", "Language Direction"]
                    )["Accuracy"]
                    .mean()
                    .reset_index()
                )
                avg_results["Model Prefix"] = model_prefix
                results.extend(avg_results.to_dict("records"))

                prediction_results = (
                    prefix_results_df.groupby(
                        ["Comparison", "Corruption Level", "Language Direction"]
                    )["All Predictions"]
                    .agg(lambda x: [item for sublist in x for item in sublist])
                    .reset_index()
                )

                prediction_results["Model Prefix"] = model_prefix
                prediction_results["RUN_ID"] = (
                    task_name
                    + " $ "
                    + prediction_results["Language Direction"]
                    + " $ "
                    + prediction_results["Model Prefix"]
                )
                predictions_to_save.extend(prediction_results.to_dict("records"))

        # Save averaged results for this task
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_name, index=False)
        print(f"Results for task '{task_name}' and model prefix saved to {save_name}")

    # Save averaged results for this task
    # print(predictions_to_save[0])
    save_name_predictions = "CROSS_E5-multilingual-base_all_predictions.csv"
    # Create a mapping: RUN_ID -> list of predictions

    run_id_to_preds = {
        row["RUN_ID"]: row["All Predictions"] for row in predictions_to_save
    }
    # print(run_id_to_preds)
    # Find the max length among all prediction lists
    max_len = max(len(preds) for preds in run_id_to_preds.values())

    # Pad each list with NaN to make them all the same length
    padded_preds = {
        run_id: preds + [np.nan] * (max_len - len(preds))
        for run_id, preds in run_id_to_preds.items()
    }

    # Ensure all lists are the same length (as you mentioned)
    # predictions_df = pd.DataFrame(padded_preds)

    # Save to CSV
    # predictions_df.to_csv(save_name_predictions, index=False)
    # print(f"Predictions matrix saved to {save_name_predictions}")


if __name__ == "__main__":
    tasks = [
        {
            "name": "Simple Noise WMT19",
            "prefixes": [
                "intfloat_multilingual-e5-base-cross-10000-samples",
            ],
            "levels": {
                "clean": "./evaluation/wmt2019_adversarial_dataset.csv",
                "simple": "./evaluation/simple_19.csv",
            },
            "versions_dict": {
                "German": ["French", "fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"],
                "French": ["German", "de_adv1", "de_adv2", "de_adv3", "de_adv4"],
            },
            "prefix": "query: ",
            "eval_dataset": "CLSD-wmt19-",
        },
        {
            "name": "Simple Noise WMT21",
            "prefixes": [
                "intfloat_multilingual-e5-base-cross-10000-samples",
            ],
            "levels": {
                "clean": "./evaluation/wmt2021_adversarial_dataset.csv",
                "simple": "./evaluation/simple_21.csv",
            },
            "versions_dict": {
                "German": ["French", "fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"],
                "French": ["German", "de_adv1", "de_adv2", "de_adv3", "de_adv4"],
            },
            "prefix": "query: ",
            "eval_dataset": "CLSD-wmt21-",
        },
        {
            "name": "Blackletter-Scanned Noise WMT19",
            "prefixes": [
                "intfloat_multilingual-e5-base-cross-10000-samples",
            ],
            "levels": {
                "clean": "./evaluation/wmt2019_adversarial_dataset.csv",
                "bl-distorted": "./evaluation/blackletter+distort_19.csv",
            },
            "versions_dict": {
                "German": ["French", "fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"],
                "French": ["German", "de_adv1", "de_adv2", "de_adv3", "de_adv4"],
            },
            "prefix": "query: ",
            "eval_dataset": "CLSD-wmt19-",
        },
        {
            "name": "Blackletter-Scanned Noise WMT21",
            "prefixes": [
                "intfloat_multilingual-e5-base-cross-10000-samples",
            ],
            "levels": {
                "clean": "./evaluation/wmt2021_adversarial_dataset.csv",
                "bl-distorted": "./evaluation/blackletter+distort_21.csv",
            },
            "versions_dict": {
                "German": ["French", "fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"],
                "French": ["German", "de_adv1", "de_adv2", "de_adv3", "de_adv4"],
            },
            "prefix": "query: ",
            "eval_dataset": "CLSD-wmt21-",
        },
        {
            "name": "SaltnPepper Noise WMT19",
            "prefixes": [
                "intfloat_multilingual-e5-base-cross-10000-samples",
            ],
            "levels": {
                "clean": "./evaluation/wmt2019_adversarial_dataset.csv",
                "SaltnPepper": "./evaluation/noise_19.csv",
            },
            "versions_dict": {
                "German": ["French", "fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"],
                "French": ["German", "de_adv1", "de_adv2", "de_adv3", "de_adv4"],
            },
            "prefix": "query: ",
            "eval_dataset": "CLSD-wmt19-",
        },
        {
            "name": "SaltnPepper Noise WMT21",
            "prefixes": [
                "intfloat_multilingual-e5-base-cross-10000-samples",
            ],
            "levels": {
                "clean": "./evaluation/wmt2021_adversarial_dataset.csv",
                "SaltnPepper": "./evaluation/noise_21.csv",
            },
            "versions_dict": {
                "German": ["French", "fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"],
                "French": ["German", "de_adv1", "de_adv2", "de_adv3", "de_adv4"],
            },
            "prefix": "query: ",
            "eval_dataset": "CLSD-wmt21-",
        },
    ]

    process_and_save_results(tasks)
