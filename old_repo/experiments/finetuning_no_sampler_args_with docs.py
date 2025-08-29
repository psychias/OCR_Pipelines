import argparse
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import pandas as pd
import random
import numpy as np


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# for multilingual-e5 query -> doc 
def tag(text: str, model_name: str, kind: str): 
    s = "" if text is None else str(text)
    if "multilingual-e5" in model_name:
        return ("query: " if kind == "query" else "passage: ") + s
    return s

def add_pair(samples, left, right, kind_left, kind_right):
    if left is None or right is None:
        return False
    a, b = str(left).strip(), str(right).strip()
    if not a or not b:
        return False
    samples.append(InputExample(texts=[
        tag(a, args.model_name, kind_left),
        tag(b, args.model_name, kind_right)
    ], label=1))
    return True



# Command-line arguments
parser = argparse.ArgumentParser(
    description="Fine-tune SentenceTransformer with different seeds and models."
)
parser.add_argument(
    "--random_seed", type=int, default=42, help="Random seed for reproducibility"
)
parser.add_argument(
    "--model_name", type=str, required=True, help="Model name for SentenceTransformer"
)
parser.add_argument(
    "--sample_size", type=int, default=10000, help="Sample size for training"
)
parser.add_argument("--batch_size", type=int, default=8, help="Size of batches")
parser.add_argument(
    "--epochs", type=int, default=1, help="Number of epochs for training"
)
args = parser.parse_args()

# Set random seed
set_random_seed(args.random_seed)

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mono_file = "./finetuning_data/TED_data_random_noise.csv"
mono_bl_real_file = "./finetuning_data/TED_data_realistic_noise.csv"
mono_snp_real_file = "./finetuning_data/TED_data_realistic_noise.csv"
cross_file = "./finetuning_data/X-News_data_random_noise.csv" 
fr_en_file = "./finetuning_data/cross_en_fr.csv"
de_en_file = "./finetuning_data/cross_en_de.csv"

doc_fr_de_file = "./documents_datasets/sample_dataset_random_noise_de.csv"  
doc_de_fr_file = "./documents_datasets/sample_dataset_random_noise_fr.csv"  

# Experiment types - Add new document experiments
experiment_types = ["mono", "cross","mono+cross","doc_mix_training"]
# experiment_types = ["cross+cross_en", "mono", "cross", "mono+cross","mono_snp_real", "doc_mix_training", "doc_within_similarity", "doc_noisy_training"]  
# experiment_types = ["cross+cross_en", "mono", "cross", "mono+cross"]


def prepare_training_samples(
    mono_df,
    mono_df_bl_real,
    mono_df_snp_real,
    cross_df,
    en_fr_df,
    en_de_df,
    doc_fr_de_df=None,  # New parameter
    doc_de_fr_df=None,  # New parameter
    batch_size=8,
    sample_size=10000,
    experiment_type="mono",
    prefix="",
):
    train_samples = []

    if experiment_type == "mono":
        mono_df = mono_df.sample(n=sample_size, random_state=args.random_seed)
        for _, row in mono_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["german_noise_random05"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["french_noise_random05"]], label=1
                )
            )
    elif experiment_type == "mono_bl_real":
        mono_df_bl_real = mono_df_bl_real.sample(n=sample_size, random_state=args.random_seed)
        for _, row in mono_df_bl_real.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["german_noise_BLDS"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["french_noise_BLDS"]], label=1
                )
            )
    elif experiment_type == "mono_snp_real":
        mono_df_snp_real = mono_df_snp_real.sample(n=sample_size, random_state=args.random_seed)
        for _, row in mono_df_snp_real.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["german_noise_SNP"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["french_noise_SNP"]], label=1
                )
            )

    elif experiment_type == "x_mono":
        cross_df = cross_df.sample(n=sample_size, random_state=args.random_seed)
        for _, row in cross_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["german_noise_random05"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["french_noise_random05"]], label=1
                )
            )
    elif experiment_type == "mono_batches":
        mono_df = mono_df.sample(n=sample_size, random_state=args.random_seed)
        de_samples = []
        fr_samples = []
        for _, row in mono_df.iterrows():
            de_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["german_noise_random05"]], label=1
                )
            )
            fr_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["french_noise_random05"]], label=1
                )
            )
        # Shuffle the samples (before mixing)
        random.shuffle(de_samples)
        random.shuffle(fr_samples)
        for i in range(0, len(de_samples), batch_size):
            train_samples.extend(de_samples[i : i + batch_size])
            train_samples.extend(fr_samples[i : i + batch_size])

    elif experiment_type == "cross":
        cross_df = cross_df.sample(n=sample_size, random_state=args.random_seed)
        for _, row in cross_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["french_noise_random05"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["german_noise_random05"]], label=1
                )
            )
    
    elif experiment_type == "cross_clean":
        cross_df = cross_df.sample(n=sample_size, random_state=args.random_seed)
        for _, row in cross_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["french"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["german"]], label=1
                )
            )

    elif experiment_type == "mono+cross":
        mono_df = mono_df.sample(n=int(sample_size / 2), random_state=args.random_seed)
        cross_df = cross_df.sample(
            n=int(sample_size / 2), random_state=args.random_seed
        )
        for _, row in mono_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["german_noise_random05"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["french_noise_random05"]], label=1
                )
            )
        for _, row in cross_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["french_noise_random05"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["german_noise_random05"]], label=1
                )
            )

    elif experiment_type == "cross+cross_en":
        cross_df = cross_df.sample(
            n=int(sample_size / 2), random_state=args.random_seed
        )
        en_de_df = en_de_df.sample(
            n=int(sample_size / 2), random_state=args.random_seed
        )
        en_fr_df = en_fr_df.sample(
            n=int(sample_size / 2), random_state=args.random_seed
        )
        for _, row in cross_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["french_noise_random05"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["german_noise_random05"]], label=1
                )
            )
        for _, row in en_de_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["en"], prefix + row["de_005"]], label=1
                )
            )
        for _, row in en_fr_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["en"], prefix + row["fr_005"]], label=1
                )
            )
    elif experiment_type == "doc_mix_training":
        # 1) doc_clean ↔ doc_noisy                 (same-language)
        # 2) translation_clean ↔ translation_noisy (other-language)
        # 3) query_clean → translation_noisy       (clean→OCR retrieval)
        # 4) doc_clean ↔ translation_noisy         (cross-lingual doc→doc)  # optional
        USE_CROSS_DOCDOC = True
        USE_SUMMARY = True
        pairs_per_row = 4 + int(USE_CROSS_DOCDOC) + int(USE_SUMMARY)

        target_pairs = sample_size
        count = 0

        def build_from_df(df):
            nonlocal count
            if df is None or len(df) == 0 or count >= target_pairs:
                return
            rows_needed = max(1, (target_pairs - count + pairs_per_row - 1) // pairs_per_row)
            rows_needed = min(rows_needed, len(df))
            for _, row in df.sample(n=rows_needed, random_state=args.random_seed).iterrows():
                if add_pair(train_samples, row.get("doc_text"), row.get("doc_text_noised"), "doc", "doc"):
                    count += 1
                    if count >= target_pairs: break

                if add_pair(train_samples, row.get("translation"), row.get("translation_noised"), "doc", "doc"):
                    count += 1
                    if count >= target_pairs: break

                if add_pair(train_samples, row.get("query"), row.get("translation_noised"), "query", "doc"):
                    count += 1
                    if count >= target_pairs: break
                if add_pair(train_samples, row.get("query"), row.get("doc_text_noised"), "query", "doc"):
                    count += 1
                    if count >= target_pairs: break
                if add_pair(train_samples, row.get("summary"), row.get("summary_noised"), "doc","doc"):
                    count += 1

                if USE_CROSS_DOCDOC and add_pair(train_samples, row.get("doc_text"), row.get("translation_noised"), "doc", "doc"):
                    count += 1
                    if count >= target_pairs: break

                if count >= target_pairs:
                    break

        build_from_df(doc_fr_de_df)
        if count < target_pairs:
            build_from_df(doc_de_fr_df)

    elif experiment_type == "doc_within_similarity":
        # Each row generates exactly 7 pairs:
        # - 4 clean-to-noisy pairs (doc_text, query, summary, translation)
        # - 3 cross-lingual pairs (doc_text→trans_noised, query→trans_noised, summary→trans_noised)
        # Target: ~sample_size total pairs, so sample rows_needed = sample_size // 7
        pairs_per_row = 7
        rows_needed = sample_size // pairs_per_row
        
        if doc_fr_de_df is not None:
            doc_fr_de_sample = doc_fr_de_df.sample(n=min(sample_size//2, len(doc_fr_de_df)), random_state=args.random_seed)
        if doc_de_fr_df is not None:
            doc_de_fr_sample = doc_de_fr_df.sample(n=min(sample_size//2, len(doc_de_fr_df)), random_state=args.random_seed)
        
        if doc_fr_de_df is not None:
            for _, row in doc_fr_de_sample.iterrows():
                train_samples.extend([
                    InputExample(texts=[prefix + str(row["doc_text"]), prefix + str(row["doc_text_noised"])], label=1),
                    InputExample(texts=[prefix + str(row["query"]), prefix + str(row["query_noised"])], label=1),
                    InputExample(texts=[prefix + str(row["summary"]), prefix + str(row["summary_noised"])], label=1),
                    InputExample(texts=[prefix + str(row["translation"]), prefix + str(row["translation_noised"])], label=1),
                    
                    InputExample(texts=[prefix + str(row["doc_text"]), prefix + str(row["translation_noised"])], label=1),
                    InputExample(texts=[prefix + str(row["query"]), prefix + str(row["translation_noised"])], label=1),
                    InputExample(texts=[prefix + str(row["summary"]), prefix + str(row["translation_noised"])], label=1),
                ])
        
        if doc_de_fr_df is not None:
            for _, row in doc_de_fr_sample.iterrows():
                train_samples.extend([
                    InputExample(texts=[prefix + str(row["doc_text"]), prefix + str(row["doc_text_noised"])], label=1),
                    InputExample(texts=[prefix + str(row["query"]), prefix + str(row["query_noised"])], label=1),
                    InputExample(texts=[prefix + str(row["summary"]), prefix + str(row["summary_noised"])], label=1),
                    InputExample(texts=[prefix + str(row["translation"]), prefix + str(row["translation_noised"])], label=1),
                    
                    InputExample(texts=[prefix + str(row["doc_text"]), prefix + str(row["translation_noised"])], label=1),
                    InputExample(texts=[prefix + str(row["query"]), prefix + str(row["translation_noised"])], label=1),
                    InputExample(texts=[prefix + str(row["summary"]), prefix + str(row["translation_noised"])], label=1),
                ])

    elif experiment_type == "doc_noisy_training":
        # Each row generates exactly 7 pairs:
        # - 4 noisy-to-clean pairs (doc_text_noised→doc_text, etc.)
        # - 3 cross-lingual pairs (doc_text_noised→translation, etc.)
        # Target: ~sample_size total pairs, so sample rows_needed = sample_size // 7
        pairs_per_row = 7
        rows_needed = sample_size // pairs_per_row
        
        if doc_fr_de_df is not None:
            doc_fr_de_sample = doc_fr_de_df.sample(n=min(sample_size//2, len(doc_fr_de_df)), random_state=args.random_seed)
        if doc_de_fr_df is not None:
            doc_de_fr_sample = doc_de_fr_df.sample(n=min(sample_size//2, len(doc_de_fr_df)), random_state=args.random_seed)
        
        if doc_fr_de_df is not None:
            for _, row in doc_fr_de_sample.iterrows():
                train_samples.extend([
                    InputExample(texts=[prefix + str(row["doc_text_noised"]), prefix + str(row["doc_text"])], label=1),
                    InputExample(texts=[prefix + str(row["query_noised"]), prefix + str(row["query"])], label=1),
                    InputExample(texts=[prefix + str(row["summary_noised"]), prefix + str(row["summary"])], label=1),
                    InputExample(texts=[prefix + str(row["translation_noised"]), prefix + str(row["translation"])], label=1),
                    
                    InputExample(texts=[prefix + str(row["doc_text_noised"]), prefix + str(row["translation"])], label=1),
                    InputExample(texts=[prefix + str(row["query_noised"]), prefix + str(row["translation"])], label=1),
                    InputExample(texts=[prefix + str(row["summary_noised"]), prefix + str(row["translation"])], label=1),
                ])
        
        if doc_de_fr_df is not None:
            for _, row in doc_de_fr_sample.iterrows():
                train_samples.extend([
                    InputExample(texts=[prefix + str(row["doc_text_noised"]), prefix + str(row["doc_text"])], label=1),
                    InputExample(texts=[prefix + str(row["query_noised"]), prefix + str(row["query"])], label=1),
                    InputExample(texts=[prefix + str(row["summary_noised"]), prefix + str(row["summary"])], label=1),
                    InputExample(texts=[prefix + str(row["translation_noised"]), prefix + str(row["translation"])], label=1),
                    
                    InputExample(texts=[prefix + str(row["doc_text_noised"]), prefix + str(row["translation"])], label=1),
                    InputExample(texts=[prefix + str(row["query_noised"]), prefix + str(row["translation"])], label=1),
                    InputExample(texts=[prefix + str(row["summary_noised"]), prefix + str(row["translation"])], label=1),
                ])

    if experiment_type != "mono_batches":
        random.shuffle(train_samples)
    return train_samples


def fine_tune_model(
    model_name,
    mono_df,
    mono_bl_df,
    mono_snp_df,
    cross_df,
    fr_en_df,
    de_en_df,
    doc_fr_de_df,
    doc_de_fr_df,
    experiment_type,
    batch_size,
    sample_size,
    file_save_name,
    prefix="",
):
    # Load SentenceTransformer model
    print(prefix)
    model = SentenceTransformer(model_name, trust_remote_code=True).to(device)
    print(
        f"Training for {experiment_type} experiment with seed {args.random_seed} using model {args.model_name}..."
    )
    train_samples = prepare_training_samples(
        mono_df,
        mono_bl_df,
        mono_snp_df,
        cross_df,
        fr_en_df,
        de_en_df,
        doc_fr_de_df, 
        doc_de_fr_df,  
        batch_size,
        sample_size,
        experiment_type,
        prefix,
    )
    print(
        f"Sample size {sample_size} for {experiment_type}: {len(train_samples)} samples"
    )

    train_dataloader = DataLoader(
        train_samples, batch_size=batch_size, shuffle=experiment_type != "mono_batches"
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(train_dataloader)),
        show_progress_bar=True,
    )

    model.save(file_save_name)
    print(f"Model fine-tuning complete and saved as '{file_save_name}'")


# Load existing datasets
mono_df = pd.read_csv(mono_file)
mono_bl_df = pd.read_csv(mono_bl_real_file)
mono_snp_df = pd.read_csv(mono_snp_real_file)
cross_df = pd.read_csv(cross_file)
fr_en_df = pd.read_csv(fr_en_file)
de_en_df = pd.read_csv(de_en_file)

doc_fr_de_df = pd.read_csv(doc_fr_de_file)
doc_de_fr_df = pd.read_csv(doc_de_fr_file)

# mono = TED
# Cross = X NEWS
# ta mono bl snp  ktlp -> german,german_noise_BLDS,german_noise_SNP,french,french_noise_BLDS,french_noise_SNP
# eno ta "non real" mila gia random noise
# random noise -> german,french,german_noise_random05,german_noise_random10,german_noise_random15,french_noise_random05,french_noise_random10,french_noise_random15




prefix = "query: " if "multilingual-e5-" in args.model_name else ""
for experiment in experiment_types:
    file_save_name = f"{args.model_name.replace('/', '_')}-{experiment}-{args.sample_size}-samples-seed{args.random_seed}"
    fine_tune_model(
        args.model_name,
        mono_df,
        mono_bl_df,
        mono_snp_df,
        cross_df,
        fr_en_df,
        de_en_df,
        doc_fr_de_df,  
        doc_de_fr_df,  
        experiment,
        args.batch_size,
        args.sample_size,
        file_save_name,
        prefix,
    )
