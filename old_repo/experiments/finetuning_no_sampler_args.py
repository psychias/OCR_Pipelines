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

# Input CSV files containing training data
mono_file = "./inputs/mono_df.csv"
mono_bl_real_file = "./inputs/mono_df_bl_real.csv"
mono_snp_real_file = "./inputs/mono_df_snp_real.csv"
cross_file = "./inputs/cross_df.csv"
fr_en_file = "./inputs/cross_en_fr.csv"
de_en_file = "./inputs/cross_en_de.csv"

# Experiment types
experiment_types = ["mono_snp_real"] # "cross_clean"]
# experiment_types = ["cross+cross_en", "mono", "cross", "mono+cross"]


def prepare_training_samples(
    mono_df,
    mono_df_bl_real,
    mono_df_snp_real,
    cross_df,
    en_fr_df,
    en_de_df,
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
                    texts=[prefix + row["de"], prefix + row["de_005"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["fr"], prefix + row["fr_005"]], label=1
                )
            )
    elif experiment_type == "mono_bl_real":
        mono_df_bl_real = mono_df_bl_real.sample(n=sample_size, random_state=args.random_seed)
        for _, row in mono_df_bl_real.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["german_noise"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["french_noise"]], label=1
                )
            )
    elif experiment_type == "mono_snp_real":
        mono_df_snp_real = mono_df_snp_real.sample(n=sample_size, random_state=args.random_seed)
        for _, row in mono_df_snp_real.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["german"], prefix + row["german_noise"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["french"], prefix + row["french_noise"]], label=1
                )
            )

    elif experiment_type == "x_mono":
        cross_df = cross_df.sample(n=sample_size, random_state=args.random_seed)
        for _, row in cross_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["de"], prefix + row["de_005"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["fr"], prefix + row["fr_005"]], label=1
                )
            )
    elif experiment_type == "mono_batches":
        mono_df = mono_df.sample(n=sample_size, random_state=args.random_seed)
        de_samples = []
        fr_samples = []
        for _, row in mono_df.iterrows():
            de_samples.append(
                InputExample(
                    texts=[prefix + row["de"], prefix + row["de_005"]], label=1
                )
            )
            fr_samples.append(
                InputExample(
                    texts=[prefix + row["fr"], prefix + row["fr_005"]], label=1
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
                    texts=[prefix + row["de"], prefix + row["fr_005"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["fr"], prefix + row["de_005"]], label=1
                )
            )
    
    elif experiment_type == "cross_clean":
        cross_df = cross_df.sample(n=sample_size, random_state=args.random_seed)
        for _, row in cross_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["de"], prefix + row["fr"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["fr"], prefix + row["de"]], label=1
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
                    texts=[prefix + row["de"], prefix + row["de_005"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["fr"], prefix + row["fr_005"]], label=1
                )
            )
        for _, row in cross_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["de"], prefix + row["fr_005"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["fr"], prefix + row["de_005"]], label=1
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
                    texts=[prefix + row["de"], prefix + row["fr_005"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["fr"], prefix + row["de_005"]], label=1
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
    )  # don't shuffle in mono batches
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(train_dataloader)),
        show_progress_bar=True,
    )

    model.save(file_save_name)
    print(f"Model fine-tuning complete and saved as '{file_save_name}'")


mono_df = pd.read_csv(mono_file)
mono_bl_df = pd.read_csv(mono_bl_real_file)
mono_snp_df = pd.read_csv(mono_snp_real_file)
cross_df = pd.read_csv(cross_file)
fr_en_df = pd.read_csv(fr_en_file)
de_en_df = pd.read_csv(de_en_file)
cross_df = pd.read_csv(cross_file)

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
        experiment,
        args.batch_size,
        args.sample_size,
        file_save_name,
        prefix,
    )
