import argparse
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import pandas as pd
import random
import numpy as np
import json 
import re 




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
        train_samples_tuple = train_samples_tuple[sample_size]

    train_samples = [InputExample(texts=[sample[0], sample[1]], label=1) for sample in train_samples_tuple]
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
summary_doc = "<summary>" # columns doc_text, query, title, summary, doc_text_noised, query_noised, summary_noised
docs_file = "incoming from simon "
lux_file_en = "./finetuning_data/lb_en_training_set.jsonl"
lux_file_de = "./finetuning_data/lb_de_training_set.jsonl"
lux_file_fr = "./finetuning_data/lb_fr_training_set.jsonl"


# 1 summary -> doc
# 1 doc-> doc (Simon)
# 1 doc -> doc ( other language) 
# 1 summary -> summary noise 
# 1 q -> doc 
# 1 q -> summary 
# 1 TED q ->q
# 1 lux q -> q, 1h -> 





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
    elif experiment_type == "mix_training":
        mono_df = mono_df.sample(n=sample_size, random_state=args.random_seed)
        doc_df = doc_df.sample(n=sample_size, random_state=args.random_seed)
        summary_df = summary_df.sample(n=sample_size, random_state=args.random_seed)
        lx_df = lx_df.sample(n=sample_size, random_state=args.random_seed)
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
        for _, row in doc_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["doc_text"], prefix + row["doc_text_noised"]], label=1
                )
            )
        for _, row in summary_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["summary"], prefix + row["summary_noised"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["summary"], prefix + row["summary_noised"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["query"], prefix + row["summary_noised"]], label=1
                )
            )
            train_samples.append(
                InputExample(
                    texts=[prefix + row["query"], prefix + row["doc_text_noised"]], label=1
                )
            )
        for _, row in lx_df.iterrows():
            train_samples.append(
                InputExample(
                    texts=[prefix + row["lx"], prefix + row["lx_noised"]], label=1
                )
            )

#             1 summary -> doc
# 1 doc-> doc (Simon)
# 1 doc -> doc ( other language) 
# 1 summary -> summary noise 
# 1 q -> doc 
# 1 q -> summary 
# 1 TED q ->q
# 1 lux q -> q, 1h -> 



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
    lx_df,
    summary_df,
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
        lx_df,
        summary_df,
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
lx_df_en = pd.read_json(lux_file_en, lines=True)
lx_df_de = pd.read_json(lux_file_de, lines=True)
lx_df_fr = pd.read_json(lux_file_fr, lines=True)
lx_df = pd.concat([lx_df_en, lx_df_de, lx_df_fr], ignore_index=True)
summary_df = pd.read_csv(summary_doc)

lb_de_training_set_loaded = load_dataset_local(lux_file_de)
lb_de_training_sentences = extract_parallel_sentences(lb_de_training_set_loaded, src_col="lb", tgt_col="de")
lb_de_sampled_training_set = prepare_training_samples(lb_de_training_sentences, sample_size=20160, random_sampling=True)

lb_fr_training_set_loaded = load_dataset_local(lux_file_fr)
lb_fr_training_sentences = extract_parallel_sentences(lb_fr_training_set_loaded,src_col='lb',tgt_col='fr')
lb_fr_sampled_training_set = prepare_training_samples(lb_fr_training_sentences, sample_size=20160, random_sampling=True)

lb_en_training_set_loaded = load_dataset_local(lux_file_en)
lb_en_training_sentences = extract_parallel_sentences(lb_en_training_set_loaded,src_col='lb',tgt_col='en')
lb_en_sampled_training_set = prepare_training_samples(lb_en_training_sentences, sample_size=20160, random_sampling=True)

print(f"Loaded training set size (in articles) de: {len(lb_de_training_set_loaded)}| fr:{len(lb_fr_training_set_loaded)}| en:{len(lb_en_training_set_loaded)} ")



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
        lx_df,
        summary_df,
        lb_en_sampled_training_set,
        lb_fr_sampled_training_set,
        lb_de_sampled_training_set
        experiment,
        args.batch_size,
        args.sample_size,
        file_save_name,
        prefix,
    )



