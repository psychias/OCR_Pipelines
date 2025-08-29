import argparse
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import pandas as pd
import random
import numpy as np

import os 
# More aggressive memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"



def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

def optimize_memory():
    """Clear GPU cache and run garbage collection more aggressively"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection multiple times
        import gc
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()
    else:
        import gc
        gc.collect()

def estimate_memory_requirements(model_name, batch_size, sample_size):
    """Estimate memory requirements for the model and training"""
    # Rough estimates based on common model sizes
    model_memory_gb = {
        'multilingual-e5-large': 2.5,
        'multilingual-e5-base': 1.5,
        'multilingual-e5-small': 0.5,
        'sentence-transformers/all-MiniLM-L6-v2': 0.5,
        'sentence-transformers/all-mpnet-base-v2': 1.0,
    }
    
    # Get base model memory estimate
    base_memory = 2.0  # Default 2GB for unknown models
    for key, mem in model_memory_gb.items():
        if key in model_name:
            base_memory = mem
            break
    
    # Estimate additional memory for training (gradients, optimizer states, etc.)
    training_overhead = base_memory * 3  # Roughly 3x for gradients + optimizer
    batch_memory = batch_size * 0.1  # Rough estimate per batch item
    
    total_estimated = base_memory + training_overhead + batch_memory
    print(f"Estimated memory requirement: {total_estimated:.1f}GB")
    
    return total_estimated

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
parser.add_argument("--batch_size", type=int, default=1, help="Size of batches")

parser.add_argument(
    "--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping"
)
parser.add_argument(
    "--epochs", type=int, default=1, help="Number of epochs for training"
)
args = parser.parse_args()

# Set random seed
set_random_seed(args.random_seed)

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print initial GPU memory status
if torch.cuda.is_available():
    print_gpu_memory()
    # More conservative memory allocation to prevent fragmentation
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available GPU memory
    # Enable memory mapping for large models
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

mono_file = "./finetuning_data/TED_data_random_noise.csv"
mono_bl_real_file = "./finetuning_data/TED_data_realistic_noise.csv"
mono_snp_real_file = "./finetuning_data/TED_data_realistic_noise.csv"
cross_file = "./finetuning_data/X-News_data_random_noise.csv" 
fr_en_file = "./finetuning_data/X-News_data_random_noise.csv"
# fr_en_file = "./finetuning_data/cross_en_fr.csv"
de_en_file ="./finetuning_data/X-News_data_random_noise.csv"
# de_en_file = "./finetuning_data/cross_en_de.csv"

doc_fr_de_file = "./generate_random_noise/sample_dataset_random_noise_de.csv"  
doc_de_fr_file = "./generate_random_noise/sample_dataset_random_noise_fr.csv"  

# Experiment types - Add new document experiments
experiment_types = ["doc_mix_training"]
# experiment_types = ["mono", "cross","mono+cross","doc_mix_training"]
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
    # Aggressive memory cleanup before starting
    optimize_memory()
    
    print(f"Starting training for {experiment_type}...")
    print_gpu_memory()
    
    # Estimate memory requirements
    estimated_memory = estimate_memory_requirements(model_name, batch_size, sample_size)
    available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 16
    
    if estimated_memory > available_memory * 0.8:
        print(f"WARNING: Estimated memory ({estimated_memory:.1f}GB) may exceed available GPU memory ({available_memory:.1f}GB)")
        # Preemptively reduce batch size
        if batch_size > 1:
            batch_size = 1
            print(f"Preemptively reducing batch size to 1")
    
    # Load SentenceTransformer model with memory optimization
    print(f"Loading model: {model_name}")
    try:
        # Load model with device mapping to prevent memory spikes
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    except Exception as e:
        print(f"Error loading model directly to GPU: {e}")
        print("Loading model to CPU first, then moving to GPU...")
        model = SentenceTransformer(model_name, trust_remote_code=True, device='cpu')
        model = model.to(device)
    
    print("Model loaded.")
    print_gpu_memory()
    
    # Enable gradient checkpointing to save memory
    if hasattr(model[0], 'auto_model'):
        model[0].auto_model.gradient_checkpointing_enable()
    
    # More aggressive memory management for large models
    available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 16
    free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3 if torch.cuda.is_available() else 16
    
    print(f"Available GPU memory: {available_memory_gb:.1f}GB, Free memory: {free_memory_gb:.1f}GB")
    
    # Dynamically adjust batch size based on available memory
    if free_memory_gb < 8:  # Less than 8GB free
        batch_size = 1
        print(f"Very low memory detected. Setting batch size to 1")
    elif free_memory_gb < 12:  # Less than 12GB free
        batch_size = min(batch_size, 2)
        print(f"Low memory detected. Limiting batch size to {batch_size}")
    elif available_memory_gb < 16 and batch_size > 4:
        batch_size = 4
        print(f"Medium memory GPU detected. Limiting batch size to {batch_size}")
    
    print(f"Final batch size: {batch_size}")
    
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
    
    # Clear memory after preparing samples
    optimize_memory()
    print_gpu_memory()

    # Check if we need to reduce batch size due to memory constraints
    available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 16
    if available_memory_gb < 16 and batch_size > 2:
        print(f"Reducing batch size from {batch_size} to 2 due to limited GPU memory ({available_memory_gb:.1f}GB)")
        batch_size = 2

    train_dataloader = DataLoader(
        train_samples, batch_size=batch_size, shuffle=experiment_type != "mono_batches"
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    print(f"Starting training with {len(train_dataloader)} batches...")
    print_gpu_memory()

    # Configure training with memory optimizations
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(train_dataloader)),
        show_progress_bar=True,
        use_amp=True,  # Enable automatic mixed precision
        optimizer_params={'lr': 2e-5},  # Lower learning rate for stability
        checkpoint_path=None,  # Disable checkpointing to save memory
        checkpoint_save_steps=0,
    )

    # Clear cache after training
    optimize_memory()
    print_gpu_memory()

    model.save(file_save_name)
    print(f"Model fine-tuning complete and saved as '{file_save_name}'")
    
    # Delete model from memory to free up space
    del model
    optimize_memory()


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
    print(f"\n{'='*50}")
    print(f"Starting experiment: {experiment}")
    print(f"{'='*50}")
    
    # Clean up memory before each experiment
    optimize_memory()
    print_gpu_memory()
    
    file_save_name = f"{args.model_name.replace('/', '_')}-{experiment}-{args.sample_size}-samples-seed{args.random_seed}"
    
    try:
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
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error in experiment {experiment}: {e}")
        print("Trying with reduced batch size...")
        optimize_memory()
        
        # Try with batch size 1 first
        reduced_batch_size = 1
        try:
            print(f"Retrying with batch size {reduced_batch_size}...")
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
                reduced_batch_size,
                args.sample_size,
                file_save_name + "_batch1",
                prefix,
            )
        except torch.cuda.OutOfMemoryError as e2:
            print(f"Still OOM with batch size 1. Trying with reduced sample size...")
            optimize_memory()
            # Try with half the sample size
            reduced_sample_size = args.sample_size // 2
            try:
                print(f"Retrying with sample size {reduced_sample_size} and batch size 1...")
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
                    1,
                    reduced_sample_size,
                    file_save_name + "_reduced",
                    prefix,
                )
            except Exception as e3:
                print(f"Failed even with reduced sample size and batch size 1: {e3}")
                print("Skipping this experiment due to memory constraints.")
                continue
        except Exception as e2:
            print(f"Failed even with reduced batch size: {e2}")
            continue
    
    # Clean up after each experiment
    optimize_memory()
