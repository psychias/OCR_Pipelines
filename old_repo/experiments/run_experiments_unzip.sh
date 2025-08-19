#!/bin/bash
set -euo pipefail

# If you have a ZIP:
ZIP="tsdae-gte_ge_fr.zip"
DIR="${ZIP%.zip}"


# Unzip only if not already extracted
if [[ -f "$ZIP" && ! -d "$DIR" ]]; then
  echo "Unzipping $ZIP -> $DIR"
  unzip -q "$ZIP" -d "$(dirname "$ZIP")"
fi

# Define random seeds and models
seeds=(42 100 123 777 999)
models=("$DIR")

# Loop over models and seeds
for model in "${models[@]}"
do
    for seed in "${seeds[@]}"
    do
        echo "Running experiments with model: $model and random seed: $seed"
        python finetuning_no_sampler_args.py --model_name "$model" --random_seed "$seed"
    done
done