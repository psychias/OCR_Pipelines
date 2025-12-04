#!/bin/bash

# Define random seeds and models
seeds=(42 100 123 777 999)
models=("Alibaba-NLP/gte-multilingual-base")

# Loop over models and seeds
for model in "${models[@]}"
do
    for seed in "${seeds[@]}"
    do
        echo "Running experiments with model: $model and random seed: $seed"
        python src/finetuning/finetuning_no_sampler_args.py --model_name "$model" --random_seed "$seed"
    done
done