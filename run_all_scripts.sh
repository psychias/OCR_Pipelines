#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# List of Python scripts to run in order
scripts=(
    # "src/evaluations_scripts/evaluation.py"
    "src/evaluations_scripts/CLSD_wmt_evaluation_gte_mono_SnP.py"
    "src/evaluations_scripts/CLSD_wmt_evaluation_gte_BL.py",
    "src/evaluations_scripts/PARALUX_evaluation.py"
)

# Loop through each script and execute it
for script in "${scripts[@]}"; do
    echo "Running $script..."
    python "$script"
    echo "$script completed."
done

echo "All scripts completed successfully."
