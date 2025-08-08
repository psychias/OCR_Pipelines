#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# List of Python scripts to run in order
scripts=(
    "CLSD_wmt_evaluation_gte_mono_batches.py"
    "CLSD_wmt_evaluation_gte_mono_SnP.py"
    "CLSD_wmt_evaluation_gte_BL.py"
)

# Loop through each script and execute it
for script in "${scripts[@]}"; do
    echo "Running $script..."
    python "$script"
    echo "$script completed."
done

echo "All scripts completed successfully."
