#!/bin/bash

# Directory where test.py is located (assuming this script is in the same directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test.py"

OUTPUT_DIR="${SCRIPT_DIR}/test_results_output"

test_args=(
  --model_path /home/hanta/minyeong/CLIPclassification/results/20250604_091744_128/clip_classifier_best_val_auc.pth # Example, replace with actual path
  
  # Pass multiple directories as separate arguments following a single --test_data_dir flag
  --test_data_dir "/mnt/d/minyeong/boundingbox/splitdataset/val/good" "mnt/d/minyeong/boundingbox/cropped_5월_split/val/good" "mnt/d/minyeong/boundingbox/cropped_5월_split/val/joint 뾰족"
                  "/mnt/d/minyeong/boundingbox/splitdataset/val/defect" "mnt/d/minyeong/boundingbox/cropped_5월_split/val/defect" "mnt/d/minyeong/boundingbox/cropped_5월_split/val/joint 뾰족 defect"
                  
  
  --model_name "openai/clip-vit-base-patch32"
  --hidden_dim 128

  --threshold 0.4
  
  --output_dir "$OUTPUT_DIR"
)

# --- Create output directory if it doesn't exist ---
mkdir -p "$OUTPUT_DIR"

# --- Execute the command ---
echo "Starting test script..."
echo "Output Dir: $OUTPUT_DIR"
# The test.py script will log the specific model_path, test_data_dir, and threshold used.

"python" "$TEST_SCRIPT" "${test_args[@]}"

echo "Test script execution finished."
echo "Results, logs, and classified images saved in a subfolder within: $OUTPUT_DIR"
