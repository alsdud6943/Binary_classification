#!/bin/bash

# Directory where test.py is located (assuming this script is in the same directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test.py"

OUTPUT_DIR="${SCRIPT_DIR}/test_results_output"

test_args=(
  --model_path /home/hanta/minyeong/CLIPclassification/results/20250612_082741_dinolarge_256_defectweight2/clip_classifier_best_val_auc.pth
  
  # Pass multiple directories as separate arguments following a single --test_data_dir flag
  # --test_data_dir "/mnt/d/minyeong/boundingbox/splitdataset/val/good" "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/good" "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/joint 뾰족"
  #                 "/mnt/d/minyeong/boundingbox/splitdataset/val/defect" "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/defect" "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/joint 뾰족 defect"

  # --test_data_dir /mnt/d/minyeong/2025-06-10_10-56-23/cropped
  --test_data_dir /mnt/d/minyeong/GLASS_testresults_all/2025-06-25_10-42-40_250616-250622_수작업으로고른것/cropped

  --model_type "dinov2"  
  # --model_name "openai/clip-vit-base-patch32"
  --model_name "facebook/dinov2-large"
  --hidden_dim 256
  --patch_size 14
  --threshold 0.8
  
  # --output_dir "$OUTPUT_DIR"
  --output_dir /mnt/d/minyeong/GLASS_testresults_all/BINARY
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
