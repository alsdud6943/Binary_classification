#!/bin/bash

# Directory where train.py is located (assuming this script is in the same directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"

train_args=(
  # **Required: Data paths**
  --good_train_data_dir "/mnt/c/Users/HANTA/Desktop/minyeong/cropped/splitdataset/train/good"
  --defect_train_data_dir "/mnt/c/Users/HANTA/Desktop/minyeong/cropped/splitdataset/train/defect"
  
  # Optional: Validation data (comment out the relevant --*_val_data_dir lines to disable)
  --good_val_data_dir "/mnt/c/Users/HANTA/Desktop/minyeong/cropped/splitdataset/val/good"
  --defect_val_data_dir "/mnt/c/Users/HANTA/Desktop/minyeong/cropped/splitdataset/val/defect"

  # Training parameters
  --learning_rate 0.0001
  --batch_size 32
  --num_epochs 100
  --model_name "openai/clip-vit-base-patch32"
  --hidden_dim 512
  --image_size 224
  --num_workers 4
  
  # Output directory for models
  --output_dir "${SCRIPT_DIR}/output_models" # Saves models in 'output_models' relative to this script

  # Flags (comment out to disable the flag)
  # --save_epoch_models  # Example: uncomment to save model checkpoint after each epoch
  --save_final_model   # Saves the final model
)

# --- Execute the command ---
"python" "$TRAIN_SCRIPT" "${train_args[@]}"

echo "Training script execution finished."

