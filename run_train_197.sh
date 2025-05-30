#!/bin/bash

# Directory where train.py is located (assuming this script is in the same directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"

train_args=(
  # training data
  --good_train_data_dir "/mnt/d/minyeong/boundingbox/splitdataset/train/good"
  --defect_train_data_dir "/mnt/d/minyeong/boundingbox/splitdataset/train/defect"
  
  # validation data
  --good_val_data_dir "/mnt/d/minyeong/boundingbox/splitdataset/val/good"
  --defect_val_data_dir "/mnt/d/minyeong/boundingbox/splitdataset/val/defect"

  --random_horizontal_flip
  # --random_vertical_flip

  # Training parameters
  --learning_rate 0.0001
  --batch_size 32
  --num_epochs 100
  --model_name "openai/clip-vit-base-patch32"
  --hidden_dim 8
  --image_size 224
  --num_workers 8
  
  # Output directory for models
  --output_dir "${SCRIPT_DIR}/results" # Saves models in 'results' relative to this script

  # Flags
  # --save_epoch_models  # save model checkpoint after each epoch
  --save_final_model   # Saves the final model
)

# --- Execute the command ---
"python" "$TRAIN_SCRIPT" "${train_args[@]}"

echo "Training script execution finished."

