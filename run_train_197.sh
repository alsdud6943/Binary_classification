#!/bin/bash

# Directory where train.py is located (assuming this script is in the same directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"

train_args=(
  # training data 
  --good_train_data_dir "/mnt/d/minyeong/boundingbox/splitdataset/train/good" "/mnt/d/minyeong/boundingbox/cropped_5월_split/train/good" "/mnt/d/minyeong/boundingbox/cropped_5월_split/train/joint 뾰족" "/mnt/d/minyeong/boundingbox/splitdataset/train/뾰족"
  --defect_train_data_dir "/mnt/d/minyeong/boundingbox/splitdataset/train/defect" "/mnt/d/minyeong/boundingbox/cropped_5월_split/train/defect" "/mnt/d/minyeong/boundingbox/cropped_5월_split/train/joint 뾰족 defect" 
  
  # validation data
  --good_val_data_dir "/mnt/d/minyeong/boundingbox/splitdataset/val/good" "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/good" "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/joint 뾰족" "/mnt/d/minyeong/boundingbox/splitdataset/val/뾰족"
  --defect_val_data_dir "/mnt/d/minyeong/boundingbox/splitdataset/val/defect" "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/defect" "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/joint 뾰족 defect" 

  --random_horizontal_flip
  # --random_vertical_flip

  # Training parameters
  --learning_rate 0.0001
  --batch_size 32
  --num_epochs 200
  --model_type "dinov2"
  # --model_type "vit"


  # --model_name "facebook/dinov2-base"
  --model_name "facebook/dinov2-large"
  # --model_name "google/vit-base-patch16-224-in21k"
  --hidden_dim 256
  --image_size 224
  --num_workers 8
  # --use_linear_probing
  --patch_size 14

  --checkpoint_path "/home/hanta/minyeong/CLIPclassification/results/20250611_135517_dinolarge_256/clip_binary_classifier_final_epoch_100.pth"

  # Output directory for models
  --output_dir "${SCRIPT_DIR}/results" # Saves models in 'results' relative to this script

  # Flags
  # --save_epoch_models  # save model checkpoint after each epoch
  --save_final_model   # Saves the final model
)

# --- Execute the command ---
"python" "$TRAIN_SCRIPT" "${train_args[@]}"

echo "Training script execution finished."

