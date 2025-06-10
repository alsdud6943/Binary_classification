#Requires -Version 5.0
# filepath: d:\0_Image_AI\_GT_Defect\_Pred_Realtime\CLIPclassification\test.ps1

# Directory where test.py is located
$ScriptDir = $PSScriptRoot
$TestScript = Join-Path $ScriptDir "test.py"

$OutputDir = Join-Path $ScriptDir "test_results_output"

$testArgs = @(
  "--model_path", "D:\0_Image_AI\_GT_Defect\_Pred_Realtime\CLIPclassification\results\20250610_085608_dino256_keepaspectratio\clip_classifier_best_val_auc.pth",
  
  # Pass multiple directories as separate arguments following a single --test_data_dir flag
  "--test_data_dir", "D:\0_Image_AI\_GT_Defect\MIN\testresults\2025-06-10_10-56-23\cropped"
#   "/mnt/d/minyeong/boundingbox/splitdataset/val/good", "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/good", "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/joint 뾰족",
                #   "/mnt/d/minyeong/boundingbox/splitdataset/val/defect", "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/defect", "/mnt/d/minyeong/boundingbox/cropped_5월_split/val/joint 뾰족 defect",
                  
  "--model_type", "dinov2",  
  # "--model_name", "openai/clip-vit-base-patch32",
  "--model_name", "facebook/dinov2-base",
  "--hidden_dim", "256",
  "--patch_size", "14",
  "--threshold", "0.8",
  
  "--output_dir", "D:\0_Image_AI\_GT_Defect\MIN\testresults\BINARY"
)

# --- Create output directory if it doesn't exist ---
if (-not (Test-Path $OutputDir)) {
  New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
}

# --- Execute the command ---
Write-Host "Starting test script..."
Write-Host "Output Dir: $OutputDir"
# The test.py script will log the specific model_path, test_data_dir, and threshold used.

python $TestScript $testArgs

Write-Host "Test script execution finished."
Write-Host "Results, logs, and classified images saved in a subfolder within: $OutputDir"
