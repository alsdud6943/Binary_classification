import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Added
from PIL import Image # Import PIL for image loading
import shutil # Import shutil for file copying
import argparse
import os
import json
import datetime
import sys

from model import CLIPBinaryClassifier
from data_keepratio import CLIPDataset # Added

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TeeLogger:
    def __init__(self, filename, original_stdout):
        self.file = open(filename, 'w')
        self.stdout = original_stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

def main(args):
    # Create a unique output directory for this test run
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    test_run_output_dir = os.path.join(args.output_dir, f"test_run_{current_time}")
    os.makedirs(test_run_output_dir, exist_ok=True)

    # Define output directories for classified images
    good_images_output_dir = os.path.join(test_run_output_dir, "classified_good")
    defect_images_output_dir = os.path.join(test_run_output_dir, "classified_defect")
    os.makedirs(good_images_output_dir, exist_ok=True)
    os.makedirs(defect_images_output_dir, exist_ok=True)

    # Setup logging to file and console
    log_file_path = os.path.join(test_run_output_dir, "test_log.txt")
    original_stdout = sys.stdout
    # sys.stdout = log_file # Old way
    tee_logger = TeeLogger(log_file_path, original_stdout)
    sys.stdout = tee_logger

    try:
        print(f"Using device: {DEVICE}")
        print(f"Test Configuration: {args}")
        print(f"Saving classified images and logs to: {test_run_output_dir}")
        print(f"Logging console output to: {log_file_path}")

        # Save the parsed arguments to a JSON file
        args_save_path = os.path.join(test_run_output_dir, "test_args.json")
        with open(args_save_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f"Saved test arguments to: {args_save_path}")

        print("Initializing model...")
        model = CLIPBinaryClassifier(
            model_type=args.model_type,
            model_name=args.model_name,
            hidden_dim=args.hidden_dim,
            use_linear_probing=args.use_linear_probing
        ).to(DEVICE)
        
        if not os.path.exists(args.model_path):
            print(f"Error: Model path {args.model_path} does not exist.")
            tee_logger.close() # Close the logger before returning
            sys.stdout = original_stdout # Restore stdout
            return
        
        print(f"Loading model weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")

        print("Initializing test dataset and dataloader...")
        try:
            test_dataset = CLIPDataset(
                good_dir=args.test_data_dir, # Use test_data_dir as good_dir
                defect_dir=[], # No defect directory for test set
                model_type=args.model_type,
                model_name=args.model_name,
                patch_size=args.patch_size,
                apply_horizontal_flip=False,
                apply_vertical_flip=False
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            print(f"Test dataset loaded: {len(test_dataset)} images.")
        except Exception as e:
            print(f"Error creating test dataset/dataloader: {e}")
            tee_logger.close()
            sys.stdout = original_stdout
            return

        total_images_processed_count = len(test_dataset)
        total_good_count = 0
        total_defect_count = 0
        
        print(f"\\nProcessing {total_images_processed_count} images from specified directories...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                pixel_values = batch["pixel_values"].to(DEVICE)
                original_image_size = batch["original_image_size"].to(DEVICE)
                image_paths = batch["image_path"] # List of image paths in the batch

                # Get prediction
                logits = model(pixel_values, original_image_size)
                probs = torch.sigmoid(logits) # Shape: [batch_size, 1] or [batch_size]

                for i in range(len(image_paths)):
                    image_path = image_paths[i]
                    image_name = os.path.basename(image_path)
                    prob_item = probs[i].item() # Get individual probability

                    destination_folder = ""
                    classification_label = ""

                    if prob_item > args.threshold: # Predicted as defect
                        destination_folder = defect_images_output_dir
                        classification_label = "Defect"
                        total_defect_count += 1
                    else: # Predicted as good
                        destination_folder = good_images_output_dir
                        classification_label = "Good"
                        total_good_count += 1
                    
                    try:
                        shutil.copy(image_path, os.path.join(destination_folder, image_name))
                        print(f"Processed '{image_name}' (from path: {image_path}): Classified as {classification_label} (Prob: {prob_item:.4f}) -> Copied to {destination_folder}")
                    except Exception as e:
                        print(f"Error copying image {image_name} to {destination_folder}: {e}")
        
        print("\\nOverall Classification Summary:")
        print(f"  Total images processed: {total_images_processed_count}")
        print(f"  Total images classified as Good: {total_good_count}")
        print(f"  Total images classified as Defect: {total_defect_count}")

        # Save test results summary to a file
        results_summary = {
            "model_path": args.model_path,
            "test_data_dirs": args.test_data_dir, # Changed to list of dirs
            "total_images_processed": total_images_processed_count,
            "classified_good_count": total_good_count,
            "classified_defect_count": total_defect_count,
            "output_good_dir": good_images_output_dir,
            "output_defect_dir": defect_images_output_dir
        }
        results_save_path = os.path.join(test_run_output_dir, "test_results_summary.json")
        with open(results_save_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"Test results summary saved to: {results_save_path}")

    finally:
        sys.stdout = original_stdout # Restore original stdout
        if hasattr(tee_logger, 'close') and callable(getattr(tee_logger, 'close')):
            tee_logger.close() # Close the file part of TeeLogger
        # This print statement will go to the actual console, as stdout is restored.
        # And it was also logged by TeeLogger before stdout was restored if it was inside the try block.
        # To ensure this specific message ONLY goes to console after logging is done:
        print(f"Console output was also logged to: {log_file_path}", file=original_stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CLIP Binary Classifier on unlabeled images")
    
    # Model and Data Arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--test_data_dir', type=str, nargs='+', required=True, help='One or more directories for test images to classify')
    
    # Model Configuration (should match the trained model)
    parser.add_argument('--model_type', type=str, default="clip", choices=["clip", "vit", "dinov2"], help='Type of model used for training: clip, vit, or dinov2')
    parser.add_argument('--model_name', type=str, default="openai/clip-vit-base-patch32", help='Name of the CLIP model (must match trained model)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for the classifier head (must match trained model)')
    parser.add_argument('--use_linear_probing', action='store_true', help='Whether linear probing was used (must match trained model)')
    parser.add_argument('--patch_size', type=int, default=32, help='Patch size used during training (crucial for preprocessing)')

    parser.add_argument('--image_size', type=int, default=224, help='Image size reference for the processor (e.g., 224, must match training processor config)')

    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classifying an image as defect')

    # Batching and workers
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')

    # Output directory
    parser.add_argument('--output_dir', type=str, default="./test_results", help='Directory to save test logs, classified images, and results')

    args = parser.parse_args()
    main(args)
