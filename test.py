import torch
import torch.nn as nn
from transformers import CLIPProcessor # Import CLIPProcessor
from PIL import Image # Import PIL for image loading
import shutil # Import shutil for file copying
import argparse
import os
import json
import datetime
import sys

from model import CLIPBinaryClassifier

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
            model_name=args.model_name,
            hidden_dim=args.hidden_dim,
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

        print(f"Loading CLIP processor for model: {args.model_name}...")
        processor = CLIPProcessor.from_pretrained(args.model_name)
        print("CLIP processor loaded successfully.")

        total_good_count = 0
        total_defect_count = 0
        total_images_processed_count = 0

        for data_dir in args.test_data_dir:
            print(f"\nProcessing images from directory: {data_dir}")
            if not os.path.isdir(data_dir):
                print(f"Warning: Directory {data_dir} does not exist. Skipping.")
                continue

            image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            if not image_files:
                print(f"No image files found in {data_dir}.")
                continue
            
            total_images_processed_count += len(image_files)
            current_dir_good_count = 0
            current_dir_defect_count = 0

            with torch.no_grad():
                for image_name in image_files:
                    image_path = os.path.join(data_dir, image_name)
                    try:
                        image = Image.open(image_path).convert("RGB")
                        original_size = image.size

                        # Preprocess image
                        inputs = processor(text=None, images=image, return_tensors="pt", padding=True)
                        pixel_values = inputs["pixel_values"].to(DEVICE)
                        
                        # Create original_image_size tensor for the model
                        # Apply the same normalization as in data.py
                        # Scaling width by 512 and height by 699.
                        original_size_normalized_tensor = torch.tensor(
                            [original_size[0] / 512.0, original_size[1] / 699.0],
                            dtype=torch.float
                        ).unsqueeze(0).to(DEVICE)

                        # Get prediction
                        logits = model(pixel_values, original_size_normalized_tensor) # Use normalized tensor
                        print(f"Debug: Logits for {image_name} from {data_dir}: {logits.item()}") # Print raw logits
                        prob = torch.sigmoid(logits).item() # Get probability for the positive class (defect)

                        destination_folder = ""
                        classification_label = ""

                        if prob > args.threshold: # Predicted as defect using the provided threshold
                            destination_folder = defect_images_output_dir
                            classification_label = "Defect"
                            current_dir_defect_count += 1
                        else: # Predicted as good
                            destination_folder = good_images_output_dir
                            classification_label = "Good"
                            current_dir_good_count += 1
                        
                        shutil.copy(image_path, os.path.join(destination_folder, image_name))
                        print(f"Processed '{image_name}' from '{data_dir}': Classified as {classification_label} (Prob: {prob:.4f}) -> Copied to {destination_folder}")

                    except Exception as e:
                        print(f"Error processing image {image_name} from {data_dir}: {e}")
            
            print(f"Summary for directory {data_dir}:")
            print(f"  Images classified as Good: {current_dir_good_count}")
            print(f"  Images classified as Defect: {current_dir_defect_count}")
            total_good_count += current_dir_good_count
            total_defect_count += current_dir_defect_count
        
        print("\nOverall Classification Summary:")
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
    parser.add_argument('--model_name', type=str, default="openai/clip-vit-base-patch32", help='Name of the CLIP model (must match trained model)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for the classifier head (must match trained model)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classifying an image as defect')

    # Output directory
    parser.add_argument('--output_dir', type=str, default="./test_results", help='Directory to save test logs, classified images, and results')

    args = parser.parse_args()
    main(args)
