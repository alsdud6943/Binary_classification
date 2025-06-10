import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CLIPBinaryClassifier

# from data import CLIPDataset
from data_keepratio import CLIPDataset

import argparse # Import argparse
import os # Import os for path operations
from sklearn.metrics import roc_auc_score # Import for AUC
# from tqdm import tqdm # Optional: for progress bars
import datetime
import json # Import the json module
import sys # Import sys for stdout redirection

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate writing
    def flush(self):
        for f in self.files:
            f.flush()

# Configuration values will be set by argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}") # Moved to main for clarity after args parsing

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # progress_bar = tqdm(dataloader, desc="Training", leave=False)
    # for batch in progress_bar:
    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device).unsqueeze(1) # Ensure labels are [batch_size, 1]
        original_image_size = batch["original_image_size"].to(device) # Get original image size

        optimizer.zero_grad()

        # Forward pass
        logits = model(pixel_values, original_image_size) # Pass original_image_size
        loss = criterion(logits, labels) # BCEWithLogitsLoss edxpects raw logits

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * pixel_values.size(0)
        
        # Calculate accuracy
        preds = torch.sigmoid(logits) > 0.5
        correct_predictions += (preds == labels.bool()).sum().item()
        total_samples += labels.size(0)
        
        # if batch_idx % 10 == 0: # Log every 10 batches
        #     print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

# Optional: Validation function
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_preds_probs = [] # Store probabilities for AUC

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            original_image_size = batch["original_image_size"].to(device) # Get original image size
            
            logits = model(pixel_values, original_image_size) # Pass original_image_size
            loss = criterion(logits, labels)
            running_loss += loss.item() * pixel_values.size(0)
            
            probs = torch.sigmoid(logits) # Get probabilities
            preds_classes = probs > 0.5 # Get predicted classes for accuracy
            correct_predictions += (preds_classes == labels.bool()).sum().item()
            total_samples += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy()) # Store true labels
            all_preds_probs.extend(probs.cpu().numpy()) # Store predicted probabilities

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    epoch_auc = -1.0 # Default if AUC cannot be calculated
    if len(all_labels) > 0 and len(set(label[0] for label in all_labels)) > 1: # Check for at least two classes for AUC
        try:
            epoch_auc = roc_auc_score(all_labels, all_preds_probs)
        except ValueError as e:
            print(f"Warning: Could not calculate AUC: {e}. This might happen if only one class is present in the batch.")
    else:
        print("Warning: AUC not calculated. Not enough samples or only one class present in validation set.")

    return epoch_loss, epoch_acc, epoch_auc

def main(args):
    # Generate a unique run ID based on the current timestamp
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(run_output_dir, exist_ok=True)

    # Setup logging to file and console
    log_file_path = os.path.join(run_output_dir, "training_log.txt")
    original_stdout = sys.stdout
    log_file = None  # Initialize to None

    try:
        log_file = open(log_file_path, 'w')
        sys.stdout = Tee(original_stdout, log_file)  # Redirect stdout to Tee object

        print(f"Using device: {DEVICE}")
        print(f"Configuration: {args}")
        print(f"Saving models and logs to: {run_output_dir}")
        # This message will now appear in the console as well.
        print(f"All console output from this script is also being logged to: {log_file_path}")


        # Save the parsed arguments to a JSON file in the run_output_dir
        args_save_path = os.path.join(run_output_dir, "training_args.json")
        with open(args_save_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f"Saved training arguments to: {args_save_path}")

        print("Initializing model...")
        model = CLIPBinaryClassifier(
            model_type=args.model_type,
            model_name=args.model_name,
            hidden_dim=args.hidden_dim,
            use_linear_probing=args.use_linear_probing # Added use_linear_probing
        ).to(DEVICE)

        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}")

        print("Loading training dataset...")
        try:
            train_dataset = CLIPDataset(
                good_dir=args.good_train_data_dir, 
                defect_dir=args.defect_train_data_dir, 
                model_type=args.model_type,
                model_name=args.model_name,
                patch_size=args.patch_size, 
                apply_horizontal_flip=args.random_horizontal_flip,  # Pass the new argument
                apply_vertical_flip=args.random_vertical_flip      # Pass the new argument
            )
            
            # Display training dataset statistics
            train_good_count, train_defect_count = train_dataset.get_label_counts()
            print(f"Training dataset loaded: {train_good_count} good images, {train_defect_count} defect images (Total: {len(train_dataset)})")

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        except ValueError as e:
            print(f"Training dataset creation failed: {e}")
            return
        except Exception as e:
            print(f"Error creating training dataset/dataloader: {e}")
            print(f"Please ensure the data directories ({args.good_train_data_dir}, {args.defect_train_data_dir}) exist with images, or provide valid data paths.")
            return

        val_dataloader = None
        if args.good_val_data_dir and args.defect_val_data_dir:
            print("Loading validation dataset...")
            try:
                # Augmentations are typically not applied to the validation set
                val_dataset = CLIPDataset(
                    good_dir=args.good_val_data_dir,
                    defect_dir=args.defect_val_data_dir,
                    model_type=args.model_type, # Added model_type
                    model_name=args.model_name, # Changed from clip_model_name
                    patch_size=args.patch_size, # Added patch_size
                    apply_horizontal_flip=False, # Explicitly False for validation
                    apply_vertical_flip=False  # Explicitly False for validation
                )
                
                # Display validation dataset statistics
                val_good_count, val_defect_count = val_dataset.get_label_counts()
                print(f"Validation dataset loaded: {val_good_count} good images, {val_defect_count} defect images (Total: {len(val_dataset)})")
                
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            except ValueError as e:
                print(f"Validation dataset creation failed: {e}")
                print("Continuing training without validation...")
            except Exception as e:
                print(f"Error creating validation dataset/dataloader: {e}. Skipping validation.")
        else:
            print("Validation data directories not specified, skipping validation.")

        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
        

        
        # # Calculate pos_weight for weighted BCE loss
        # if train_defect_count > 0:
        #     pos_weight_value = train_good_count / train_defect_count
        #     print(f"Calculated pos_weight for BCEWithLogitsLoss: {pos_weight_value:.2f} (good_count={train_good_count} / defect_count={train_defect_count})")
        # else:
        #     pos_weight_value = 1.0 # Default if no defect samples, though this case should be handled by dataset loading
        #     print("Warning: No defect samples found in training data. Using default pos_weight=1.0 for BCEWithLogitsLoss.")
        
        # pos_weight_tensor = torch.tensor([pos_weight_value], device=DEVICE)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        criterion = nn.BCEWithLogitsLoss()  # Use default BCEWithLogitsLoss without pos_weight


        best_val_auc = -1.0  # Initialize best validation AUC

        print(f"Starting training for {args.num_epochs} epochs...")
        for epoch in range(args.num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, DEVICE)
            print(f"Epoch {epoch+1}/{args.num_epochs} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            if val_dataloader:
                val_loss, val_acc, val_auc = validate_one_epoch(model, val_dataloader, criterion, DEVICE)
                print(f"Epoch {epoch+1}/{args.num_epochs} -> Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    print(f"New best validation AUC: {best_val_auc:.4f}. Saving model...")
                    best_model_save_path = os.path.join(run_output_dir, "clip_classifier_best_val_auc.pth") # Use run_output_dir
                    # os.makedirs(run_output_dir, exist_ok=True) # Already created above
                    torch.save(model.state_dict(), best_model_save_path)
                    print(f"Saved best model checkpoint: {best_model_save_path}")
        
            if args.save_epoch_models:
                model_save_path = os.path.join(run_output_dir, f"clip_classifier_epoch_{epoch+1}.pth") # Use run_output_dir
                # os.makedirs(run_output_dir, exist_ok=True) # Already created above
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved model checkpoint: {model_save_path}")

        print("Training finished.")
        if args.save_final_model:
            final_model_path = os.path.join(run_output_dir, f"clip_binary_classifier_final_epoch_{args.num_epochs}.pth") # Use run_output_dir and add epoch number
            # os.makedirs(run_output_dir, exist_ok=True) # Already created above
            torch.save(model.state_dict(), final_model_path)
            print(f"Final model saved as {final_model_path}")

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout  # Restore original stdout
        if log_file:  # Check if log_file was successfully opened
            log_file.close()
        # This message will now only go to the terminal, confirming the log file location.
        print(f"Console output was logged to: {log_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP Binary Classifier")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model_type', type=str, default="clip", choices=["clip", "vit", "dinov2"], help='Type of model to use: clip, vit, or dinov2')
    parser.add_argument('--model_name', type=str, default="openai/clip-vit-base-patch32", help='Name of the CLIP model to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for the classifier head')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for CLIP model input')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--use_linear_probing', action='store_true', help='Use linear probing instead of MLP head for classification')
    parser.add_argument('--patch_size', type=int, help='Patch size for ViT/DINOv2 models')

    # Made data directories required as dummy data creation is removed
    parser.add_argument('--good_train_data_dir', type=str, nargs='+', required=True, help='Directory or directories for good training images')
    parser.add_argument('--defect_train_data_dir', type=str, nargs='+', required=True, help='Directory or directories for defect training images')
    
    parser.add_argument('--good_val_data_dir', type=str, nargs='+', default=None, help='Directory or directories for good validation images (optional)')
    parser.add_argument('--defect_val_data_dir', type=str, nargs='+', default=None, help='Directory or directories for defect validation images (optional)')

    parser.add_argument('--output_dir', type=str, default="./results", help='Directory to save model checkpoints and final model')
    parser.add_argument('--save_epoch_models', action='store_true', help='Save model checkpoint after each epoch')
    parser.add_argument('--save_final_model', action='store_true', help='Save the final model after training')
    
    parser.add_argument('--random_horizontal_flip', action='store_true', help='Apply random horizontal flip to training images')
    parser.add_argument('--random_vertical_flip', action='store_true', help='Apply random vertical flip to training images')

    args = parser.parse_args()
    
    main(args)
