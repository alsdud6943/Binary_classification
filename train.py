import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CLIPBinaryClassifier
from data import CLIPDataset
import argparse # Import argparse
import os # Import os for path operations
from sklearn.metrics import roc_auc_score # Import for AUC
# from tqdm import tqdm # Optional: for progress bars

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
        loss = criterion(logits, labels) # BCEWithLogitsLoss expects raw logits

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
    print(f"Using device: {DEVICE}")
    print(f"Configuration: {args}")

    print("Initializing model...")
    model = CLIPBinaryClassifier(model_name=args.model_name, hidden_dim=args.hidden_dim).to(DEVICE)

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    print("Loading training dataset...")
    try:
        train_dataset = CLIPDataset(
            good_dir=args.good_train_data_dir, 
            defect_dir=args.defect_train_data_dir, 
            clip_model_name=args.model_name, 
            imagesize=args.image_size
        )
        if len(train_dataset) == 0:
            print(f"Error: No images found in training directories: {args.good_train_data_dir} or {args.defect_train_data_dir}. Please check the paths and ensure images exist.")
            return # Exit if no images found

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    except Exception as e:
        print(f"Error creating training dataset/dataloader: {e}")
        print(f"Please ensure the data directories ({args.good_train_data_dir}, {args.defect_train_data_dir}) exist with images, or provide valid data paths.")
        return

    val_dataloader = None
    if args.good_val_data_dir and args.defect_val_data_dir:
        print("Loading validation dataset...")
        try:
            val_dataset = CLIPDataset(
                good_dir=args.good_val_data_dir,
                defect_dir=args.defect_val_data_dir,
                clip_model_name=args.model_name,
                imagesize=args.image_size
            )
            if len(val_dataset) > 0:
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            else:
                print(f"Warning: No images found in validation directories {args.good_val_data_dir} or {args.defect_val_data_dir}. Skipping validation.")
        except Exception as e:
            print(f"Error creating validation dataset/dataloader: {e}. Skipping validation.")
    else:
        print("Validation data directories not specified, skipping validation.")

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

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
                best_model_save_path = os.path.join(args.output_dir, "clip_classifier_best_val_auc.pth")
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(model.state_dict(), best_model_save_path)
                print(f"Saved best model checkpoint: {best_model_save_path}")
    
        if args.save_epoch_models:
            model_save_path = os.path.join(args.output_dir, f"clip_classifier_epoch_{epoch+1}.pth")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved model checkpoint: {model_save_path}")

    print("Training finished.")
    if args.save_final_model:
        final_model_path = os.path.join(args.output_dir, "clip_binary_classifier_final.pth")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved as {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP Binary Classifier")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model_name', type=str, default="openai/clip-vit-base-patch32", help='Name of the CLIP model to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for the classifier head')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for CLIP model input')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')

    # Made data directories required as dummy data creation is removed
    parser.add_argument('--good_train_data_dir', type=str, required=True, help='Directory for good training images')
    parser.add_argument('--defect_train_data_dir', type=str, required=True, help='Directory for defect training images')
    
    parser.add_argument('--good_val_data_dir', type=str, default=None, help='Directory for good validation images (optional)')
    parser.add_argument('--defect_val_data_dir', type=str, default=None, help='Directory for defect validation images (optional)')

    parser.add_argument('--output_dir', type=str, default="./output_models", help='Directory to save model checkpoints and final model')
    parser.add_argument('--save_epoch_models', action='store_true', help='Save model checkpoint after each epoch')
    parser.add_argument('--save_final_model', action='store_true', help='Save the final model after training')
    
    args = parser.parse_args()
    
    main(args)
