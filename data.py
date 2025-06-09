import torch
# from torchvision import transforms # No longer needed for resize here
import PIL
import glob
from transformers import CLIPProcessor # Added
import random # Import the random module

# Constants from customdataset_test.py (or CLIP defaults)
# For CLIP, the normalization is often part of the processor.
# However, if custom normalization is intended before CLIP processing:
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Standard ImageNet mean
IMAGENET_STD = [0.229, 0.224, 0.225]  # Standard ImageNet std

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, good_dir, defect_dir, clip_model_name="openai/clip-vit-base-patch32", apply_horizontal_flip=False, apply_vertical_flip=False):
        super().__init__()
        # Store image paths and their corresponding labels
        self.image_data = []
        
        good_image_paths = []
        if isinstance(good_dir, str):
            good_dir = [good_dir]
        
        # Check each good directory individually
        error_messages = []
        for directory in good_dir:
            dir_images = sorted(glob.glob(directory + "/*.jpg"))
            if not dir_images:
                error_messages.append(f"No images found in good directory: {directory}")
            else:
                good_image_paths.extend(dir_images)
        
        defect_image_paths = []
        if isinstance(defect_dir, str):
            defect_dir = [defect_dir]
        
        # Check each defect directory individually
        for directory in defect_dir:
            dir_images = sorted(glob.glob(directory + "/*.jpg"))
            if not dir_images:
                error_messages.append(f"No images found in defect directory: {directory}")
            else:
                defect_image_paths.extend(dir_images)
        
        if error_messages:
            for msg in error_messages:
                print(f"Error: {msg}")
            raise ValueError("Dataset creation failed: Missing images in required directories")
        
        # Add images to dataset
        for path in good_image_paths:
            self.image_data.append({"path": path, "label": 0}) # 0 for good
        for path in defect_image_paths:
            self.image_data.append({"path": path, "label": 1}) # 1 for defect
        
        # Initialize CLIP processor
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Store augmentation flags
        self.apply_horizontal_flip = apply_horizontal_flip
        self.apply_vertical_flip = apply_vertical_flip

        # Configure the image_processor to resize to 224x224 and disable center cropping
        self.processor.image_processor.do_resize = True
        self.processor.image_processor.size = {"height": 224, "width": 224}
        self.processor.image_processor.do_center_crop = False
        # The resample filter (e.g., BICUBIC) is usually a default in CLIPImageProcessor.

    def __getitem__(self, idx):
        item_info = self.image_data[idx]
        
        image_path = item_info["path"]
        label = item_info["label"]
        
        try:
            image = PIL.Image.open(image_path).convert("RGB")

            # Apply random flips if enabled
            if self.apply_horizontal_flip and random.random() < 0.5:
                image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            if self.apply_vertical_flip and random.random() < 0.5:
                image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)

            original_size_tuple = image.size # Store original image size as tuple (width, height)
            
            # Convert original_image_size to a normalized FloatTensor
            # Scaling width by 512 and height by 699.
            original_image_size_tensor = torch.tensor(
                [original_size_tuple[0] / 512.0, original_size_tuple[1] / 699.0],
                dtype=torch.float
            )
            
            # Process image using CLIPProcessor
            # The processor handles resizing, cropping (usually center crop), normalization, and tensor conversion.
            processed_inputs = self.processor(images=image, return_tensors="pt", padding=True, truncation=True)
            pixel_values = processed_inputs['pixel_values'].squeeze(0) # Remove batch dim

        except PIL.UnidentifiedImageError:
            print(f"Warning: Could not open image {image_path}. Skipping.")
            # Consider how to handle this error more gracefully in a batch,
            # e.g., return None and filter in collate_fn, or raise an error.
            # For now, re-raising to stop execution if an image is bad.
            raise
        except Exception as e:
            print(f"Error processing image {image_path} with CLIPProcessor: {e}")
            raise

        return {
            "pixel_values": pixel_values, 
            "image_path": image_path,
            "label": torch.tensor(label, dtype=torch.float), 
            "original_image_size": original_image_size_tensor # Return normalized tensor
        }

    def get_label_counts(self):
        """Return the count of good (label=0) and defect (label=1) images."""
        good_count = sum(1 for item in self.image_data if item["label"] == 0)
        defect_count = sum(1 for item in self.image_data if item["label"] == 1)
        return good_count, defect_count

    def __len__(self):
        return len(self.image_data)