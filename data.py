import torch
# from torchvision import transforms # No longer needed for resize here
import PIL
import glob
from transformers import CLIPProcessor, AutoProcessor # Added AutoProcessor
import random # Import the random module

# Constants from customdataset_test.py (or CLIP defaults)
# For CLIP, the normalization is often part of the processor.
# However, if custom normalization is intended before CLIP processing:
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Standard ImageNet mean
IMAGENET_STD = [0.229, 0.224, 0.225]  # Standard ImageNet std

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, good_dir, defect_dir, model_type, model_name, patch_size: int, apply_horizontal_flip=False, apply_vertical_flip=False): # Added patch_size
        super().__init__()
        # Store image paths and their corresponding labels
        self.image_data = []
        self.patch_size = patch_size # Store patch_size
        
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
        
        self.model_type = model_type.lower()
        # Initialize processor based on model_type
        if self.model_type == "clip":
            self.processor = CLIPProcessor.from_pretrained(model_name)
            print(f"Initialized CLIPProcessor for model: {model_name}")
        elif self.model_type == "vit" or self.model_type == "dinov2":
            self.processor = AutoProcessor.from_pretrained(model_name)
            print(f"Initialized AutoProcessor for model type '{self.model_type}': {model_name}")
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose 'clip', 'vit', or 'dinov2'.")
        
        # Store augmentation flags
        self.apply_horizontal_flip = apply_horizontal_flip
        self.apply_vertical_flip = apply_vertical_flip

        # Configure the image_processor to resize to 224x224 and disable center cropping
        # This assumes the loaded processor has an 'image_processor' attribute that can be configured this way.
        if hasattr(self.processor, 'image_processor') and self.processor.image_processor is not None:
            self.processor.image_processor.do_resize = True # Processor will resize after our custom crop
            self.processor.image_processor.size = {"height": 224, "width": 224} # Consistent 224x224 resize
            self.processor.image_processor.do_center_crop = False # We do a custom top-left crop
            # The resample filter (e.g., BICUBIC) is usually a default in CLIPImageProcessor.
            print(f"Configured image_processor for {self.model_type}: custom crop to multiple of patch_size {self.patch_size}, then resize to 224x224, no center crop.")
        else:
            print(f"Warning: Processor for {self.model_type} does not have a standard 'image_processor' attribute or it's None. Preprocessing settings (resize, crop) might not be applied as expected.")

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

            original_width, original_height = image.size
            




            # # Calculate dimensions for cropping to multiples of patch_size
            # crop_w = (original_width // self.patch_size) * self.patch_size
            # crop_h = (original_height // self.patch_size) * self.patch_size

            # if crop_w <= 0 or crop_h <= 0:
            #     raise ValueError(
            #         f"Image {image_path} (original size: {original_width}x{original_height}) "
            #         f"is too small to be cropped to a non-zero multiple of patch size {self.patch_size}. "
            #         f"Resulting crop dimensions would be {crop_w}x{crop_h}."
            #     )
            
            # image = image.crop((0, 0, crop_w, crop_h))
            # # original_size_tuple = image.size # This is now (crop_w, crop_h)
            




            # Convert original_image_size to actual pixel dimensions (no normalization)
            # Use the actual width and height values directly
            original_image_size_tensor = torch.tensor(
                [original_width, original_height], # Use actual pixel dimensions
                dtype=torch.float
            )
            
            # Process image using CLIPProcessor
            # The processor handles resizing, cropping (usually center crop), normalization, and tensor conversion.
            processed_inputs = self.processor(images=image, return_tensors="pt")
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