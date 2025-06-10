import torch
import torch.nn as nn
from transformers import CLIPModel, ViTModel, AutoModel # Add AutoModel
from typing import Optional

class CLIPBinaryClassifier(nn.Module):
    def __init__(self, model_type, model_name="openai/clip-vit-base-patch32", hidden_dim=128, use_linear_probing: bool = False): # Removed clip_transformer_block_index
        super(CLIPBinaryClassifier, self).__init__()
        self.model_type = model_type.lower()
        self.use_linear_probing = use_linear_probing # Store use_linear_probing

        if self.model_type == "clip":
            self.vision_backbone = CLIPModel.from_pretrained(model_name)
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
            vision_embedding_dim = self.vision_backbone.config.vision_config.hidden_size
        elif self.model_type == "vit":
            self.vision_backbone = ViTModel.from_pretrained(model_name)
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
            vision_embedding_dim = self.vision_backbone.config.hidden_size
        elif self.model_type == "dinov2": # Add DINOv2 option
            self.vision_backbone = AutoModel.from_pretrained(model_name)
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
            vision_embedding_dim = self.vision_backbone.config.hidden_size
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'clip', 'vit', or 'dinov2'.")

        self.hidden_dim = hidden_dim

        if self.use_linear_probing:
            classification_head_input_dim = vision_embedding_dim
            self.classifier = nn.Linear(classification_head_input_dim, 1)
            print(f"Using Linear Probing head (vision embedding only) for {self.model_type}.")
        else:
            classification_head_input_dim = vision_embedding_dim + 2 # Add 2 for original_image_size for MLP
            self.classifier = nn.Sequential(
                nn.Linear(classification_head_input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1)
            )
            print(f"Using MLP head (vision embedding + image size) for {self.model_type}.")

    def forward(self, pixel_values, original_image_size):
        """
        Args:
            pixel_values: Tensor of preprocessed image data.
                          Expected shape: (batch_size, num_channels, height, width)
            original_image_size: Tensor of original image sizes (width, height), potentially normalized.
                                 Expected shape: (batch_size, 2). 
        """
        if self.model_type == "clip":
            # Get image features from the vision transformer part of CLIP
            vision_outputs = self.vision_backbone.vision_model(
                pixel_values=pixel_values
                # Removed output_hidden_states argument
            )
            cls_embedding = vision_outputs.pooler_output # CLS token embedding
        elif self.model_type == "vit":
            # Get image features from ViTModel
            vision_outputs = self.vision_backbone(pixel_values=pixel_values)
            cls_embedding = vision_outputs.pooler_output # ViTModel also has pooler_output for CLS
        elif self.model_type == "dinov2": # Add DINOv2 option
            vision_outputs = self.vision_backbone(pixel_values=pixel_values)
            cls_embedding = vision_outputs.pooler_output # DINOv2 typically uses CLS token output from pooler
        else:
            raise ValueError(f"Unsupported model_type during forward pass: {self.model_type}")

        if self.use_linear_probing:
            combined_features = cls_embedding
        else:
            if original_image_size is None:
                raise ValueError("original_image_size must be provided for MLP head.")
            original_image_size_f = original_image_size.float().to(cls_embedding.device)
            combined_features = torch.cat((cls_embedding, original_image_size_f), dim=1)
        
        logits = self.classifier(combined_features)
        return logits