import torch
import torch.nn as nn
from transformers import CLIPModel, ViTModel, AutoModel # Add AutoModel
from typing import Optional

class CLIPBinaryClassifier(nn.Module):
    def __init__(self, model_type, model_name="openai/clip-vit-base-patch32", hidden_dim=128, use_linear_probing: bool = False):
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
            # For MLP head: vision embedding only
            classification_head_input_dim = vision_embedding_dim
            self.classifier = nn.Sequential(
                nn.Linear(classification_head_input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1)
            )
            print(f"Using MLP head (vision embedding only) for {self.model_type}.")

    def forward(self, pixel_values, original_image_size=None):
        """
        Args:
            pixel_values: Tensor of preprocessed image data.
                          Expected shape: (batch_size, num_channels, height, width)
            original_image_size: Not used anymore (kept for backward compatibility).
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

        # Use only vision embeddings for classification
        combined_features = cls_embedding
        
        logits = self.classifier(combined_features)
        return logits