import torch
import torch.nn as nn
from transformers import CLIPModel

class CLIPBinaryClassifier(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", hidden_dim=128):
        super(CLIPBinaryClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)

        # Freeze CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Define the classifier head with a hidden layer
        # Add 2 to input dimension for original_image_size (width, height)
        classification_head_input_dim = self.clip_model.config.vision_config.hidden_size + 2 
        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classification_head_input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1) # 1 output for binary classification
        )

    def forward(self, pixel_values, original_image_size):
        """
        Args:
            pixel_values: Tensor of preprocessed image data.
                          Expected shape: (batch_size, num_channels, height, width)
            original_image_size: Tensor of original image sizes (width, height).
                                 Expected shape: (batch_size, 2)
        """
        # Get image features from the vision transformer
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        
        # The CLS token embedding is accessed via pooler_output
        cls_embedding = vision_outputs.pooler_output # Shape: (batch_size, vision_hidden_size)
        
        # Ensure original_image_size is a float tensor and on the correct device
        original_image_size_f = original_image_size.float().to(cls_embedding.device)
        
        # Concatenate CLS embedding with original image size
        combined_features = torch.cat((cls_embedding, original_image_size_f), dim=1)
        
        # Pass combined features through the linear classifier
        logits = self.classifier(combined_features)
        return logits

if __name__ == '__main__':
    # Example usage (for testing the model structure)
    model = CLIPBinaryClassifier()
    print("CLIPBinaryClassifier initialized.")
    print(f"Classifier head: {model.classifier}")

    # Check trainable parameters (should only be the classifier's parameters)
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)