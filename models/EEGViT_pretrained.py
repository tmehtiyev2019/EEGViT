import torch
from transformers import ViTModel, ViTConfig
from torch import nn
from torch.nn import functional as F

class EEGViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layer 1 remains unchanged
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)

        # Additional convolutional layers
        # Convolutional layer 2 - reduced number of channels to prevent too many parameters
        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,  # Keeping the same number of channels
            kernel_size=(1, 7),  # Smaller kernel to capture detailed features
            stride=(1, 1),
            padding=(0,1),
            bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)

        # Skip connection for conv2
        self.skip_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False
        )
        self.skip_bn2 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout(p=0.1)  # Regularization
        
        # Define model_name and config as before
        model_name = "google/vit-base-patch16-224"
        config = ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})  # Matching conv2's output channels
        config.update({'image_size': (129,14)})
        config.update({'patch_size': (8,1)})

        # Initialize the Vision Transformer
        self.model = ViTModel.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        
        # Classifier for regression - unchanged
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        # Pass through the first conv layer
        x1 = self.relu1(self.batchnorm1(self.conv1(x)))
        
        # Pass through the second conv layer with skip connection
        x2 = self.relu2(self.batchnorm2(self.conv2(x1))) + self.skip_bn2(self.skip_conv2(x1))
        x2 = self.dropout(x2)
        
        # Flatten and pass through ViT
        x2 = x2.flatten(2).transpose(1, 2)  # Ensure proper dimensionality for ViT
        x2 = self.model(pixel_values=x2).last_hidden_state
        
        # Apply classifier to the [CLS] token
        x_out = self.classifier(x2[:, 0])
        
        return x_out
