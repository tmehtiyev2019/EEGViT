import torch
import transformers
from transformers import ViTModel
from torch import nn

class EEGViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        # First Convolution Layer
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0, 2),
            bias=False
        )
        
        # Depthwise Separable Convolution Layer
        self.depthwise_conv = nn.Conv2d(
            in_channels=256, 
            out_channels=256, 
            kernel_size=(3, 3),
            padding=1, 
            groups=256
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=256, 
            out_channels=512, 
            kernel_size=1
        )
        
        # Batch Normalization
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        self.batchnorm2 = nn.BatchNorm2d(512, False)

        # Vision Transformer Configuration
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 512})
        config.update({'image_size': (129,14)})
        config.update({'patch_size': (8,1)})
        
        model = transformers.ViTForImageClassification.from_pretrained(
            model_name, 
            config=config, 
            ignore_mismatched_sizes=True
        )
        model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            512, 
            768, 
            kernel_size=(8, 1), 
            stride=(8, 1), 
            padding=(0,0), 
            groups=256
        )
        model.classifier = nn.Sequential(
            nn.Linear(768, 1000, bias=True),
            nn.Dropout(p=0.1),
            nn.Linear(1000, 2, bias=True)
        )
        self.ViT = model
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.batchnorm2(x)
        x = self.ViT.forward(x).logits
        return x
