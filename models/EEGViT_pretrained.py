import torch
from transformers import ViTModel, ViTConfig
from torch import nn
from torch.nn import functional as F

class EEGViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0,2),
            bias=False
        )
        
        # The rest of the convolutional layers are removed to match the expected input of the ViT model
        
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(p=0.1)

        # Initialize the ViT model with the original configuration
        model_name = "google/vit-base-patch16-224"
        self.model = ViTModel.from_pretrained(model_name)

        # The classifier is updated to output a continuous value for the eye position
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1)  # For regression task
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))
        x = self.dropout(x)
        
        # Flatten the output for the ViT model
        x = x.flatten(2)
        x = x.transpose(1, 2)  # Transpose for correct shape
        
        # Adjust the size of the sequence for the ViT model
        n, sequence_length, _ = x.size()
        if sequence_length > self.model.config.num_hidden_layers:
            x = x[:, :self.model.config.num_hidden_layers, :]
        elif sequence_length < self.model.config.num_hidden_layers:
            padding = x.new_zeros((n, self.model.config.num_hidden_layers - sequence_length, self.model.config.hidden_size))
            x = torch.cat([x, padding], dim=1)
        
        outputs = self.model(inputs_embeds=x)
        x = outputs.last_hidden_state[:, 0]  # Take the [CLS] token
        
        # Pass through the classifier
        x = self.classifier(x)
        return x

