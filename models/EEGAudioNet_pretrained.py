from transformers import Wav2Vec2Model
import torch
from torch import nn

class EEGAudioNet_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.depthwise = nn.Conv2d(
            in_channels=256,
            out_channels=768,
            kernel_size=(8, 1),
            stride=(8, 1),
            groups=256, # Depthwise convolution
            bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(768)

        # Load a pre-trained Wav2Vec 2.0 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1000, 2)
        )
        
    def forward(self, x):
        # Apply the convolutional and batchnorm layers
        x = self.conv1(x)  # [Batch, Channels, Height, Width]
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
    
        # Reshape the output to match Wav2Vec2 input expectations
        # Flatten the channel and height dimensions and maintain the sequence in the width dimension
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels*height, width)  # [Batch, New Channels, Sequence Length]
        
        # Permute to [Batch, Sequence Length, New Channels] as Wav2Vec2 expects a "channel" last format
        x = x.permute(0, 2, 1)
    
        # Get the embeddings from the Wav2Vec 2.0 model
        # We need to match the input dimensions that Wav2Vec2 expects
        outputs = self.wav2vec2(x).last_hidden_state
    
        # Apply any additional pooling or reshaping here if necessary
        # For example, you might want to average across the time dimension
        x = torch.mean(outputs, dim=1)
    
        # Pass the embeddings through the classifier
        x = self.classifier(x)
    
        return x

