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
        x = self.conv1(x)  # [Batch, 256 Channels, Height, Width]
        x = self.batchnorm1(x)
        x = self.depthwise(x)  # [Batch, 768 Channels, New Height, Width]
        x = self.batchnorm2(x)
    
        # Reshape the output to a 1D sequence
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels * height * width)  # Flatten to [Batch, Sequence Length]
        
        # Normalize the data to match the expected format of Wav2Vec 2.0
        # You might need to scale the data to a range that Wav2Vec 2.0 was trained on.
        x = torch.tanh(x)  # Example normalization, adjust as needed
    
        # Get the embeddings from the Wav2Vec 2.0 model
        outputs = self.wav2vec2(x).last_hidden_state
    
        # Apply any additional pooling or reshaping here if necessary
        x = torch.mean(outputs, dim=1)
    
        # Pass the embeddings through the classifier
        x = self.classifier(x)
    
        return x

