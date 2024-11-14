import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights  # Import weights enum

class CellCounter(nn.Module):
    '''
    A ResNet-based model for cell counting. The ResNet is pre-trained on ImageNet, 
    and we replace the final fully connected layer to output a single scalar (cell count).
    '''

    def __init__(self, fine_tune=True):
        super(CellCounter, self).__init__()
        # Load the pre-trained ResNet model with specified weights
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer
        # ResNet's fc layer is by default (2048 -> 1000), we change it to (2048 -> 1)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=1)

        # Freeze all layers by default
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Fine-tune the last two ResNet layers if specified
        if fine_tune:
            for param in self.resnet.layer4.parameters():  # Unfreeze layer4
                param.requires_grad = True
            for param in self.resnet.fc.parameters():  # Unfreeze final fully connected layer
                param.requires_grad = True

    def forward(self, x):
        # Forward pass through ResNet
        x = self.resnet(x)
        # Return the cell count as a scalar
        return x
