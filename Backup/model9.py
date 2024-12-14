import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class CellCounter(nn.Module):
    '''
    A ResNet-based model for cell counting and cell location prediction.
    The ResNet is pre-trained on ImageNet.
    The final layers output both the cell count and the predicted cell locations.
    '''

    def __init__(self, fine_tune=True, num_locations=1000):
        super(CellCounter, self).__init__()
        # Load the pre-trained ResNet model with specified weights
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer with outputs for cell count
        # ResNet's fc layer is by default (2048 -> 1000), we change it to (2048 -> 2 + num_locations * 2)
        # where the additional outputs are used for cell location predictions
        self.resnet.fc = nn.Linear(
            in_features=self.resnet.fc.in_features, 
            out_features=2 + num_locations * 2
        )

        # Freeze all layers by default
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Fine-tune the last two ResNet layers if specified
        if fine_tune:
            for param in self.resnet.layer4.parameters():  # Unfreeze layer4
                param.requires_grad = True
            for param in self.resnet.fc.parameters():  # Unfreeze final fully connected layer
                param.requires_grad = True

        self.num_locations = num_locations

    def forward(self, x):
        # Forward pass through ResNet
        x = self.resnet(x)
        
        # Separate outputs for cell count, uncertainty, and cell locations
        cell_count = x[:, 0]
        uncertainty = x[:, 1]

        # Predicted locations, reshape to [batch_size, num_locations, 2] for x, y coordinates
        predicted_locations = x[:, 2:].view(-1, self.num_locations, 2)
        
        return cell_count, predicted_locations