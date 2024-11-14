import torch
import torch.nn as nn
import torchvision.models as models

# Creating the model with ResNet backbone for cell counting
class CellCounter(nn.Module):
    '''
    A ResNet-based model for cell counting. The ResNet is pre-trained on ImageNet, 
    and we replace the final fully connected layer to output a single scalar (cell count).
    '''

    def __init__(self):
        super(CellCounter, self).__init__()
        # Load the pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        # self.resnet = models.resnet50(pretrained=False)

        # Replace the final fully connected layer
        # ResNet's fc layer is by default (2048 -> 1000), we change it to (2048 -> 1)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=1)

    def forward(self, x):
        # Forward pass through ResNet
        x = self.resnet(x)

        # Return the cell count as a scalar
        return x