import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class CellCounter(nn.Module):
    def __init__(self, fine_tune=True, num_locations=1000):
        super(CellCounter, self).__init__()
        # Raw Image Branch
        self.raw_resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.raw_resnet.fc = nn.Identity()  # Remove classification head

        # Cluster-Separated Branch
        self.cluster_resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.cluster_resnet.fc = nn.Identity()  # Remove classification head

        # Freeze all layers by default
        for param in self.raw_resnet.parameters():
            param.requires_grad = False
        for param in self.cluster_resnet.parameters():
            param.requires_grad = False

        # Fine-tune the last two ResNet layers if specified
        if fine_tune:
            for param in self.raw_resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.raw_resnet.fc.parameters():
                param.requires_grad = True
            for param in self.cluster_resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.cluster_resnet.fc.parameters():
                param.requires_grad = True

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2048 * 2, 512),  # Combine features from both ResNets
            nn.ReLU(),
            nn.Linear(512, 2 + num_locations * 2)  # Predict cell count, uncertainty, and locations
        )

        self.num_locations = num_locations

    def forward(self, x, cluster_input):
        # Extract features for original and cluster-separated images
        original_features = self.raw_resnet(x)
        cluster_features = self.cluster_resnet(cluster_input)

        # Combine features
        combined_features = torch.cat([original_features, cluster_features], dim=1)

        # Predict using fusion layer
        predictions = self.fusion(combined_features)

        # Split predictions
        cell_count = predictions[:, 0]
        uncertainty = predictions[:, 1]
        predicted_locations = predictions[:, 2:].view(-1, self.num_locations, 2)

        return cell_count, uncertainty, predicted_locations
