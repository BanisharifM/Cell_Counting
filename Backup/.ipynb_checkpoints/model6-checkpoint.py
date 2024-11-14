import torch
import torch.nn as nn
import torch.nn.functional as F

class CellCounter(nn.Module):
    def __init__(self, use_batch_norm=False):
        super(CellCounter, self).__init__()
        self.use_batch_norm = use_batch_norm

        # Convolutional and pooling layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.pool3 = nn.MaxPool2d(2, 2)

        # Calculate the final feature map size dynamically
        sample_input = torch.randn(1, 3, 256, 256)  # Replace (256, 256) with your actual input dimensions
        final_dim = self._get_flattened_size(sample_input)

        # Fully connected layers
        self.fc1 = nn.Linear(final_dim, 512)
        self.fc2 = nn.Linear(512, 1)

    def _get_flattened_size(self, x):
        """Passes a tensor through conv and pooling layers to calculate flattened size."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        cell_count = self.fc2(x)  # Predicted cell count

        # Placeholder for predicted_locations: Replace with actual logic if needed
        batch_size = cell_count.size(0)
        predicted_locations = torch.zeros(batch_size, 10, 2).to(cell_count.device)  # Example shape [batch, num_cells, 2]

        return cell_count, predicted_locations

