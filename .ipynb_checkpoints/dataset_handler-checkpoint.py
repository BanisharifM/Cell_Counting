import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class CellDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        gt_path = img_path.replace('images', 'ground_truth').replace('val/', '').replace('train', '').replace('.tiff', '.csv')

        # Load the image
        img = Image.open(img_path).convert('RGB')

        # Load ground truth data
        gt = pd.read_csv(gt_path)

        # Extract cell count
        label = torch.tensor(gt.shape[0], dtype=torch.float32).unsqueeze(0)  # Number of rows = cell count

        # Extract cell locations using 'X' and 'Y' columns
        if 'X' in gt.columns and 'Y' in gt.columns:
            cell_locations = torch.tensor(gt[['X', 'Y']].values, dtype=torch.float32)
        else:
            raise KeyError("CSV file does not contain 'X' and 'Y' columns for cell locations.")

        # Apply the transform if available
        if self.transform:
            img = self.transform(img)

        # Return image, cell count, and cell locations
        return img, label, cell_locations
