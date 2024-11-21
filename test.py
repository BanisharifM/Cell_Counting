import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
import pandas as pd
from PIL import Image
import numpy as np
from Backup.model5 import CellCounter
from train import evaluate_model  # Ensure this imports from the correct training script

# Directory to load the test data
output_dir = "Testing/6/"
os.makedirs(output_dir, exist_ok=True)

# Custom Dataset class for test data
class TestCellDataset(Dataset):
    def __init__(self, image_paths, gt_folder, transform=None):
        self.image_paths = image_paths
        self.gt_folder = gt_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = img_path.split('/')[-1].replace('.tiff', '.csv')
        gt_path = f"{self.gt_folder}/{img_name}"

        img = Image.open(img_path).convert('RGB')
        gt = pd.read_csv(gt_path)
        label = torch.tensor(gt.shape[0], dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img_transformed = self.transform(img)
        else:
            img_transformed = img

        cell_locations = torch.tensor(gt[['X', 'Y']].values, dtype=torch.float32)
        return img_transformed, label, cell_locations  # cell_locations is retained for consistency

def get_test_loader(batch_size=1):
    test_image_paths = glob("IDCIA_Augmentated_V2/images/test/*.tiff")
    gt_folder = "IDCIA_Augmentated_V2/ground_truth"

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_dataset = TestCellDataset(test_image_paths, gt_folder=gt_folder, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader

def test_model():
    # Load the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCounter().to(device)
    model.load_state_dict(torch.load("Experiments/36/best_model.pth"))
    model.eval()

    # Data loader
    test_loader = get_test_loader(batch_size=1)

    for i, (img_tensor, label, cell_locations) in enumerate(test_loader):
        img_tensor = img_tensor.to(device)
        predicted_count, _ = model(img_tensor)
        
        # Log the predicted count and actual count from ground truth
        actual_count = int(label.item())  # Ground truth count of cells
        predicted_count_int = int(predicted_count.item())  # Model's predicted count of cells
        print(f"Image {i+1} | Predicted Cell Count: {predicted_count_int} | Actual Cell Count: {actual_count}")

    # Define the criterion for evaluation
    criterion = nn.L1Loss()

    # Evaluate model on test data
    test_loss, test_mae, test_rmse, test_percentage_accuracy = evaluate_model(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test Percentage Accuracy: {test_percentage_accuracy:.2f}%")

if __name__ == "__main__":
    test_model()
