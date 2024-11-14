import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
import pandas as pd
from PIL import Image
from model import CellCounter
from main8 import evaluate_model  # Adjust to your actual filename

# Custom Dataset class for test data
class TestCellDataset(Dataset):
    def __init__(self, image_paths, gt_folder, transform=None):
        self.image_paths = image_paths
        self.gt_folder = gt_folder  # Folder where ground truth files are located
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = img_path.split('/')[-1].replace('.tiff', '.csv')
        gt_path = f"{self.gt_folder}/{img_name}"  # Use the general gt_folder path

        # Load image and ground truth
        img = Image.open(img_path).convert('RGB')
        gt = pd.read_csv(gt_path)
        label = torch.tensor(gt.shape[0], dtype=torch.float32).unsqueeze(0)

        # Apply the transform if available
        if self.transform:
            img = self.transform(img)

        return img, label

def get_test_loader(batch_size=8):
    test_image_paths = glob("IDCIA_Augmentated_V2/images/test/*.tiff")
    gt_folder = "IDCIA_Augmentated_V2/ground_truth"  # Define the path to ground truth CSVs

    # Define test transform (same as validation)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create test dataset and dataloader
    test_dataset = TestCellDataset(test_image_paths, gt_folder=gt_folder, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader

def test_model():
    # Load the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCounter().to(device)
    model.load_state_dict(torch.load("Experiments/24/cell_counter_resnet.pth"))
    model.eval()

    # Get test data loader
    test_loader = get_test_loader(batch_size=8)

    # Define the criterion (for consistency with training)
    criterion = nn.L1Loss()
    
    # Evaluate on the test set
    test_loss, test_mae, test_rmse, test_acp, test_accuracy, test_percentage_accuracy = evaluate_model(
        model, test_loader, criterion, device, threshold=0.10
    )

    # Print test metrics
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test ACP: {test_acp:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Percentage Accuracy: {test_percentage_accuracy:.2f}%")

if __name__ == "__main__":
    test_model()
