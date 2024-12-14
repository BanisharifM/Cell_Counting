import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
import pandas as pd
from PIL import Image
from raw_model import CellCounter  # Ensure this matches your raw model's file path

# Directory to save the testing results
output_dir = "Raw_Model/Raw_Result"
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
        return img_transformed, label, cell_locations

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

def calculate_metrics(pred_count, true_count, n=30):
    """Calculate corrected metrics including ACP."""
    mae = torch.abs(pred_count - true_count).mean().item()
    rmse = torch.sqrt(torch.mean((pred_count - true_count) ** 2)).item()
    percentage_accuracy = (1 - (torch.abs(pred_count - true_count) / true_count)).clamp(0, 1).mean().item() * 100

    # Dynamic ACP calculation
    acp_rates = torch.where(true_count < n, 0.10, 0.05)  # Use 0.10 for small cell counts, 0.05 for larger ones
    acp = ((torch.abs(pred_count - true_count) <= acp_rates * true_count).float().mean().item()) * 100

    return mae, rmse, percentage_accuracy, acp

def test_model():
    # Load the best raw model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCounter().to(device)
    model.load_state_dict(torch.load("Raw_Model/Raw_Result/cell_counter.pth"))  # Ensure the correct raw model path
    model.eval()

    # Data loader
    test_loader = get_test_loader(batch_size=1)

    all_pred_counts = []
    all_true_counts = []

    print("\nTest Set Predictions:")
    with torch.no_grad():
        for i, (img_tensor, label, _) in enumerate(test_loader):
            img_tensor, label = img_tensor.to(device), label.to(device)
            predicted_count = model(img_tensor).squeeze(0)  # Raw model predicts cell count only
            
            # Log the predicted count and actual count from ground truth
            predicted_count_int = int(round(predicted_count.item()))
            actual_count = int(label.item())
            print(f"Image {i+1} | Predicted Cell Count: {predicted_count_int} | Actual Cell Count: {actual_count}")

            all_pred_counts.append(predicted_count)
            all_true_counts.append(label)

    all_pred_counts = torch.cat(all_pred_counts)
    all_true_counts = torch.cat(all_true_counts)

    # Calculate metrics
    mae, rmse, percentage_accuracy, acp = calculate_metrics(all_pred_counts, all_true_counts)

    print("\nFinal Metrics on Test Set:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, Accuracy: {percentage_accuracy:.2f}%, ACP: {acp:.2f}%")

if __name__ == "__main__":
    test_model()
