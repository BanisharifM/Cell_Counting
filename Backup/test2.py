import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from model import CellCounter
from main10 import evaluate_model

# Directory to save images with predicted cell locations
output_dir = "Testing/5/"
os.makedirs(output_dir, exist_ok=True)

# Custom Dataset class for test data
class TestCellDataset(Dataset):
    def __init__(self, image_paths, gt_folder, transform=None, return_path=False):
        self.image_paths = image_paths
        self.gt_folder = gt_folder
        self.transform = transform
        self.return_path = return_path  # Control if img_path is returned

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = img_path.split('/')[-1].replace('.tiff', '.csv')
        gt_path = f"{self.gt_folder}/{img_name}"

        img = Image.open(img_path).convert('RGB')
        gt = pd.read_csv(gt_path)
        label = torch.tensor(gt.shape[0], dtype=torch.float32).unsqueeze(0)
        cell_locations = torch.tensor(gt[['X', 'Y']].values, dtype=torch.float32)  # True cell locations

        if self.transform:
            img_transformed = self.transform(img)
        else:
            img_transformed = img

        if self.return_path:
            return img_transformed, label, cell_locations, img_path
        else:
            return img_transformed, label, cell_locations

def draw_predictions_on_image(image, cell_centers, output_path):
    """Draws small red circles at the predicted cell centers on the image."""
    draw = ImageDraw.Draw(image)
    radius = 3  # Radius for the circle
    for (x, y) in cell_centers:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
    image.save(output_path)

def get_test_loader(batch_size=1, return_path=False):
    test_image_paths = glob("IDCIA_Augmentated_V2/images/test/*.tiff")
    gt_folder = "IDCIA_Augmentated_V2/ground_truth"
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = TestCellDataset(test_image_paths, gt_folder=gt_folder, transform=test_transform, return_path=return_path)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def test_model():
    # Load the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCounter().to(device)
    model.load_state_dict(torch.load("Experiments/26/best_model.pth"))
    model.eval()

    # Data loader with img_path included
    test_loader_with_paths = get_test_loader(batch_size=1, return_path=True)

    for i, (img_tensor, label, cell_locations, img_path) in enumerate(test_loader_with_paths):
        img_tensor = img_tensor.to(device)
        predicted_count, predicted_locations = model(img_tensor)
        
        actual_count = int(label.item())
        predicted_count_int = int(predicted_count.item())
        print(f"Image: {img_path[0]} | Predicted Cell Count: {predicted_count_int} | Actual Cell Count: {actual_count}")
        
        predicted_locations = predicted_locations[0].cpu().detach().numpy()
        
        # Rescale predictions to original image size
        original_image = Image.open(img_path[0]).convert('RGB')
        width, height = original_image.size
        predicted_locations[:, 0] *= width / img_tensor.shape[2]
        predicted_locations[:, 1] *= height / img_tensor.shape[3]
        
        # Draw predictions on image
        cell_centers = [(x, y) for x, y in predicted_locations[:predicted_count_int]]
        output_path = os.path.join(output_dir, os.path.basename(img_path[0]).replace('.tiff', '_pred.png'))
        draw_predictions_on_image(original_image, cell_centers, output_path)
        print(f"Saved predicted image: {output_path}")

    # Evaluation metrics calculation
    criterion = nn.L1Loss()
    test_loader = get_test_loader(batch_size=1, return_path=False)
    test_loss, test_mae, test_rmse, test_percentage_accuracy, test_accuracy = evaluate_model(
        model, test_loader, criterion, device, threshold=0.05
    )

    # Display metrics
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test Accuracy (within 5% threshold): {test_accuracy:.2f}%")
    print(f"Test Percentage Accuracy: {test_percentage_accuracy:.2f}%")

if __name__ == "__main__":
    test_model()
