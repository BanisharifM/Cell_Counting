import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from dataset_handler import CellDataset
from model import CellCounter

def custom_collate_fn(batch):
    images, labels, cell_locations = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

def get_test_loader(batch_size=16):
    # Test dataset paths
    test_image_paths = glob("IDCIA_Augmentated_V2/images/test/*.tiff")
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = CellDataset(test_image_paths, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    return test_loader

def calculate_metrics(pred_count, true_count):
    """Calculate corrected metrics."""
    mae = torch.abs(pred_count - true_count).mean().item()
    rmse = torch.sqrt(torch.mean((pred_count - true_count) ** 2)).item()
    percentage_accuracy = (1 - (torch.abs(pred_count - true_count) / true_count)).clamp(0, 1).mean().item() * 100
    return mae, rmse, percentage_accuracy

def evaluate_on_test(best_model_path, batch_size=16):
    # Load the test data
    test_loader = get_test_loader(batch_size)

    # Load the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCounter()
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    # Metrics variables
    all_pred_counts = []
    all_true_counts = []

    # Evaluate on the test set
    print("\nTest Set Predictions:")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            pred_counts, _ = model(inputs)  # Ignore location outputs

            for j in range(inputs.size(0)):  # Loop through the batch
                predicted_count_int = int(round(pred_counts[j].item()))
                actual_count = int(labels[j].item())
                print(f"Image {i * batch_size + j + 1} | Predicted Cell Count: {predicted_count_int} | Actual Cell Count: {actual_count}")

            all_pred_counts.append(pred_counts)
            all_true_counts.append(labels)

    # Concatenate all predictions and ground truth
    all_pred_counts = torch.cat(all_pred_counts)
    all_true_counts = torch.cat(all_true_counts)

    # Calculate final metrics
    mae, rmse, percentage_accuracy = calculate_metrics(all_pred_counts, all_true_counts)
    print("\nFinal Metrics on Test Set:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, Accuracy: {percentage_accuracy:.2f}%")


if __name__ == "__main__":
    # Define the best model path and batch size
    best_model_path = "Experiments/42/best_model.pth"
    batch_size = 16

    # Evaluate on the test set
    evaluate_on_test(best_model_path, batch_size)
