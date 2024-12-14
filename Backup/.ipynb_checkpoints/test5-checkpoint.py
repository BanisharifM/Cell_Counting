import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from dataset_handler import CellDataset
from model import CellCounter
import numpy as np
import cv2

def preprocess_clusters(image):
    """
    Detect and separate clusters in the input image.
    Uses a combination of GaussianBlur and Watershed Algorithm.
    """
    # Ensure the input is 3-channel
    if len(image.shape) == 2:  # Already grayscale
        gray = image
    elif image.shape[-1] == 3:  # RGB or BGR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Apply Gaussian Blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Convert to uint8 for thresholding
    blurred_uint8 = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Binary thresholding
    _, binary = cv2.threshold(blurred_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Distance transform and Watershed
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(binary, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Convert markers to the required type
    markers = markers.astype(np.int32)

    # Watershed algorithm
    watershed_input = np.stack([gray] * 3, axis=-1).astype(np.uint8)  # Ensure 3 channels
    markers = cv2.watershed(watershed_input, markers)

    # Extract the cluster-separated result
    cluster_separated = np.zeros_like(gray)
    cluster_separated[markers > 1] = 255

    # Ensure 3 channels
    cluster_separated = np.stack([cluster_separated] * 3, axis=-1)

    return cluster_separated


def custom_collate_fn(batch):
    images, labels, cell_locations = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels


def get_test_loader(batch_size=16):
    test_image_paths = glob("IDCIA_Augmentated_V2/images/test/*.tiff")
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = CellDataset(test_image_paths, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
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


def evaluate_on_test(best_model_path, batch_size=16, n=30):
    test_loader = get_test_loader(batch_size)

    # Load the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCounter()
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    all_pred_counts = []
    all_true_counts = []

    print("\nTest Set Predictions:")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device).view(-1)

            # Generate cluster-separated inputs
            cluster_inputs = torch.stack([
                torch.tensor(
                    preprocess_clusters(img.cpu().permute(1, 2, 0).numpy()),  # Convert to HWC for OpenCV
                    dtype=torch.float32
                ).permute(2, 0, 1)  # Convert back to CHW for PyTorch
                for img in inputs
            ]).to(device)

            # Forward pass
            pred_counts, _, _ = model(inputs, cluster_inputs)

            for j in range(inputs.size(0)):
                predicted_count_int = int(round(pred_counts[j].item()))
                actual_count = int(labels[j].item())
                print(f"Image {i * batch_size + j + 1} | Predicted Cell Count: {predicted_count_int} | Actual Cell Count: {actual_count}")

            all_pred_counts.append(pred_counts)
            all_true_counts.append(labels)

    all_pred_counts = torch.cat(all_pred_counts)
    all_true_counts = torch.cat(all_true_counts)

    mae, rmse, percentage_accuracy, acp = calculate_metrics(all_pred_counts, all_true_counts, n=n)
    print("\nFinal Metrics on Test Set:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, Accuracy: {percentage_accuracy:.2f}%, ACP: {acp:.2f}%")


if __name__ == "__main__":
    best_model_path = "Experiments/63/best_model.pth"
    batch_size = 16
    evaluate_on_test(best_model_path, batch_size)
