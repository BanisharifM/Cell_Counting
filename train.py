# Project Contributors:
# - Mahdi BanisharifDehkordi
# - Gretta Buttelmann
# - Faezeh Rajabi Kouchi
# - Kojo Adu-Gyamfi
# References:
# - Starter code for CellCounter was provided as part of the project setup.
# - DenseLoss and DenseWeight methodologies were adapted from Steininger et al., 2021.

# Specific contributions include improving the training loop structure, preprocessing functions, and metric calculations.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from glob import glob
from tqdm import tqdm
from dataset_handler import CellDataset  # Custom dataset handler
from model import CellCounter  # Updated model with global average pooling
from denseweight import DenseWeight  # For DenseLoss methodology
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Set the output directory for saving models and plots
output_dir = "Experiments/63/"
os.makedirs(output_dir, exist_ok=True)

def preprocess_clusters(image):
    """
    Detect and separate clusters in the input image using a combination of GaussianBlur and Watershed Algorithm.
    
    This function preprocesses images to handle tightly packed cell clusters, which can hinder accurate cell counting.
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
    """
    Custom collate function for DataLoader.
    Handles padding for cell locations to ensure all sequences in a batch have the same length.
    """
    images, labels, cell_locations = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    cell_locations = [loc.clone().detach() for loc in cell_locations]
    padded_cell_locations = pad_sequence(cell_locations, batch_first=True, padding_value=-1)
    return images, labels, padded_cell_locations

def get_data_loaders(batch_size=16):
    """
    Creates data loaders for training and validation datasets.
    Handles transformations for images, including resizing, normalization, and augmentation.
    """
    train_image_paths = glob("IDCIA_Augmentated_V2/images/train/*.tiff")
    val_image_paths = glob("IDCIA_Augmentated_V2/images/val/*.tiff")
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = CellDataset(train_image_paths, transform=train_transform)
    val_dataset = CellDataset(val_image_paths, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    return train_loader, val_loader

def calculate_metrics(pred_count, true_count):
    """
    Calculate evaluation metrics: Mean Absolute Error (MAE), Root Mean Square Error (RMSE),
    and Percentage Accuracy.
    """
    mae = torch.abs(pred_count - true_count).mean().item()
    rmse = torch.sqrt(torch.mean((pred_count - true_count) ** 2)).item()
    percentage_accuracy = (1 - (torch.abs(pred_count - true_count) / true_count)).clamp(0, 1).mean().item() * 100
    return mae, rmse, percentage_accuracy

# Weight_decay= 1e-5
# Learning_Rate= 5e-5

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=5e-5, alpha=1.0, weight_decay=1e-5 , patience=100):
    """
    Trains the model with specified parameters.
    Includes DenseLoss for handling unbalanced datasets and early stopping for avoiding overfitting.
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)
    dw = DenseWeight(alpha=alpha)

    best_val_loss = float('inf')
    best_epoch = -1
    train_losses, val_losses = [], []
    train_metrics = {"MAE": [], "RMSE": [], "PercentageAccuracy": []}
    val_metrics = {"MAE": [], "RMSE": [], "PercentageAccuracy": []}
    early_stop_counter = 0

    # Add the missing epoch loop
    for epoch in range(num_epochs):  # Define the epoch loop
        model.train()
        running_loss = 0.0
        total_mae, total_rmse, total_percentage_accuracy = 0.0, 0.0, 0.0

        for inputs, labels, cell_locations in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1)

            # Convert each image to NumPy, apply preprocessing, and convert back to tensor
            cluster_inputs = torch.stack([
                torch.tensor(
                    preprocess_clusters(img.cpu().permute(1, 2, 0).numpy()),  # Convert to HWC format for OpenCV
                    dtype=torch.float32
                ).permute(2, 0, 1)  # Convert back to CHW format for PyTorch
                for img in inputs
            ]).to(device)

            # Forward pass
            cell_count, uncertainty, predicted_locations = model(inputs, cluster_inputs)

            # Initialize location loss
            total_location_loss = 0

            # Calculate location loss
            for j in range(inputs.size(0)):
                true_loc = cell_locations[j].to(device)
                num_locations = true_loc.size(0)
                pred_loc = predicted_locations[j].view(-1, 2)[:num_locations]
                dists = torch.cdist(pred_loc.unsqueeze(0), true_loc.unsqueeze(0)).squeeze(0)
                min_dists, min_indices = dists.min(dim=1)
                location_loss = F.mse_loss(pred_loc, true_loc[min_indices])
                total_location_loss += location_loss

            # Count loss and total loss
            count_loss = criterion(cell_count, labels).mean()
            total_loss = count_loss + total_location_loss / inputs.size(0)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

            # Calculate metrics
            mae, rmse, percentage_accuracy = calculate_metrics(cell_count, labels)
            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_percentage_accuracy += percentage_accuracy * inputs.size(0)

        # Log epoch loss
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_metrics["MAE"].append(total_mae / len(train_loader.dataset))
        train_metrics["RMSE"].append(total_rmse / len(train_loader.dataset))
        train_metrics["PercentageAccuracy"].append(total_percentage_accuracy / len(train_loader.dataset))

        # Evaluate on validation set
        val_loss, val_mae, val_rmse, val_percentage_accuracy = evaluate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_metrics["MAE"].append(val_mae)
        val_metrics["RMSE"].append(val_rmse)
        val_metrics["PercentageAccuracy"].append(val_percentage_accuracy)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1  # Save the epoch (1-based indexing)
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Best model updated at epoch {best_epoch} with Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

    return model, train_losses, val_losses, train_metrics, val_metrics, best_epoch, best_val_loss

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_mae, total_rmse, total_percentage_accuracy = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for inputs, labels, cell_locations in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1)

            # Generate cluster-separated images
            cluster_inputs = torch.stack([
                torch.tensor(
                    preprocess_clusters(img.cpu().permute(1, 2, 0).numpy()),  # Convert to HWC format for OpenCV
                    dtype=torch.float32
                ).permute(2, 0, 1)  # Convert back to CHW format for PyTorch
                for img in inputs
            ]).to(device)

            # Forward pass
            cell_count, uncertainty, predicted_locations = model(inputs, cluster_inputs)

            # Count loss
            count_loss = criterion(cell_count, labels).mean()

            # Location loss
            total_location_loss = 0
            for j in range(inputs.size(0)):
                true_loc = cell_locations[j].to(device)
                num_locations = true_loc.size(0)
                pred_loc = predicted_locations[j].view(-1, 2)[:num_locations]
                dists = torch.cdist(pred_loc.unsqueeze(0), true_loc.unsqueeze(0)).squeeze(0)
                min_dists, min_indices = dists.min(dim=1)
                location_loss = F.mse_loss(pred_loc, true_loc[min_indices])
                total_location_loss += location_loss

            total_loss += count_loss + total_location_loss / inputs.size(0)

            # Metrics calculations
            mae, rmse, percentage_accuracy = calculate_metrics(cell_count, labels)
            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_percentage_accuracy += percentage_accuracy * inputs.size(0)

    return (
        total_loss / len(data_loader),
        total_mae / len(data_loader.dataset),
        total_rmse / len(data_loader.dataset),
        total_percentage_accuracy / len(data_loader.dataset),
    )


def plot_losses(train_losses, val_losses):
    # Ensure all tensors are moved to CPU and converted to NumPy
    train_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.yscale('log')  # Log scale to handle large loss values
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()



def plot_metrics(train_metrics, val_metrics):
    """
    Plot each metric separately from the metrics dictionaries.
    """
    for metric_name in train_metrics.keys():
        # Convert tensors to CPU and NumPy arrays for plotting
        train_values = [float(metric) for metric in train_metrics[metric_name]]
        val_values = [float(metric) for metric in val_metrics[metric_name]]

        # Plot the metric
        plt.figure(figsize=(10, 5))
        plt.plot(train_values, label=f'Training {metric_name}')
        plt.plot(val_values, label=f'Validation {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Training and Validation {metric_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{metric_name.lower()}_plot.png'))
        plt.close()


def main():
    batch_size, num_epochs, learning_rate = 32, 300, 5e-4
    train_loader, val_loader = get_data_loaders(batch_size)
    model = CellCounter()

    # Train the model
    trained_model, train_losses, val_losses, train_metrics, val_metrics, best_epoch, best_val_loss = train_model(
        model, train_loader, val_loader, num_epochs, learning_rate, patience=100
    )

    # Plot losses and metrics
    plot_losses(train_losses, val_losses)
    plot_metrics(train_metrics, val_metrics)


    print(f"Training complete. Best model saved in '{output_dir}best_model.pth' (Epoch {best_epoch}) with Val Loss: {best_val_loss:.4f}.")

    # Load the best model for final evaluation
    best_model = CellCounter()
    best_model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    best_model.eval()

    # Recalculate metrics directly on the validation set
    criterion = nn.L1Loss(reduction='none')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = best_model.to(device)

    # Directly calculate metrics
    all_cell_counts = []
    all_labels = []

    for inputs, labels, _ in val_loader:
        inputs, labels = inputs.to(device), labels.to(device).view(-1)
        cell_counts, _ = best_model(inputs)
        all_cell_counts.append(cell_counts)
        all_labels.append(labels)

    all_cell_counts = torch.cat(all_cell_counts)
    all_labels = torch.cat(all_labels)

    # Final metrics
    val_mae, val_rmse, val_percentage_accuracy = calculate_metrics(all_cell_counts, all_labels)
    print(f"Final Evaluation on Best Model:")
    print(f"MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, Accuracy: {val_percentage_accuracy:.2f}%")




if __name__ == "__main__":
    main()
