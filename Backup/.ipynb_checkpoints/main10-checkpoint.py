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
from dataset_handler import CellDataset
from model import CellCounter
from denseweight import DenseWeight
import numpy as np
import matplotlib.pyplot as plt

# Set the output directory for saving models and plots
output_dir = "Experiments/29/"
os.makedirs(output_dir, exist_ok=True)

def custom_collate_fn(batch):
    images, labels, cell_locations = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    cell_locations = [loc.clone().detach() for loc in cell_locations]
    padded_cell_locations = pad_sequence(cell_locations, batch_first=True, padding_value=-1)
    return images, labels, padded_cell_locations

def get_data_loaders(batch_size=16):
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
    """Calculate corrected metrics."""
    # Absolute difference-based metrics
    mae = torch.abs(pred_count - true_count).mean().item()
    rmse = torch.sqrt(torch.mean((pred_count - true_count) ** 2)).item()
    
    # Percentage Accuracy: Penalizes both over-prediction and under-prediction
    percentage_accuracy = (1 - (torch.abs(pred_count - true_count) / true_count)).clamp(0, 1).mean().item() * 100
    return mae, rmse, percentage_accuracy

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, alpha=1.0, threshold=0.10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dw = DenseWeight(alpha=alpha)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_metrics = {"MAE": [], "RMSE": [], "PercentageAccuracy": []}
    val_metrics = {"MAE": [], "RMSE": [], "PercentageAccuracy": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_mae, total_rmse, total_percentage_accuracy = 0.0, 0.0, 0.0

        for i, (inputs, labels, cell_locations) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            cell_count, predicted_locations = model(inputs)
            
            # Calculate location loss
            total_location_loss = 0
            for j in range(inputs.size(0)):  # Loop over batch size
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

            # Metrics calculations
            mae, rmse, percentage_accuracy = calculate_metrics(cell_count, labels)
            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_percentage_accuracy += percentage_accuracy * inputs.size(0)

        # Log epoch metrics
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

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train MAE: {train_metrics['MAE'][-1]:.4f}, Val MAE: {val_metrics['MAE'][-1]:.4f}")
        print(f"Train RMSE: {train_metrics['RMSE'][-1]:.4f}, Val RMSE: {val_metrics['RMSE'][-1]:.4f}")
        print(f"Train Percentage Accuracy: {train_metrics['PercentageAccuracy'][-1]:.2f}%, Val Percentage Accuracy: {val_metrics['PercentageAccuracy'][-1]:.2f}%")

    return model, train_losses, val_losses, train_metrics, val_metrics

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_mae, total_rmse, total_percentage_accuracy = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for inputs, labels, cell_locations in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            cell_count, predicted_locations = model(inputs)

            # Count loss
            count_loss = criterion(cell_count, labels).mean()

            # Location loss
            total_location_loss = 0
            for j in range(inputs.size(0)):  # Loop over batch size
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

    return total_loss / len(data_loader), total_mae / len(data_loader.dataset), total_rmse / len(data_loader.dataset), total_percentage_accuracy / len(data_loader.dataset)

def plot_losses(train_losses, val_losses):
    # Convert tensors to CPU and numpy arrays for plotting
    train_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

def plot_metrics(train_metrics, val_metrics, metric_name):
    # Convert tensors to CPU and numpy arrays for plotting
    train_metrics = [metric.cpu().numpy() if isinstance(metric, torch.Tensor) else metric for metric in train_metrics]
    val_metrics = [metric.cpu().numpy() if isinstance(metric, torch.Tensor) else metric for metric in val_metrics]
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{metric_name.lower()}_plot.png'))
    plt.close()

def main():
    batch_size, num_epochs, learning_rate = 16, 300, 5e-5
    train_loader, val_loader = get_data_loaders(batch_size)
    model = CellCounter()

    trained_model, train_losses, val_losses, train_metrics, val_metrics = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    plot_losses(train_losses, val_losses)
    for metric in ["MAE", "RMSE", "PercentageAccuracy"]:
        plot_metrics(train_metrics[metric], val_metrics[metric], metric)

    print("Training complete. Best model saved in 'Experiments/25/best_model.pth'")


if __name__ == "__main__":
    main()
