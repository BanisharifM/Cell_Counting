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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set the output directory for saving models and plots
output_dir = "Experiments/26/"
os.makedirs(output_dir, exist_ok=True)

def custom_collate_fn(batch):
    images, labels, cell_locations = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    cell_locations = [loc.clone().detach() for loc in cell_locations]  # Update to avoid UserWarning
    padded_cell_locations = pad_sequence(cell_locations, batch_first=True, padding_value=-1)  # Pad with -1

    return images, labels, padded_cell_locations

def get_data_loaders(batch_size=16):

    train_image_paths = glob("IDCIA_Augmentated_V2/images/train/*.tiff")
    val_image_paths = glob("IDCIA_Augmentated_V2/images/val/*.tiff")

    # Define transforms
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

    # Create datasets
    train_dataset = CellDataset(train_image_paths, transform=train_transform)
    val_dataset = CellDataset(val_image_paths, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    return train_loader, val_loader 


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, alpha=1.0, threshold=0.10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dw = DenseWeight(alpha=alpha)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_metrics = {"MAE": [], "RMSE": [], "ACP": [], "Accuracy": [], "PercentageAccuracy": []}
    val_metrics = {"MAE": [], "RMSE": [], "ACP": [], "Accuracy": [], "PercentageAccuracy": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss, total_mae, total_rmse, total_acp, correct_cells, total_percentage_accuracy = 0.0, 0.0, 0.0, 0.0, 0, 0.0

        for i, (inputs, labels, cell_locations) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            cell_count, predicted_locations = model(inputs)
            
            # Calculate location loss
            total_location_loss = 0
            for j in range(inputs.size(0)):  # Loop over batch size
                true_loc = cell_locations[j].to(device)
                num_locations = true_loc.size(0)

                pred_loc = predicted_locations[j].view(-1, 2)[:num_locations]

                # Calculate pairwise distance and find closest matches
                dists = torch.cdist(pred_loc.unsqueeze(0), true_loc.unsqueeze(0)).squeeze(0)
                min_dists, min_indices = dists.min(dim=1)
                location_loss = F.mse_loss(pred_loc, true_loc[min_indices])
                total_location_loss += location_loss

            # Calculate count loss and total loss
            count_loss = criterion(cell_count, labels).mean()
            total_loss = count_loss + total_location_loss / inputs.size(0)  # Normalize by batch size

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

            # Metrics calculations
            mae = torch.mean(torch.abs(cell_count - labels)).item()
            rmse = torch.sqrt(torch.mean((cell_count - labels) ** 2)).item()
            acp = torch.mean((torch.abs(cell_count - labels) <= 0.05 * torch.abs(labels)).float()).item()

            # Threshold-based accuracy calculation
            predicted_cells = torch.round(cell_count)
            actual_cells = torch.round(labels)
            lower_bound = actual_cells * (1 - threshold)
            upper_bound = actual_cells * (1 + threshold)
            correct_cells += torch.sum((predicted_cells >= lower_bound) & (predicted_cells <= upper_bound)).item()

            # Percentage accuracy calculation
            percentage_accuracy = torch.mean((predicted_cells / actual_cells).clamp(0, 1)).item() * 100
            total_percentage_accuracy += percentage_accuracy

            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_acp += acp * inputs.size(0)

        # Log epoch metrics
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_metrics["MAE"].append(total_mae / len(train_loader.dataset))
        train_metrics["RMSE"].append(total_rmse / len(train_loader.dataset))
        train_metrics["ACP"].append(total_acp / len(train_loader.dataset))
        train_metrics["Accuracy"].append(correct_cells / len(train_loader.dataset) * 100)
        train_metrics["PercentageAccuracy"].append(total_percentage_accuracy / len(train_loader))

        # Evaluate on validation set
        val_loss, val_mae, val_rmse, val_acp, val_accuracy, val_percentage_accuracy = evaluate_model(
            model, val_loader, criterion, device, threshold
        )
        val_losses.append(val_loss)
        val_metrics["MAE"].append(val_mae)
        val_metrics["RMSE"].append(val_rmse)
        val_metrics["ACP"].append(val_acp)
        val_metrics["Accuracy"].append(val_accuracy)
        val_metrics["PercentageAccuracy"].append(val_percentage_accuracy)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train MAE: {train_metrics['MAE'][-1]:.4f}, Val MAE: {val_metrics['MAE'][-1]:.4f}")
        print(f"Train RMSE: {train_metrics['RMSE'][-1]:.4f}, Val RMSE: {val_metrics['RMSE'][-1]:.4f}")
        print(f"Train ACP: {train_metrics['ACP'][-1]:.4f}, Val ACP: {val_metrics['ACP'][-1]:.4f}")
        print(f"Train Accuracy: {train_metrics['Accuracy'][-1]:.2f}%, Val Accuracy: {val_metrics['Accuracy'][-1]:.2f}%")
        print(f"Train Percentage Accuracy: {train_metrics['PercentageAccuracy'][-1]:.2f}%, Val Percentage Accuracy: {val_metrics['PercentageAccuracy'][-1]:.2f}%")

    return model, train_losses, val_losses, train_metrics, val_metrics



def evaluate_model(model, data_loader, criterion, device, threshold=0.05):
    model.eval()
    total_loss, total_mae, total_rmse, total_acp, correct_cells, total_percentage_accuracy = 0.0, 0.0, 0.0, 0.0, 0, 0.0

    with torch.no_grad():
        for inputs, labels, cell_locations in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            cell_count, predicted_locations = model(inputs)

            # Calculate count loss
            count_loss = criterion(cell_count, labels).mean()

            # Calculate location loss
            total_location_loss = 0
            for j in range(inputs.size(0)):  # Loop over batch size
                true_loc = cell_locations[j].to(device)
                num_locations = true_loc.size(0)

                # Predicted locations for the current item in batch
                pred_loc = predicted_locations[j].view(-1, 2)[:num_locations]

                # Calculate pairwise distance and find closest matches
                dists = torch.cdist(pred_loc.unsqueeze(0), true_loc.unsqueeze(0)).squeeze(0)
                min_dists, min_indices = dists.min(dim=1)
                location_loss = F.mse_loss(pred_loc, true_loc[min_indices])
                total_location_loss += location_loss

            # Combine count and location losses
            total_loss += count_loss + total_location_loss / inputs.size(0)

            # Metrics calculations
            mae = torch.mean(torch.abs(cell_count - labels)).item()
            rmse = torch.sqrt(torch.mean((cell_count - labels) ** 2)).item()
            acp = torch.mean((torch.abs(cell_count - labels) <= 0.05 * torch.abs(labels)).float()).item()

            predicted_cells = torch.round(cell_count)
            actual_cells = torch.round(labels)
            lower_bound = actual_cells * (1 - threshold)
            upper_bound = actual_cells * (1 + threshold)
            correct_cells += torch.sum((predicted_cells >= lower_bound) & (predicted_cells <= upper_bound)).item()

            percentage_accuracy = torch.mean((predicted_cells / actual_cells).clamp(0, 1)).item() * 100
            total_percentage_accuracy += percentage_accuracy

            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_acp += acp * inputs.size(0)

    accuracy = correct_cells / len(data_loader.dataset) * 100
    percentage_accuracy = total_percentage_accuracy / len(data_loader)
    return total_loss / len(data_loader), total_mae / len(data_loader.dataset), total_rmse / len(data_loader.dataset), total_acp / len(data_loader.dataset), accuracy, percentage_accuracy



def plot_losses(train_losses, val_losses):
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
    batch_size, num_epochs, learning_rate = 16, 200, 5e-5
    train_loader, val_loader = get_data_loaders(batch_size)
    model = CellCounter()

    trained_model, train_losses, val_losses, train_metrics, val_metrics = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    plot_losses(train_losses, val_losses)
    for metric in ["MAE", "RMSE", "ACP", "Accuracy", "PercentageAccuracy"]:
        plot_metrics(train_metrics[metric], val_metrics[metric], metric)

    print("Training complete. Best model saved in 'Experiments/25/best_model.pth'")


if __name__ == "__main__":
    main()
