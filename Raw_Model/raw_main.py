'''
File: main.py
Author: Abdurahman Mohammed
Date: 2024-09-05
Description: A Python script that trains a cell counting model using the IDCIA dataset.

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from raw_dataset_handler import CellDataset
from raw_model import CellCounter


def get_data_loaders(batch_size=8):
    '''
    Creates training and validation data loaders for the IDCIA dataset.

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        train_loader (DataLoader): A DataLoader object for the training set.
        val_loader (DataLoader): A DataLoader object for the validation set.
    '''
    # Get training and testing image paths
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
    def custom_collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        cell_locations = [torch.empty(0, 2) for _ in range(len(labels))]  # Dummy locations for raw model
        return images, labels, cell_locations

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    return train_loader, val_loader


def calculate_metrics(pred_count, true_count):
    """
    Calculate metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
    and Percentage Accuracy.
    """
    mae = torch.abs(pred_count - true_count).mean().item()
    rmse = torch.sqrt(torch.mean((pred_count - true_count) ** 2)).item()
    percentage_accuracy = (1 - (torch.abs(pred_count - true_count) / true_count)).clamp(0, 1).mean().item() * 100
    return mae, rmse, percentage_accuracy


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3):
    '''
    Trains the cell counting model on the training set and evaluates it on the validation set.

    Args:
        model (CellCounter): The model to train.
        train_loader (DataLoader): The DataLoader for the training set.
        val_loader (DataLoader): The DataLoader for the validation set.
        num_epochs (int): The number of epochs to train the model.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        model (CellCounter): The trained model.
        train_losses (list): A list of training losses for each epoch.
        val_losses (list): A list of validation losses for each epoch.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_metrics = {"MAE": [], "RMSE": [], "PercentageAccuracy": []}
    val_metrics = {"MAE": [], "RMSE": [], "PercentageAccuracy": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_mae, total_rmse, total_percentage_accuracy = 0.0, 0.0, 0.0

        for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Metrics calculations
            mae, rmse, percentage_accuracy = calculate_metrics(outputs, labels)
            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_percentage_accuracy += percentage_accuracy * inputs.size(0)

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_metrics["MAE"].append(total_mae / len(train_loader.dataset))
        train_metrics["RMSE"].append(total_rmse / len(train_loader.dataset))
        train_metrics["PercentageAccuracy"].append(total_percentage_accuracy / len(train_loader.dataset))
        print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}")
        print(f"Train MAE: {train_metrics['MAE'][-1]:.4f}, Train RMSE: {train_metrics['RMSE'][-1]:.4f}, "
              f"Train Percentage Accuracy: {train_metrics['PercentageAccuracy'][-1]:.2f}%")

        # Evaluate the model on the validation set
        val_loss, val_mae, val_rmse, val_percentage_accuracy = evaluate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_metrics["MAE"].append(val_mae)
        val_metrics["RMSE"].append(val_rmse)
        val_metrics["PercentageAccuracy"].append(val_percentage_accuracy)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, "
              f"Val Percentage Accuracy: {val_percentage_accuracy:.2f}%")

    return model, train_losses, val_losses, train_metrics, val_metrics


def evaluate_model(model, data_loader, criterion, device):
    '''
    Evaluates the model on a validation or test set.

    Args:
        model (CellCounter): The model to evaluate.
        data_loader (DataLoader): The DataLoader for the validation or test set.
        criterion (torch.nn.Module): The loss function to use.
        device (torch.device): The device to run the evaluation on.

    Returns:
        total_loss (float): The average loss over the dataset.
        mae, rmse, percentage_accuracy (float): The calculated metrics.
    '''
    model.eval()
    total_loss = 0.0
    total_mae, total_rmse, total_percentage_accuracy = 0.0, 0.0, 0.0

    with torch.no_grad():
        for inputs, labels, _ in data_loader:  # Ignore `cell_locations` since the raw model doesn't use it
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Metrics calculations
            mae = torch.abs(outputs - labels).mean().item()
            rmse = torch.sqrt(torch.mean((outputs - labels) ** 2)).item()
            percentage_accuracy = (
                (1 - (torch.abs(outputs - labels) / labels))
                .clamp(0, 1)
                .mean()
                .item()
                * 100
            )
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
    '''
    Plots the training and validation losses.

    Args:
        train_losses (list): A list of training losses for each epoch.
        val_losses (list): A list of validation losses for each epoch.
    '''

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

def main():
    # Set hyperparameters
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-3

    # Get data loaders
    train_loader, val_loader = get_data_loaders(batch_size)

    # Create the model
    model = CellCounter()

    # Train the model
    trained_model, train_losses, val_losses, train_metrics, val_metrics = train_model(
        model, train_loader, val_loader, num_epochs, learning_rate
    )

    # Print final metrics for clarity
    for epoch in range(num_epochs):
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_losses[epoch]:.4f}, "
            f"Val Loss = {val_losses[epoch]:.4f}, "
            f"Train MAE = {train_metrics['MAE'][epoch]:.4f}, "
            f"Val MAE = {val_metrics['MAE'][epoch]:.4f}, "
            f"Train RMSE = {train_metrics['RMSE'][epoch]:.4f}, "
            f"Val RMSE = {val_metrics['RMSE'][epoch]:.4f}, "
            f"Train Percentage Accuracy = {train_metrics['PercentageAccuracy'][epoch]:.2f}%, "
            f"Val Percentage Accuracy = {val_metrics['PercentageAccuracy'][epoch]:.2f}%"
        )

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses)

    # Print summary of best validation metrics
    best_epoch = val_losses.index(min(val_losses))
    print("\nBest Epoch:")
    print(
        f"Epoch {best_epoch + 1}: "
        f"Val Loss = {val_losses[best_epoch]:.4f}, "
        f"Val MAE = {val_metrics['MAE'][best_epoch]:.4f}, "
        f"Val RMSE = {val_metrics['RMSE'][best_epoch]:.4f}, "
        f"Val Percentage Accuracy = {val_metrics['PercentageAccuracy'][best_epoch]:.2f}%"
    )

    # Save the model
    torch.save(trained_model.state_dict(), "Raw_Model/Raw_Result/cell_counter.pth")
    print("Training complete. Model saved to 'Raw_Result/cell_counter.pth'")

if __name__ == "__main__":
    main()