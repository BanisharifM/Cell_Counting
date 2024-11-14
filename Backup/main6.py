import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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


def get_data_loaders(batch_size=8):

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, alpha=1.0, threshold=0.10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dw = DenseWeight(alpha=alpha)

    train_losses, val_losses = [], []
    train_metrics = {"MAE": [], "RMSE": [], "ACP": [], "Accuracy": []}
    val_metrics = {"MAE": [], "RMSE": [], "ACP": [], "Accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss, total_mae, total_rmse, total_acp, correct_cells = 0.0, 0.0, 0.0, 0.0, 0
        all_targets = [label.cpu().numpy() for _, label in train_loader]
        weights = dw.fit(np.array(all_targets).reshape(-1, 1))

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Loss and metrics
            loss = criterion(outputs, labels)
            weighted_loss = (loss * torch.tensor(weights[i]).to(device)).mean()
            weighted_loss.backward()
            optimizer.step()
            running_loss += weighted_loss.item()

            mae = torch.mean(torch.abs(outputs - labels)).item()
            rmse = torch.sqrt(torch.mean((outputs - labels) ** 2)).item()
            acp = torch.mean((torch.abs(outputs - labels) <= 0.05 * torch.abs(labels)).float()).item()

            # Threshold-based accuracy calculation
            predicted_cells = torch.round(outputs)
            actual_cells = torch.round(labels)
            lower_bound = actual_cells * (1 - threshold)
            upper_bound = actual_cells * (1 + threshold)
            correct_cells += torch.sum((predicted_cells >= lower_bound) & (predicted_cells <= upper_bound)).item()

            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_acp += acp * inputs.size(0)

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_metrics["MAE"].append(total_mae / len(train_loader.dataset))
        train_metrics["RMSE"].append(total_rmse / len(train_loader.dataset))
        train_metrics["ACP"].append(total_acp / len(train_loader.dataset))
        train_metrics["Accuracy"].append(correct_cells / len(train_loader.dataset) * 100)

        val_loss, val_mae, val_rmse, val_acp, val_accuracy = evaluate_model(model, val_loader, criterion, device, threshold)
        val_losses.append(val_loss)
        val_metrics["MAE"].append(val_mae)
        val_metrics["RMSE"].append(val_rmse)
        val_metrics["ACP"].append(val_acp)
        val_metrics["Accuracy"].append(val_accuracy)

        print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train MAE: {train_metrics['MAE'][-1]:.4f}, Val MAE: {val_metrics['MAE'][-1]:.4f}")
        print(f"Train RMSE: {train_metrics['RMSE'][-1]:.4f}, Val RMSE: {val_metrics['RMSE'][-1]:.4f}")
        print(f"Train ACP: {train_metrics['ACP'][-1]:.4f}, Val ACP: {val_metrics['ACP'][-1]:.4f}")
        print(f"Train Accuracy: {train_metrics['Accuracy'][-1]:.2f}%, Val Accuracy: {val_metrics['Accuracy'][-1]:.2f}%")

    return model, train_losses, val_losses, train_metrics, val_metrics



def evaluate_model(model, data_loader, criterion, device, threshold=0.05):
    model.eval()
    total_loss, total_mae, total_rmse, total_acp, correct_cells = 0.0, 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels).mean()
            total_loss += loss.item()

            mae = torch.mean(torch.abs(outputs - labels)).item()
            rmse = torch.sqrt(torch.mean((outputs - labels) ** 2)).item()
            acp = torch.mean((torch.abs(outputs - labels) <= 0.05 * torch.abs(labels)).float()).item()

            # Threshold-based accuracy calculation
            predicted_cells = torch.round(outputs)
            actual_cells = torch.round(labels)
            lower_bound = actual_cells * (1 - threshold)
            upper_bound = actual_cells * (1 + threshold)
            correct_cells += torch.sum((predicted_cells >= lower_bound) & (predicted_cells <= upper_bound)).item()

            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_acp += acp * inputs.size(0)

    accuracy = correct_cells / len(data_loader.dataset) * 100
    return total_loss / len(data_loader), total_mae / len(data_loader.dataset), total_rmse / len(data_loader.dataset), total_acp / len(data_loader.dataset), accuracy


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()


def plot_metrics(train_metrics, val_metrics, metric_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.savefig(f'{metric_name.lower()}_plot.png')
    plt.close()


def main():
    batch_size, num_epochs, learning_rate = 16, 100,  5e-5
    train_loader, val_loader = get_data_loaders(batch_size)
    model = CellCounter()

    trained_model, train_losses, val_losses, train_metrics, val_metrics = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    plot_losses(train_losses, val_losses)
    for metric in ["MAE", "RMSE", "ACP", "Accuracy"]:
        plot_metrics(train_metrics[metric], val_metrics[metric], metric)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    test_loss, test_mae, test_rmse, test_acp, test_accuracy = evaluate_model(trained_model, val_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test ACP: {test_acp:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    torch.save(trained_model.state_dict(), "cell_counter_resnet.pth")


if __name__ == "__main__":
    main()
