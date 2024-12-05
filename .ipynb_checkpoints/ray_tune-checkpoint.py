import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from glob import glob
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from dataset_handler import CellDataset
from model import CellCounter
from denseweight import DenseWeight
from train import preprocess_clusters


# Set up paths
TRAIN_PATH = "/u/mbanisharifdehkordi/Github/Cell Counting/IDCIA_Augmentated_V2/images/train"
VAL_PATH = "/u/mbanisharifdehkordi/Github/Cell Counting/IDCIA_Augmentated_V2/images/val"
STORAGE_PATH = "/u/mbanisharifdehkordi/Github/Cell Counting/ray_results"

# Validation function for dataset paths
def validate_dataset():
    train_paths = glob(os.path.join(TRAIN_PATH, "*.tiff"))
    val_paths = glob(os.path.join(VAL_PATH, "*.tiff"))

    if not train_paths:
        raise FileNotFoundError("No training images found in train directory!")
    if not val_paths:
        raise FileNotFoundError("No validation images found in val directory!")

    print(f"Found {len(train_paths)} training images and {len(val_paths)} validation images.")

# Custom collate function
def custom_collate_fn(batch):
    images, labels, cell_locations = zip(*batch)

    images = torch.stack(images)
    labels = torch.stack(labels)

    # Handle varying cell location sizes
    max_len = max(loc.size(0) for loc in cell_locations)
    padded_cell_locations = torch.full((len(cell_locations), max_len, 2), -1.0)
    for i, loc in enumerate(cell_locations):
        if loc.size(0) > 0:
            padded_cell_locations[i, :loc.size(0)] = loc

    return images, labels, padded_cell_locations

# DataLoader setup
def get_data_loaders(batch_size=16):
    train_image_paths = glob(os.path.join(TRAIN_PATH, "*.tiff"))
    val_image_paths = glob(os.path.join(VAL_PATH, "*.tiff"))

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Fixed Normalize
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Fixed Normalize
    ])
    train_dataset = CellDataset(train_image_paths, transform=train_transform)
    val_dataset = CellDataset(val_image_paths, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    return train_loader, val_loader


# Training function
from ray.air import session  # Import the session module from ray.air

def train_tune_model(config):
    # Initialize DataLoaders with given batch_size from config
    train_loader, val_loader = get_data_loaders(config["batch_size"])

    # Model, loss function, and optimizer
    model = CellCounter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    dw = DenseWeight(alpha=config["alpha"])

    # Training loop
    for epoch in range(10):  # Replace with the desired number of epochs
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1)

            # Generate cluster-separated images
            cluster_inputs = torch.stack([
                torch.tensor(
                    preprocess_clusters(img.cpu().permute(1, 2, 0).numpy()),  # Convert to HWC format for OpenCV
                    dtype=torch.float32
                ).permute(2, 0, 1)  # Convert back to CHW format for PyTorch
                for img in inputs
            ]).to(device)

            optimizer.zero_grad()

            # Forward pass
            cell_count, uncertainty, predicted_locations = model(inputs, cluster_inputs)
            loss = criterion(cell_count, labels).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
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
                loss = criterion(cell_count, labels).mean()
                val_loss += loss.item()

        # Report metrics to Ray Tune
        session.report({"val_loss": val_loss / len(val_loader), "train_loss": running_loss / len(train_loader)})


# Main Ray Tune function
def main():
    validate_dataset()

    # Define the scheduler with an increased max_t
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        grace_period=10,  # Let each trial run at least 10 iterations
        max_t=200         # Allow up to 200 iterations per trial
    )

    tuner = tune.run(
        train_tune_model,
        config={
            "batch_size": tune.choice([32, 64]),  # Focus on larger batch sizes
            "learning_rate": tune.loguniform(1e-4, 1e-3),  # Centered around 5e-4
            "weight_decay": tune.loguniform(1e-6, 1e-4),  # Centered around 1e-5
            "alpha": 1.0,  # Fixed value, not part of tuning
        },
        num_samples=15,  # Increase sample count for more exploration
        resources_per_trial={"cpu": 4, "gpu": 1},
        storage_path=STORAGE_PATH,
        scheduler=scheduler,  # Scheduler with max_t defined
    )


if __name__ == "__main__":
    main()
