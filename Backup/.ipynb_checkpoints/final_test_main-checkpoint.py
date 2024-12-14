import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from model9 import CellCounter

# Global paths
MODEL_PATH = "Experiments/59/best_model.pth"  # Path to the best model
CSV_PATH = "Result/submission.csv"  # Path to the CSV file
IMAGE_FOLDER = "IDCIA_Test_Dataset/images/"  # Folder containing the images
OUTPUT_CSV_PATH = "Experiments/63/submission.csv"  # Output CSV file path

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img)

def preprocess_clusters(image):
    """
    Detect and separate clusters in the input image.
    Uses a combination of GaussianBlur and Watershed Algorithm.
    """
    if len(image.shape) == 2:  # Grayscale
        gray = image
    elif image.shape[-1] == 3:  # RGB
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    blurred_uint8 = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(blurred_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(binary, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = markers.astype(np.int32)

    watershed_input = np.stack([gray] * 3, axis=-1).astype(np.uint8)
    markers = cv2.watershed(watershed_input, markers)
    cluster_separated = np.zeros_like(gray)
    cluster_separated[markers > 1] = 255
    return np.stack([cluster_separated] * 3, axis=-1)

# Load model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCounter().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

# Predict cell count and update CSV
def predict_and_update_csv(model, device):
    # Load the CSV file
    df = pd.read_csv(CSV_PATH)
    if 'prediction' not in df.columns:
        df['prediction'] = None  # Add 'prediction' column if it doesn't exist

    for idx, row in df.iterrows():
        image_name = row['filename']
        image_path = os.path.join(IMAGE_FOLDER, image_name)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping...")
            df.at[idx, 'prediction'] = 'Image Not Found'
            continue

        # Preprocess raw and cluster-separated inputs
        raw_img_tensor = preprocess_image(image_path).unsqueeze(0).to(device)
        cluster_img_tensor = torch.tensor(
            preprocess_clusters(
                np.array(Image.open(image_path).convert('RGB'))
            ).transpose(2, 0, 1)  # Convert HWC to CHW
        ).unsqueeze(0).float().to(device)

        # Predict
        with torch.no_grad():
            predicted_count, _, _ = model(raw_img_tensor, cluster_img_tensor)
            predicted_count = int(round(predicted_count.item()))
            df.at[idx, 'prediction'] = predicted_count
            # print(f"Processed {image_name}: Predicted count = {predicted_count}")

    # Save the updated CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Updated CSV saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    # Load the model
    model, device = load_model()

    # Predict and update CSV
    predict_and_update_csv(model, device)
