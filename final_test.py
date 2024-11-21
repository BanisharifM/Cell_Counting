import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from Backup.model5 import CellCounter

# Global paths
MODEL_PATH = "Experiments/36/best_model.pth"  # Path to the best model
CSV_PATH = "Result/submission.csv"  # Path to the CSV file
IMAGE_FOLDER = "IDCIA_Test_Dataset/images/"  # Folder containing the images
OUTPUT_CSV_PATH = "Result/updated_submission.csv"  # Output CSV file path

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # Add batch dimension

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCounter().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

def predict_and_update_csv(model, device):
    # Load the CSV file
    df = pd.read_csv(CSV_PATH)
    if 'prediction' not in df.columns:
        raise ValueError("The CSV file does not have a 'prediction' column.")

    for idx, row in df.iterrows():
        image_name = row['filename']
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping...")
            continue
        
        # Preprocess image
        img_tensor = preprocess_image(image_path).to(device)

        # Predict cell count
        with torch.no_grad():
            predicted_count, _ = model(img_tensor)
            df.at[idx, 'prediction'] = int(predicted_count.item())  # Update the prediction column

        print(f"Processed {image_name}: Predicted count = {int(predicted_count.item())}")

    # Save the updated CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Updated CSV saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    # Load model
    model, device = load_model()

    # Predict and update CSV
    predict_and_update_csv(model, device)
