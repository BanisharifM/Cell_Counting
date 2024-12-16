# Automated Cell Counting for Stem Cell Differentiation Analysis

This repository contains the complete pipeline for **automated cell counting** using microscopic images. The project leverages **machine learning techniques** and advanced training methodologies to improve performance on the **IDCIA v2 dataset**.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dataset](#dataset)
3. [Data Processing](#data-processing)
4. [Model](#model)
5. [Evaluation](#evaluation)
6. [Streamlit Application](#streamlit-application)
7. [Libraries and Requirements](#libraries-and-requirements)
8. [Submission Format](#submission-format)
9. [Reminders](#reminders)

---

## Dataset

The dataset used is the **IDCIA v2** dataset, which contains:
- **250 fluorescent microscopic images** of Adult Hippocampal Progenitor Cells (AHPCs).
- Corresponding **ground truth CSV files** containing `x,y` coordinates for each cell.

### Example Ground Truth CSV:
```plaintext
X,Y
100,200
300,400
...
```

- Each image is resized to **256x256 pixels** during preprocessing.
- Data is split into **80/10/10** for training, validation, and test subsets.

---

## Data Processing

### 1. **Data Exploration**
*Written by Gretta Buttelmann*

Data exploration insights are available in the **Jupyter Notebook** `cellcount_dataset.ipynb`. It is optional and not required for running the model.

---

### 2. **Dataset Split (80/10/10)**
*Written by Gretta Buttelmann*

Run the script to split the dataset into train, validation, and test sets.

#### Usage:
```bash
# General usage
python split_data.py '<path_to_data_directory>' '<path_to_csv>'

# Example
python split_data.py '/path/to/data/IDCIA_v2' '/path/to/data/cell_count_datasets.csv'
```

This script will create **train/val/test directories** in both the **images** and **ground truth folders**.

---

### 3. **Data Augmentation**
*Written by Kojo Adu-Gyamfi*

To augment the training dataset, run the following script. The script performs **brightness, noise, rotation (90°, 180°, 270°), and flipping** augmentations.

#### Usage:
Edit file paths at the top of the script to match your directories:
```bash
python data_preprocessing.py
```

Output: A new directory with augmented training images.

---

## Model

The proposed model leverages a **ResNet-50 architecture**, replacing the final fully connected layer with a **regression node** to predict the cell count.

### Key Features:
1. **Transfer Learning**: Pre-trained weights from the ImageNet-1k dataset.
2. **Custom Loss Function**: Combines count loss (MAE) and location loss (MSE).
3. **Advanced Training Techniques**:
   - Adaptive learning rate scheduler (ReduceLROnPlateau)
   - Early stopping
   - Batch size adjustment
   - Cluster input transformation for dense regions.

---

## Evaluation

The model's performance is evaluated using the following metrics:
1. **Mean Absolute Error (MAE)**
2. **Root Mean Square Error (RMSE)**
3. **Acceptable Count Percent (ACP)**: Allows error thresholds based on cell count.
4. **Percent Accuracy**: Average accuracy across images.

---

## Streamlit Application

A **Streamlit app** is provided for real-time cell count predictions.

### Usage:
Run the Streamlit app using the following command:
```bash
streamlit run src/main_cell_counter.py
```

### Features:
1. Upload **TIFF** images.
2. Predict cell counts for selected images.
3. View a downloadable summary table in **Excel** format.

**Example Output Screenshots**:
1. **Starting Page**  
![streamlit_start](https://github.com/user-attachments/assets/13e9f547-07c5-4f8a-b72b-23cec3e572bf)

2. **Image Upload**  
![streamlit_upload](https://github.com/user-attachments/assets/ff104d99-0d35-471f-86b1-81084737d04a)

3. **Predicted Results**  
![streamlit_output](https://github.com/user-attachments/assets/d7135bba-da41-4d4d-9d01-a454884b50ff)

---

## Libraries and Requirements

### Core Dependencies
The project requires the following libraries. Install them using `pip`:
```bash
pip install -r requirements.txt
```

### `requirements.txt`:
```plaintext
numpy==1.24.3
matplotlib==3.8.0
tqdm==4.66.1
denseweight==0.1.2
pillow==10.0.0
scikit-learn==1.3.0
scipy==1.9.3
h5py==3.9.0
pandas==2.1.0
fsspec==2022.5.0
blosc2==2.0.0
cython>=0.29.21
FuzzyTM>=0.4.0
google-auth-oauthlib==0.5.0
tensorboard-data-server==0.7.0
protobuf==3.20.3
urllib3==1.26.15
```

---

## Submission Format

Submit predictions in a CSV file with the following format:
```plaintext
Image,Cell Count
image_001.tiff,10
image_002.tiff,20
...
```

The `Image` column contains the file name, and the `Cell Count` column contains the predicted cell count.

---

## Reminders

1. The dataset is provided **exclusively** for this project. Redistribution or use for other purposes is not allowed.
2. Ensure your submission CSV matches the format specified above.

---

## Acknowledgments
This project is part of **COM S 571X: Machine Learning for Vision** at Iowa State University. Developed by:
- Mahdi BanisharifDehkordi
- Gretta Buttelmann
- Faezeh Rajabi Kouchi
- Kojo Adu-Gyamfi


---
