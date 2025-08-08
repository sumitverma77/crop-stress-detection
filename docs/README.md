# Crop Stress Detection Project - Documentation

## Project: 22BCE10679 SUMIT VERMA - Crop Stress Detection (Nutrient Deficiency, Drought)

This project aims to develop a machine learning model capable of detecting crop stress, specifically identifying nutrient deficiencies and drought conditions from images.

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Project Structure](#2-project-structure)
3.  [Setup Instructions](#3-setup-instructions)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
4.  [Dataset](#4-dataset)
5.  [Training the Model](#5-training-the-model)
6.  [Making Predictions](#6-making-predictions)
7.  [Evaluation and Results](#7-evaluation-and-results)
8.  [Future Work](#8-future-work)

---

### 1. Introduction

Brief overview of the problem, the importance of crop stress detection, and the approach taken in this project (e.g., transfer learning with CNNs).

### 2. Project Structure

```
22BCE10679_SumitVerma_CropStressDetection/
├── src/
│   ├── train.py                # Main script for training the model
│   ├── predict.py              # Script for making predictions
│   ├── data_preprocessing.py   # Handles data loading and augmentation
│   └── model_utils.py          # Utility functions for model creation, training, evaluation
├── data/
│   ├── images/                 # Placeholder for actual image data (subdirectories for classes)
│   ├── README.md               # Instructions for data acquisition and structure
│   ├── train.csv               # Optional: Training labels/metadata
│   └── test.csv                # Optional: Testing labels/metadata
├── models/
│   ├── README.md               # Explains contents of this directory
│   ├── crop_stress_model.h5    # Saved trained model
│   └── best_model.h5           # Best model saved during training (based on validation loss)
├── docs/
│   ├── README.md               # This documentation file
│   └── report.pdf              # Project report (placeholder)
├── output/
│   ├── README.md               # Explains contents of this directory
│   └── sample_predictions/     # Sample prediction images/results (placeholder)
├── requirements.txt            # List of Python dependencies
└── notebooks/                  # Optional: Jupyter notebooks for EDA, experimentation
    └── eda.ipynb               # Example: Exploratory Data Analysis notebook
```

### 3. Setup Instructions

#### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

#### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # If this project is part of a larger repository, adjust accordingly
    # git clone [repository_url]
    # cd 22BCE10679_SumitVerma_CropStressDetection
    ```
    (Assuming you are already in the `c:/Users/ASUS/Desktop/Sumit/vscode` directory and the project folder is created.)

2.  **Navigate to the project directory:**
    ```bash
    cd 22BCE10679_SumitVerma_CropStressDetection
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

4.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

5.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 4. Dataset

Refer to `data/README.md` for detailed instructions on how to obtain and structure the dataset. Ensure your data is correctly placed in `data/images/` with appropriate class subdirectories before proceeding with training or prediction.

### 5. Training the Model

To train the model, run the `train.py` script from the `src/` directory.

```bash
python src/train.py --data_dir data/images --model_save_path models/crop_stress_model.h5 --image_size 224 224 --batch_size 32 --epochs 10
```

*   `--data_dir`: Path to your image dataset.
*   `--model_save_path`: Path where the trained model will be saved.
*   `--image_size`: Target image dimensions (width height).
*   `--batch_size`: Number of samples per gradient update.
*   `--epochs`: Number of training epochs.

Training progress, including loss and accuracy for both training and validation sets, will be displayed in the console. The best model (based on validation loss) will be saved as `models/best_model.h5`.

### 6. Making Predictions

To make predictions on new images, use the `predict.py` script.

**Predicting a single image:**

```bash
python src/predict.py --model_path models/crop_stress_model.h5 --image_path path/to/your/image.jpg --image_size 224 224
```

**Predicting images in a directory:**

```bash
python src/predict.py --model_path models/crop_stress_model.h5 --image_path path/to/your/image_directory/ --image_size 224 224
```

*   `--model_path`: Path to the trained model file (e.g., `models/crop_stress_model.h5` or `models/best_model.h5`).
*   `--image_path`: Path to the image file or a directory containing images for prediction.
*   `--image_size`: Image dimensions used during model training.

The script will output the predicted class and confidence for each image.

### 7. Evaluation and Results

After training, the `train.py` script will automatically evaluate the model on the validation set. The performance metrics (loss and accuracy) will be printed to the console.

**Target Performance:** The model aims to achieve >80% accuracy on the validation set. For this image classification task, accuracy is the primary metric.

### 8. Future Work

*   **Expand Dataset:** Incorporate more diverse images covering various crop types, stress levels, and environmental conditions.
*   **Advanced Models:** Experiment with more complex CNN architectures (e.g., ResNet, EfficientNet) or ensemble methods.
*   **Localization:** Implement object detection techniques to not only classify stress but also pinpoint the affected areas in the image.
*   **Real-time Monitoring:** Develop a system for real-time or near real-time stress detection using drones or fixed cameras.
*   **Deployment:** Deploy the model as a web service or mobile application for practical use.
