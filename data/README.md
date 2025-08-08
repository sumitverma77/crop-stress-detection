# Data Directory

This directory is intended to store the image datasets used for training and evaluating the Crop Stress Detection model.

**Important Note on Data Size:**
The full dataset for crop stress detection (including images for nutrient deficiency and drought) can be very large. Due to size constraints, the actual image files are not included directly in this repository.

**To access the dataset:**

Please follow these instructions to download or access the required dataset:

[**INSERT DATASET DOWNLOAD/ACCESS INSTRUCTIONS HERE**]

**Expected Structure:**

Once downloaded and extracted, please ensure your data is organized into subdirectories within the `data/images/` folder, with each subdirectory representing a specific class (e.g., 'Drought', 'Healthy', 'Nutrient_Deficiency').

Example structure:

```
data/
├── images/
│   ├── Drought/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── Healthy/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── Nutrient_Deficiency/
│       ├── image_001.jpg
│       ├── image_002.jpg
│       └── ...
├── train.csv (Optional: if labels are in a CSV, otherwise inferred from directory names)
└── test.csv (Optional: if labels are in a CSV)
```

**Placeholder Files:**

*   `data/images/`: This directory is a placeholder. You will place your downloaded image data here.
*   `data/train.csv`: Placeholder for training metadata/labels (if applicable).
*   `data/test.csv`: Placeholder for testing metadata/labels (if applicable).

**Before running `train.py` or `predict.py`, ensure the dataset is correctly placed and structured as described above.**
