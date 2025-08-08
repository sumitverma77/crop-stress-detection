# Models Directory

This directory is designated for storing the trained machine learning models for Crop Stress Detection.

**Contents:**

*   **`crop_stress_model.h5`**: This will be the primary trained model file saved by `train.py`. It will contain the model architecture, weights, and optimizer state.
*   **`best_model.h5`**: During training, an `EarlyStopping` and `ModelCheckpoint` callback is configured to save the model with the best validation loss to this file.

**Usage:**

*   After successfully running `src/train.py`, the trained model will be saved here.
*   The `src/predict.py` script will load models from this directory to make predictions.

**Note:**
This directory might initially be empty. Trained models will appear here after the training process is completed.
