# # evaluate.py
# # Script for evaluating the trained Crop Stress Detection model

# import os
# import argparse
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# def plot_confusion_matrix(cm, class_names, output_path):
#     """
#     Plots and saves the confusion matrix as a heatmap.
    
#     Args:
#         cm (np.array): The confusion matrix.
#         class_names (list): A list of class names for the labels.
#         output_path (str): Path to save the plot image.
#     """
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Confusion matrix plot saved to {output_path}")

# def main(args):
#     # Define paths and parameters
#     model_path = args.model_path
#     test_data_dir = args.test_data_dir
#     image_size = tuple(args.image_size)
#     batch_size = args.batch_size
#     output_dir = "output"
#     os.makedirs(output_dir, exist_ok=True)

#     # --- 1. Load the Trained Model ---
#     print(f"Loading model from {model_path}...")
#     try:
#         model = tf.keras.models.load_model(model_path)
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return

#     # --- 2. Prepare the Test Data Generator ---
#     # For evaluation, we only need to rescale the images. No augmentation is needed.
#     test_datagen = ImageDataGenerator(rescale=1./255)
    
#     print(f"Loading test data from {test_data_dir}...")
#     test_generator = test_datagen.flow_from_directory(
#         test_data_dir,
#         target_size=image_size,
#         batch_size=batch_size,
#         class_mode='categorical',
#         shuffle=False  # IMPORTANT: Keep shuffle=False for correct label ordering
#     )
#     print(f"Found {test_generator.n} images belonging to {test_generator.num_classes} classes.")

#     # --- 3. Evaluate the Model ---
#     print("\nEvaluating model on the test set...")
#     loss, accuracy = model.evaluate(test_generator)
#     print(f"Test Loss: {loss:.4f}")
#     print(f"Test Accuracy: {accuracy:.4f}")

#     # --- 4. Generate Predictions for Detailed Metrics ---
#     print("Generating predictions for classification report and confusion matrix...")
#     predictions = model.predict(test_generator)
#     y_pred = np.argmax(predictions, axis=1)
#     y_true = test_generator.classes
#     class_names = list(test_generator.class_indices.keys())

#     # --- 5. Generate and Save Reports ---
#     # Classification Report
#     report = classification_report(y_true, y_pred, target_names=class_names)
#     print("\nClassification Report:")
#     print(report)

#     # Confusion Matrix
#     cm = confusion_matrix(y_true, y_pred)
    
#     # Save combined report to a text file
#     report_path = os.path.join(output_dir, "evaluation_report.txt")
#     with open(report_path, "w") as f:
#         f.write("="*30 + "\n")
#         f.write("  Model Evaluation Report\n")
#         f.write("="*30 + "\n\n")
#         f.write(f"Test Loss: {loss:.4f}\n")
#         f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
#         f.write("--- Classification Report ---\n")
#         f.write(report)
#         f.write("\n\n--- Confusion Matrix ---\n")
#         f.write(np.array2str(cm))
#     print(f"\nFull evaluation report saved to {report_path}")

#     # Plot and save the confusion matrix
#     cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
#     plot_confusion_matrix(cm, class_names, cm_plot_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate a trained Crop Stress Detection model.")
    
#     # Define command-line arguments with defaults matching your training script
#     parser.add_argument("--model_path", type=str, default="models/crop_stress_model.h5", 
#                         help="Path to the trained .h5 model file.")
#     parser.add_argument("--test_data_dir", type=str, default="22BCE10679_SumitVerma_CropStressDetection/data/images/Dataset/test", 
#                         help="Directory containing the test image data.")
#     parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], 
#                         help="Image size (height width) used during training.")
#     parser.add_argument("--batch_size", type=int, default=32, 
#                         help="Batch size for evaluation.")
    
#     args = parser.parse_args()
#     main(args)
# evaluate.py (Improved with Visual Predictions)
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # <-- NEW: Import OpenCV
import random
import config # <-- Use the central config

# --- This function is unchanged ---
def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix plot saved to {output_path}")

# --- NEW: Function to generate the prediction collage ---
def generate_prediction_collage(test_generator, y_true, y_pred, class_names, num_images=16):
    """Creates and saves a collage of test images with their actual and predicted labels."""
    # Get the list of all image file paths from the generator
    filepaths = [os.path.join(test_generator.directory, f) for f in test_generator.filenames]
    
    # Randomly select indices for the images to display
    sample_indices = random.sample(range(len(filepaths)), min(num_images, len(filepaths)))
    
    # Setup plot
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols
    plt.figure(figsize=(15, 4 * num_rows))

    for i, idx in enumerate(sample_indices):
        plt.subplot(num_rows, num_cols, i + 1)
        
        # Load and display the original image
        img = cv2.imread(filepaths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')

        # Get true and predicted labels
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        
        # Set title color based on correctness
        title_color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f"Actual: {true_label}\nPredicted: {pred_label}", color=title_color)

    plt.tight_layout()
    collage_path = os.path.join(config.OUTPUT_DIR, "prediction_examples.png")
    plt.savefig(collage_path)
    plt.close()
    print(f"Prediction examples collage saved to {collage_path}")


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- Load Model and Data ---
    print(f"Loading model from {config.MODEL_PATH}...")
    model = tf.keras.models.load_model(config.MODEL_PATH)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    print(f"Found {test_generator.n} images belonging to {test_generator.num_classes} classes.")

    # --- Evaluate and Predict ---
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    # --- Generate Outputs ---
    # 1. Classification Report and Confusion Matrix Plot
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n", report)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))

    # 2. NEW: Visual Prediction Collage
    generate_prediction_collage(test_generator, y_true, y_pred, class_names)

if __name__ == "__main__":
    main()