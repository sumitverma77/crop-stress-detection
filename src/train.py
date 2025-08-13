# train.py
# Main script for training the Crop Stress Detection model

import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_and_preprocess_data
from model_utils import create_model, train_model, evaluate_model

def main(args):
    # Define paths
    base_data_dir = args.data_dir
    train_data_dir = os.path.join(base_data_dir, "train")
    validation_data_dir = os.path.join(base_data_dir, "val")
    model_save_path = args.model_save_path
    
    # Load and preprocess data
    train_generator, validation_generator = load_and_preprocess_data(train_data_dir, validation_data_dir, args.image_size, args.batch_size)

    # Create and compile the model
    model = create_model(args.image_size, train_generator.num_classes, args.learning_rate)
    
    # Train the model
    history = train_model(model, train_generator, validation_generator, args.epochs)
    
    # Evaluate the model
    evaluate_model(model, validation_generator)
    
    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save training history to output/training_log.txt
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "training_log.txt")
    with open(history_path, "w") as f:
        f.write("Epoch,Loss,Accuracy,Val_Loss,Val_Accuracy\n")
        for epoch in range(len(history.history['loss'])):
            f.write(f"{epoch+1},"
                    f"{history.history['loss'][epoch]:.4f},"
                    f"{history.history['accuracy'][epoch]:.4f},"
                    f"{history.history['val_loss'][epoch]:.4f},"
                    f"{history.history['val_accuracy'][epoch]:.4f}\n")
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Crop Stress Detection model.")
    parser.add_argument("--data_dir", type=str, default="22BCE10679_SumitVerma_CropStressDetection/data/images/Dataset", help="Base directory containing train/val/test image data subdirectories.")
    parser.add_argument("--model_save_path", type=str, default="models/crop_stress_model.h5", help="Path to save the trained model.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Image size (height width).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    
    args = parser.parse_args()
    main(args)
