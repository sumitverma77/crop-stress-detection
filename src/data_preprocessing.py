# data_preprocessing.py
# Script for data loading and preprocessing

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_and_preprocess_data(train_data_dir, val_data_dir, image_size, batch_size):
    """
    Loads and preprocesses image data using ImageDataGenerator.
    Assumes data is organized in subdirectories by class (e.g., data_dir/class1/, data_dir/class2/).
    """
    
    # Data augmentation and preprocessing for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Preprocessing for validation (only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Check if data_dir exists
    if not os.path.exists(train_data_dir):
        print(f"Error: Training data directory '{train_data_dir}' not found.")
        print("Please ensure your image data is organized in subdirectories within this path,")
        print("e.g., data/images/Dataset/train/Drought, data/images/Dataset/train/Healthy.")
        exit()
    if not os.path.exists(val_data_dir):
        print(f"Error: Validation data directory '{val_data_dir}' not found.")
        print("Please ensure your image data is organized in subdirectories within this path,")
        print("e.g., data/images/Dataset/val/Drought, data/images/Dataset/val/Healthy.")
        exit()

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    print(f"Found {train_generator.num_classes} classes: {list(train_generator.class_indices.keys())}")
    print(f"Training images: {train_generator.samples}")
    print(f"Validation images: {validation_generator.samples}")

    return train_generator, validation_generator

if __name__ == "__main__":
    # Example usage (for testing purposes)
    # This part will not run when imported as a module
    print("Running data_preprocessing.py as a standalone script for testing...")
    
    # Create dummy data directories for testing
    dummy_base_dir = "temp_dummy_data_dataset"
    dummy_train_dir = os.path.join(dummy_base_dir, "train")
    dummy_val_dir = os.path.join(dummy_base_dir, "val")

    os.makedirs(os.path.join(dummy_train_dir, "class_a"), exist_ok=True)
    os.makedirs(os.path.join(dummy_train_dir, "class_b"), exist_ok=True)
    os.makedirs(os.path.join(dummy_val_dir, "class_a"), exist_ok=True)
    os.makedirs(os.path.join(dummy_val_dir, "class_b"), exist_ok=True)
    
    # Create dummy files (e.g., empty text files) to simulate images
    with open(os.path.join(dummy_train_dir, "class_a", "img1.txt"), "w") as f: f.write("dummy")
    with open(os.path.join(dummy_train_dir, "class_a", "img2.txt"), "w") as f: f.write("dummy")
    with open(os.path.join(dummy_train_dir, "class_b", "img3.txt"), "w") as f: f.write("dummy")
    with open(os.path.join(dummy_train_dir, "class_b", "img4.txt"), "w") as f: f.write("dummy")
    with open(os.path.join(dummy_val_dir, "class_a", "img5.txt"), "w") as f: f.write("dummy")
    with open(os.path.join(dummy_val_dir, "class_b", "img6.txt"), "w") as f: f.write("dummy")

    try:
        train_gen, val_gen = load_and_preprocess_data(dummy_train_dir, dummy_val_dir, (224, 224), 32)
        print("Data generators created successfully for dummy data.")
    except Exception as e:
        print(f"Error during dummy data processing: {e}")
    finally:
        # Clean up dummy data
        import shutil
        if os.path.exists(dummy_base_dir):
            shutil.rmtree(dummy_base_dir)
        print("Dummy data cleaned up.")
