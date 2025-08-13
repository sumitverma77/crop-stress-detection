# predict.py
# Script for making predictions using the trained Crop Stress Detection model

import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def load_image(img_path, image_size):
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    return img_array / 255.0 # Normalize

def main(args):
    # Load the trained model
    model = tf.keras.models.load_model(args.model_path)
    print(f"Model loaded from {args.model_path}")

    # Assuming the model was trained with ImageDataGenerator, get class names
    # This is a placeholder; in a real scenario, you'd save/load class names
    # The class names should match the order determined by ImageDataGenerator during training (alphabetical by directory name)
    class_names = ["Blast", "BLB", "healthy", "hispa", "leaf_spot"]

    if os.path.isdir(args.image_path):
        print(f"Processing images in directory: {args.image_path}")
        for img_name in os.listdir(args.image_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_img_path = os.path.join(args.image_path, img_name)
                processed_image = load_image(full_img_path, args.image_size)
                predictions = model.predict(processed_image)
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                predicted_class_name = class_names[predicted_class_index]
                print(f"Image: {img_name}, Predicted Class: {predicted_class_name} (Confidence: {predictions[0][predicted_class_index]:.2f})")
    else:
        # Load and preprocess the input image
        processed_image = load_image(args.image_path, args.image_size)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]

        print(f"Prediction for {args.image_path}:")
        print(f"Predicted Class: {predicted_class_name}")
        print(f"Confidence: {predictions[0][predicted_class_index]:.2f}")
        # You can also print all class probabilities if needed
        # for i, prob in enumerate(predictions[0]):
        #     print(f"  {class_names[i]}: {prob:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained Crop Stress Detection model.")
    parser.add_argument("--model_path", type=str, default="models/crop_stress_model.h5", help="Path to the trained model file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file or directory of images for prediction.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Image size (height width) used for model training.")
    
    args = parser.parse_args()
    main(args)
