# model_utils.py
# Utility functions for model creation, training, and evaluation

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_model(image_size, num_classes, learning_rate=0.0001):
    """
    Creates a transfer learning model based on MobileNetV2.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x) # Added Dropout layer
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Unfreeze the last 20 layers of the base model for fine-tuning
    for layer in base_model.layers[:-20]: # Freeze all but the last 20 layers
        layer.trainable = False
    for layer in base_model.layers[-20:]: # Unfreeze the last 20 layers
        layer.trainable = True
        
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model created and compiled successfully.")
    model.summary()
    return model

def train_model(model, train_generator, validation_generator, epochs):
    """
    Trains the given model using the provided data generators.
    """
    print("Starting model training...")
    
    # Callbacks for early stopping and model checkpointing
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), # Increased patience
        ModelCheckpoint('models/best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    print("Model training finished.")
    return history

def evaluate_model(model, test_generator):
    """
    Evaluates the trained model on the test/validation set.
    """
    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Model Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy

if __name__ == "__main__":
    # This part is for testing the functions independently
    print("Running model_utils.py as a standalone script for testing...")
    
    # Dummy parameters for testing
    dummy_image_size = (224, 224)
    dummy_num_classes = 3
    
    # Test create_model
    print("\n--- Testing create_model ---")
    dummy_model = create_model(dummy_image_size, dummy_num_classes)
    
    # Note: To fully test train_model and evaluate_model, you would need
    # actual data generators. This is just a placeholder for structure.
    print("\n--- To test train_model and evaluate_model, run train.py with actual data. ---")
