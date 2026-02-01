#!/usr/bin/env python3
"""Create demo models for testing the Model Analyzer."""

import os
import sys

def create_demo_models():
    """Create simple demo models for testing."""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")

        # Create output directory
        demo_dir = "demo_models"
        os.makedirs(demo_dir, exist_ok=True)

        # Model 1: Simple ConvNet
        print("\nCreating ConvNet model...")
        model1 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model1)
        tflite_model1 = converter.convert()

        # Save TFLite model
        model1_path = os.path.join(demo_dir, "convnet.tflite")
        with open(model1_path, 'wb') as f:
            f.write(tflite_model1)
        print(f"Saved: {model1_path}")

        # Save as SavedModel for ONNX conversion
        saved_model_path = os.path.join(demo_dir, "convnet_saved")
        model1.save(saved_model_path)
        print(f"Saved: {saved_model_path}")

        # Model 2: Simple MLP
        print("\nCreating MLP model...")
        model2 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model2)
        tflite_model2 = converter.convert()

        # Save TFLite model
        model2_path = os.path.join(demo_dir, "mlp.tflite")
        with open(model2_path, 'wb') as f:
            f.write(tflite_model2)
        print(f"Saved: {model2_path}")

        print(f"\nDemo models created in: {demo_dir}")
        print("\nYou can now analyze them with:")
        print(f"  python run.py {demo_dir}")

    except ImportError:
        print("TensorFlow not installed. Skipping demo model creation.")
        print("Install with: pip install tensorflow")

    except Exception as e:
        print(f"Error creating demo models: {e}")


if __name__ == "__main__":
    create_demo_models()
