"""
SIMPLE PREDICTION SCRIPT - Based on train.py
Classifies any random image using the trained MobileNetV2 model
"""

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import json
import os

# Configuration - matching train.py settings
MODEL_PATH = "/home/kavyan/Documents/avcc_output2/best_model.h5"  # From train.py OUTPUT_DIR
CLASS_NAMES_PATH = "/home/kavyan/Documents/avcc_output2/class_names.json"  # From train.py
IMG_SIZE = (224, 320)  # Matching train.py IMG_SIZE
FALLBACK_MODEL_PATH = "/home/kavyan/Documents/avcc_output2/mobilenetv2_classifier_final.h5"

def load_and_preprocess_image(image_path):
    """
    Load and preprocess image exactly like train.py
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array ready for prediction
    """
    try:
        # Load image using PIL
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"   Converted image from {image.mode} to RGB")
        
        # Resize to match training size (224, 320)
        image = image.resize((IMG_SIZE[1], IMG_SIZE[0]))  # PIL uses (width, height)
        print(f"   Resized image to: {IMG_SIZE[1]} x {IMG_SIZE[0]}")
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        # Apply MobileNetV2 preprocessing (same as train.py)
        image_array = preprocess_input(image_array)
        
        return image_array
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def load_model_and_classes():
    """
    Load the trained model and class names
    
    Returns:
        tuple: (model, class_names) or (None, None) if failed
    """
    # Try to load the best model first
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"⚠️  Best model not found at: {model_path}")
        print(f"   Trying fallback model...")
        model_path = FALLBACK_MODEL_PATH
        
    if not os.path.exists(model_path):
        print(f"❌ No model found!")
        print(f"   Please train a model first using: python3 train.py")
        return None, None
    
    try:
        # Load model
        print(f"📂 Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        
        # Load class names
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)
            print(f"✅ Loaded {len(class_names)} class names: {class_names}")
        else:
            print(f"⚠️  Class names file not found: {CLASS_NAMES_PATH}")
            # Create default class names based on model output
            num_classes = model.output_shape[-1]
            class_names = [f"class_{i}" for i in range(num_classes)]
            print(f"   Using default class names: {class_names}")
        
        return model, class_names
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def predict_image(image_path, show_top_n=3):
    """
    Predict the class of an image
    
    Args:
        image_path (str): Path to the image file
        show_top_n (int): Number of top predictions to show
        
    Returns:
        tuple: (predicted_class, confidence, all_predictions) or (None, 0, None)
    """
    print(f"\n🔍 ANALYZING IMAGE: {os.path.basename(image_path)}")
    print("="*60)
    
    # Load model and class names
    model, class_names = load_model_and_classes()
    if model is None:
        return None, 0, None
    
    # Load and preprocess image
    print(f"📷 Processing image...")
    processed_image = load_and_preprocess_image(image_path)
    if processed_image is None:
        return None, 0, None
    
    # Make prediction
    print(f"🧠 Making prediction...")
    try:
        predictions = model.predict(processed_image, verbose=0)
        probabilities = predictions[0]  # Get first (and only) prediction
        
        # Get top prediction
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx] * 100
        
        # Get top N predictions
        top_indices = np.argsort(probabilities)[::-1][:show_top_n]
        top_predictions = [(class_names[i], probabilities[i] * 100) for i in top_indices]
        
        return predicted_class, confidence, top_predictions
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None, 0, None

def main():
    """Main function to run prediction"""
    print("🚗 VEHICLE CLASSIFICATION PREDICTOR")
    print("   Based on train.py MobileNetV2 model")
    print("="*60)
    
    # Default image path - CHANGE THIS TO YOUR IMAGE
    # 23353MAV4_10-04-202509.01.49.jpg

    # /home/kavyan/Documents/avcc/z_backup_img/backup-image/mav5/
    # 82463Bus_15-05-202505.08.36.jpg
    default_image = "/home/kavyan/Documents/avcc/test/car/111LCV_22-04-202523.36.51.jpg"
    
    # Get image path from command line argument or use default
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"📁 Using provided image: {image_path}")
    else:
        image_path = default_image
        print(f"📁 Using default image: {image_path}")
        print(f"   To use different image: python3 predict2.py /path/to/your/image.jpg")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print(f"   Please provide a valid image path")
        return
    
    # Make prediction
    predicted_class, confidence, top_predictions = predict_image(image_path)
    
    # Display results
    if predicted_class is not None:
        print(f"\n🎯 PREDICTION RESULTS")
        print("="*60)
        print(f"📊 TOP PREDICTION:")
        print(f"   🚗 Vehicle Type: {predicted_class.upper()}")
        print(f"   📈 Confidence: {confidence:.1f}%")
        
        if top_predictions and len(top_predictions) > 1:
            print(f"\n📋 TOP {len(top_predictions)} PREDICTIONS:")
            for i, (class_name, prob) in enumerate(top_predictions, 1):
                print(f"   {i}. {class_name.upper()}: {prob:.1f}%")
        
        print("="*60)
        
        # Confidence assessment
        if confidence >= 80:
            print("✅ HIGH CONFIDENCE - Very reliable prediction!")
        elif confidence >= 60:
            print("🟨 MODERATE CONFIDENCE - Good prediction")
        elif confidence >= 40:
            print("⚠️  LOW CONFIDENCE - Prediction may be uncertain")
        else:
            print("❌ VERY LOW CONFIDENCE - Model may need retraining")
            
    else:
        print(f"\n❌ PREDICTION FAILED")
        print("   Please check your model and image files")

if __name__ == "__main__":
    main()
