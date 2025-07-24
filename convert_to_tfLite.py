import tensorflow as tf
import numpy as np
import os
import json

# Configuration - match your train.py settings
IMG_SIZE = (224, 320)  # From train.py
OUTPUT_DIR = "/home/kavyan/Documents/avcc_output2/"  # Updated to correct path

# Try to find the correct model file (matching train.py outputs)
model_candidates = [
    "best_model.h5",  # From train.py ModelCheckpoint
    "mobilenetv2_classifier_final.h5",  # From train.py final save
    "best_efficientnetb3_model.h5",  # Fallback
    "efficientnetb3_vehicle_classifier_final.h5",  # Fallback
    "best_mobilenetv2_model.h5",  # Fallback
    "mobilenetv2_vehicle_classifier_final.h5"  # Fallback
]

model_path = None
for candidate in model_candidates:
    full_path = os.path.join(OUTPUT_DIR, candidate)
    if os.path.exists(full_path):
        model_path = full_path
        model_name = candidate
        break

if model_path is None:
    print("❌ No trained model found!")
    print("Available models should be one of:")
    for candidate in model_candidates:
        print(f"   • {candidate}")
    exit(1)

print(f"🔍 Found model: {model_name}")
print(f"📂 Loading model from: {model_path}")

# Load your trained model
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Original model loaded successfully!")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Model output shape: {model.output_shape}")
    print(f"   Total parameters: {model.count_params():,}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Convert to TensorFlow Lite
print("\n🔄 Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimizations for smaller size and faster inference
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set representative dataset for quantization using correct input shape
def representative_data_gen():
    """Generate sample data matching your model's input shape"""
    print("   Generating representative dataset for quantization...")
    for _ in range(100):
        # Use correct input shape from your model
        input_shape = model.input_shape[1:]  # Remove batch dimension
        data = np.random.rand(1, *input_shape).astype(np.float32)
        # Apply preprocessing similar to your training
        data = data / 255.0  # Normalize to [0,1]
        yield [data]

# Optional: Uncomment for INT8 quantization (smaller model, faster inference)
# This will make the model much smaller but slightly less accurate
# converter.representative_dataset = representative_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

# Convert the model
print("🚀 Starting conversion...")
try:
    tflite_model = converter.convert()
    print("✅ Model conversion successful!")
    
    # Generate output filename based on input model
    base_name = os.path.splitext(model_name)[0]
    tflite_model_path = os.path.join(OUTPUT_DIR, f'{base_name}.tflite')
    
    # Save the TFLite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"💾 TensorFlow Lite model saved as: {tflite_model_path}")
    
    # Check model sizes
    original_size = os.path.getsize(model_path)
    tflite_size = os.path.getsize(tflite_model_path)
    
    print(f"\n📊 MODEL SIZE COMPARISON:")
    print(f"   Original model (.h5): {original_size / (1024*1024):.2f} MB")
    print(f"   TFLite model (.tflite): {tflite_size / (1024*1024):.2f} MB")
    print(f"   Size reduction: {((original_size - tflite_size) / original_size) * 100:.1f}%")
    
    # Load class names for reference
    class_names_path = os.path.join(OUTPUT_DIR, "class_names.json")
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        print(f"\n📋 MODEL INFO:")
        print(f"   Classes: {len(class_names)}")
        print(f"   Class names: {class_names}")
        
        # Save model info for later use
        model_info = {
            "model_path": tflite_model_path,
            "input_shape": list(model.input_shape[1:]),  # Remove batch dimension
            "output_shape": list(model.output_shape[1:]),
            "num_classes": len(class_names),
            "class_names": class_names,
            "preprocessing": "mobilenet_v2" if "mobilenet" in model_name.lower() else "efficientnet"
        }
        
        info_path = os.path.join(OUTPUT_DIR, f'{base_name}_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"   Model info saved: {info_path}")
    
    print(f"\n🎉 CONVERSION COMPLETED SUCCESSFULLY!")
    print(f"   Your TFLite model is ready for mobile deployment!")
    
except Exception as e:
    print(f"❌ Error during conversion: {e}")
    print(f"   This might be due to unsupported operations in your model.")
    print(f"   Try enabling INT8 quantization or check model compatibility.")