import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import json
import matplotlib.pyplot as plt

# Set parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 320)
# sftp://192.168.1.107/home/kits/Documents/sourabh/sorted_output/
DATA_DIR = "/home/kavyan/Documents/avcc_img"  # Update this to your dataset path                        
# sftp://192.168.1.107/home/kits/Documents/sourabh/final/
OUTPUT_DIR = "/home/kavyan/Documents/avcc_output2/"
EPOCHS = 100  # Increased epochs from 50 to 100

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading datasets...")

# Load datasets with data augmentation
train_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

val_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# Get class names
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Save class names
class_names_path = os.path.join(OUTPUT_DIR, "class_names.json")
with open(class_names_path, "w") as f:
    json.dump(class_names, f)
print(f"Class names saved to: {class_names_path}")

# Data augmentation for better generalization
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1)
])

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE

# Apply preprocessing and augmentation
def preprocess_train(x, y):
    x = data_augmentation(x)  # Apply augmentation
    x = preprocess_input(x)   # MobileNetV2 preprocessing
    return x, y

def preprocess_val(x, y):
    x = preprocess_input(x)   # Only preprocessing for validation
    return x, y

train_dataset = train_dataset.map(preprocess_train, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(preprocess_val, num_parallel_calls=AUTOTUNE)

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Load the MobileNetV2 base model
print("Creating model...")
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False  # Start with frozen base model

# Build model with better architecture
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile with initial settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),  # Increased from 0.001 to 0.003
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # Increased patience for more epochs
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More aggressive LR reduction
        patience=7,  # Slightly more patience
        min_lr=1e-8,  # Lower minimum LR
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, "best_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("Starting initial training (frozen base model)...")
# Phase 1: Train with frozen base model
history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

print("Starting fine-tuning (unfrozen base model)...")
# Phase 2: Fine-tuning - unfreeze some layers of base model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before fine_tune_at
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),  # Increased from 0.00001 to 0.0003
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training with fine-tuning
history2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    initial_epoch=len(history1.history['accuracy']),
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model_path = os.path.join(OUTPUT_DIR, "mobilenetv2_classifier_final.h5")
model.save(model_path)

print("\nTraining completed!")
print(f"Final training accuracy: {history2.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history2.history['val_accuracy'][-1]:.4f}")
print(f"Best model saved to: {os.path.join(OUTPUT_DIR, 'best_model.h5')}")
print(f"Final model saved to: {model_path}")
print(f"Class names saved to: {class_names_path}")

# Plot training history
def plot_training_history(history1, history2):
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.show()

# Plot and save training history
plot_training_history(history1, history2)
print(f"Training plots saved to: {os.path.join(OUTPUT_DIR, 'training_history.png')}")


