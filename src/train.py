import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Bidirectional, Reshape, Attention
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import model
from sklearn.preprocessing import LabelEncoder
import pickle

# **1️⃣ Paths & Constants**
DATA_DIR = "data/augmented"  # Change to "data/processed" if needed
# IMAGE_DIR = DATA_DIR
TRAIN_CSV = os.path.join(DATA_DIR, "train_augmented.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_augmented.csv")

IMG_WIDTH, IMG_HEIGHT = 128, 32
BATCH_SIZE = 32 # change this to reduce memory usage (the ideal value is 32)
EPOCHS = 50
AUTOTUNE = tf.data.experimental.AUTOTUNE
CHECKPOINT_PATH = "models/best_model.h5"
NEW_MODEL_PATH = "models/bandungbondowoso.keras"
NUM_CLASSES = 64

# **2️⃣ Data Loading Functions**
def load_data(csv_path):
    print("Loading data")
    df = pd.read_csv(csv_path)
    image_paths = df["image_path"].values  # Use paths directly from CSV
    labels = df["label"].astype(str).values  # Convert labels to strings

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)  # Convert labels to indices
    # Save the label encoder
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    num_classes = len(label_encoder.classes_)  # Get actual class count

    print(f"Max encoded label: {max(labels)}")
    print(f"Min encoded label: {min(labels)}")
    print(f"Number of unique encoded labels: {len(set(labels))}")
    print(f"Computed NUM_CLASSES: {num_classes}")

    return image_paths, labels, num_classes  # Return num_classes too!


def preprocess_image(image_path, label):
    """Read, decode, resize, and normalize images."""
    print("Processing data")
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # Convert to grayscale
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH)) / 255.0  # Normalize
    return image, label

# def create_tf_dataset(csv_path, batch_size):
#     print("Creating tf dataset")
#     """Efficiently create a dataset without loading everything into memory."""
#     def parse_csv_line(line):
#         parts = tf.strings.split(line, ",")  # Adjust if separator is different
#         image_path = parts[0]
#         label = tf.strings.to_number(parts[1], out_type=tf.int32)  # Convert label to integer
#         return image_path, label

#     dataset = tf.data.TextLineDataset(csv_path).skip(1)  # Skip header
#     dataset = dataset.map(parse_csv_line, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
#     dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

#     return dataset


def create_tf_dataset(csv_path, batch_size):
    print("Creating tf dataset")
    """Create a TensorFlow dataset from a CSV file."""
    image_paths, labels, _ = load_data(csv_path)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

if __name__ == "__main__":
    # Load datasets
    
    print("Loading Train Dataset")
    train_image_paths, train_labels, NUM_CLASSES = load_data(TRAIN_CSV)
    print()

    print("Loading Validation Dataset")
    val_image_paths, val_labels, _ = load_data(VAL_CSV)  # We only need labels here

    print("Creating training dataset")
    train_dataset = create_tf_dataset(TRAIN_CSV, BATCH_SIZE)
    print("")
    print("Creating validation dataset")
    val_dataset = create_tf_dataset(VAL_CSV, BATCH_SIZE)
    
    
    # **Load Existing Model or Create New**
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading existing model from {CHECKPOINT_PATH}...")
        ocr_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    else:
        print("No previous checkpoint found. Creating a new model...")
        ocr_model = model.build_ocr_model(NUM_CLASSES)
        ocr_model.summary()
        
    # **Compile the Model (Ensure it's compiled after loading)**
    ocr_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # **Define Checkpoints**
    checkpoint = tf.keras.callbacks.ModelCheckpoint(NEW_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
    # **Continue Training**
    history = ocr_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint]
        )
        # **Save Final Model in `.keras` Format**
    ocr_model.save(NEW_MODEL_PATH)
    print(f"✅ Training complete! Model saved as {NEW_MODEL_PATH}")

    # # **4️⃣ Train the Model**
    # # Checkpoint to save best model
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor="val_loss", save_best_only=True, verbose=1)

    # print("Training the model")
    # history = ocr_model.fit(
    #     train_dataset,
    #     validation_data=val_dataset,
    #     epochs=EPOCHS,
    #     callbacks=[checkpoint]
    # )
    # # **5️⃣ Save Final Model**
    # model.save("models/final_handwriting_model.h5")
    # print("✅ Training complete! Model saved.")
