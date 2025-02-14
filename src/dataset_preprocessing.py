import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Constants
IMG_HEIGHT = 32
IMG_WIDTH = 128
DATASET_DIR = "data/processed"
AUGMENTED_DIR = "data/augmented"

# Ensure output directory exists
os.makedirs(AUGMENTED_DIR, exist_ok=True)

# # Define Augmentations
# AUGMENTATION = iaa.Sequential([
#     iaa.Affine(rotate=(-5, 5)),  # Small random rotation
#     iaa.Affine(translate_px={"x": (-3, 3), "y": (-3, 3)}),  # Small shift
#     iaa.Multiply((0.8, 1.2)),  # Brightness adjustment
# ])

def preprocess_image(image_path):
    """Loads an image, converts to grayscale, resizes, and normalizes."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    # Check if the image was loaded properly
    if image is None:
        print(f"⚠️ Warning: Unable to read image {image_path}. Skipping...")
    else:
        # Resize only if the image is valid
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # Resize
        # image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # Resize
        image = image.astype(np.float32) / 255.0  # Normalize (0-1 range)
    return image

def augment_image(image):
    """Applies augmentation using TensorFlow operations."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert to tensor
    image = tf.expand_dims(image, axis=-1)  # Add channel dimension (from [H, W] to [H, W, 1])

    # Apply augmentations
    image = tf.image.random_brightness(image, max_delta=0.2)  # Random brightness
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Contrast variation
    image = tf.image.random_flip_left_right(image)  # Horizontal flip
    # image = tf.image.random_translation(image, translations=[3, 3])  # Small random shift
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=80, max_jpeg_quality=100)  # Compression artifacts

    return image.numpy()  # Convert back to NumPy

def process_dataset(csv_filename, output_csv):
    """Processes images listed in a CSV and saves them in the augmented directory."""
    df = pd.read_csv(os.path.join(DATASET_DIR, csv_filename))
    new_data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path, label = row["image_path"], row["label"]
        image = preprocess_image(image_path)

        if image is None:
            continue

        # Save preprocessed image
        new_path = os.path.join(AUGMENTED_DIR, os.path.basename(image_path))
        cv2.imwrite(new_path, (image * 255).astype(np.uint8))

        # Save augmented versions
        for i in range(3):  # 3 augmented versions per image
            aug_image = augment_image(image)
            aug_path = os.path.join(AUGMENTED_DIR, f"aug_{i}_" + os.path.basename(image_path))
            cv2.imwrite(aug_path, (aug_image * 255).astype(np.uint8))
            new_data.append([aug_path, label])

        new_data.append([new_path, label])

    # Save updated CSV
    df_new = pd.DataFrame(new_data, columns=["image_path", "label"])
    df_new.to_csv(os.path.join(AUGMENTED_DIR, output_csv), index=False)

    print(f"✅ Processed {csv_filename} → {output_csv} ({len(df_new)} images)")

if __name__ == "__main__":
    process_dataset("train.csv", "train_augmented.csv")
    process_dataset("val.csv", "val_augmented.csv")
    process_dataset("test.csv", "test_augmented.csv")
