import os
import shutil
import pandas as pd
from tqdm import tqdm

RAW_DIR = "raw_datasets"
PROCESSED_DIR = "data/processed"
IMAGE_DIR = os.path.join(PROCESSED_DIR, "images")
LABELS_CSV = os.path.join(PROCESSED_DIR, "labels.csv")


def ensure_dirs():
    os.makedirs(IMAGE_DIR, exist_ok=True)


def dataset_already_processed():
    """Check if processed images exist to avoid redundant processing."""
    return os.path.exists(LABELS_CSV) and len(os.listdir(IMAGE_DIR)) > 0


def process_handwriting_recognition():
    """Process labeled dataset from Handwriting-Recognition-Datasets."""
    dataset_path = os.path.join(RAW_DIR, "Handwriting-Recognition-Dataset")
    if not os.path.exists(dataset_path):
        print("📌 Handwriting-Recognition-Dataset not found. Skipping...")
        return []

    label_files = [
        ("train_v2/train/", "written_name_train_v2.csv"),
        ("validation_v2/validation/", "written_name_validation_v2.csv"),
        ("test_v2/test/", "written_name_test_v2.csv")
    ]
    
    data = []
    for folder, csv_file in label_files:
        csv_path = os.path.join(dataset_path, csv_file)
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        image_folder = os.path.join(dataset_path, folder)

        for _, row in tqdm(df.iterrows(), desc=f"Processing {csv_file}", total=len(df)):
            src_path = os.path.join(image_folder, row["FILENAME"])
            if os.path.exists(src_path):
                new_name = f"{row['IDENTITY']}_{row['FILENAME']}"
                dst_path = os.path.join(IMAGE_DIR, new_name)
                shutil.copy(src_path, dst_path)
                data.append((dst_path, row["IDENTITY"]))

    return data


def process_doctors_prescription():
    """Process the new Doctor’s Handwritten Prescription BD dataset."""
    dataset_path = os.path.join(RAW_DIR, "Doctor’s Handwritten Prescription BD dataset")
    if not os.path.exists(dataset_path):
        print("📌 Doctor’s Handwritten Prescription BD dataset not found. Skipping...")
        return []

    label_files = [
        ("Training/training_words/", "Training/training_labels.csv"),
        ("Validation/validation_words/", "Validation/validation_labels.csv"),
        ("Testing/testing_words/", "Testing/testing_labels.csv"),
    ]

    data = []
    for folder, csv_file in label_files:
        csv_path = os.path.join(dataset_path, csv_file)
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df.rename(columns={"IMAGE": "image_path", "GENERIC_NAME": "label"}, inplace=True)
        image_folder = os.path.join(dataset_path, folder)

        for _, row in tqdm(df.iterrows(), desc=f"Processing {csv_file}", total=len(df)):
            src_path = os.path.join(image_folder, f"{row['image_path']}")
            if os.path.exists(src_path):
                new_name = f"{row['label']}_{row['image_path']}"
                dst_path = os.path.join(IMAGE_DIR, new_name)
                shutil.copy(src_path, dst_path)
                data.append((dst_path, row["label"]))
            else:
                print(f"⚠️ Missing image: {src_path}")

    return data

def process_ocr_dataset():
    """Process character-based dataset from OCR_dataset."""
    dataset_path = os.path.join(RAW_DIR, "OCR_dataset")
    data = []

    for data_folder in ["data/training_data", "data/testing_data", "data2/training_data", "data2/testing_data"]:
        full_path = os.path.join(dataset_path, data_folder)
        if not os.path.exists(full_path):
            continue
        
        for char_folder in os.listdir(full_path):
            char_folder_path = os.path.join(full_path, char_folder)
            if os.path.isdir(char_folder_path):
                for img_name in os.listdir(char_folder_path):
                    src_path = os.path.join(char_folder_path, img_name)
                    if os.path.exists(src_path):
                        new_name = f"{char_folder}_{img_name}"
                        dst_path = os.path.join(IMAGE_DIR, new_name)
                        shutil.copy(src_path, dst_path)
                        data.append((dst_path, char_folder))
    
    return data

def main():
    ensure_dirs()

    # Check if we already processed datasets
    if dataset_already_processed():
        print("✅ Processed dataset already exists. Only adding new data...")
    else:
        print("🚀 No processed dataset found. Processing from raw datasets...")
        ocr_data = process_ocr_dataset()
        all_data = ocr_data
        df = pd.DataFrame(all_data, columns=["image_path", "label"])
        df.to_csv(LABELS_CSV, index=False)
        print(f"Dataset processing complete. {len(df)} images processed.")

    # # Process only new datasets
    # doctors_data = process_doctors_prescription()
    # if doctors_data:
    #     # Append new data to existing CSV
    #     df_existing = pd.read_csv(LABELS_CSV) if os.path.exists(LABELS_CSV) else pd.DataFrame(columns=["image_path", "label"])
    #     df_new = pd.DataFrame(doctors_data, columns=["image_path", "label"])
    #     df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    #     df_combined.to_csv(LABELS_CSV, index=False)
    #     print(f"✅ Added {len(doctors_data)} new images. Total: {len(df_combined)}")
    # else:
    #     print("📌 No new data found. Skipping CSV update.")

if __name__ == "__main__":
    main()
