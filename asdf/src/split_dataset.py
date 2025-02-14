import pandas as pd
import os
from sklearn.model_selection import train_test_split

PROCESSED_DIR = "data/processed"
LABELS_CSV = os.path.join(PROCESSED_DIR, "labels.csv")

def check_dataset():
    df = pd.read_csv(LABELS_CSV)
    # Count occurrences of each label
    label_counts = df["label"].value_counts()
    # Print classes that have fewer than 5 samples
    rare_classes = label_counts[label_counts < 5]
    print("Rare classes (fewer than 5 samples):")
    print(rare_classes)
    # Remove classes with fewer than 5 samples
    rare_classes = label_counts[label_counts < 5].index
    df = df[~df["label"].isin(rare_classes)]
    print(df["label"].value_counts())



def split_dataset():
    """Splits the dataset into train, validation, and test sets."""
    df = pd.read_csv(LABELS_CSV)
    
    # Remove NaN values in 'label' column
    df = df.dropna(subset=["label"])
     
    # Count occurrences of each label
    label_counts = df["label"].value_counts()

    # # Filter out labels that appear only once
    # rare_classes = label_counts[label_counts < 2].index
    # df = df[~df["label"].isin(rare_classes)]

    # Replace rare labels with 'OTHER'
    # df["label"] = df["label"].apply(lambda x: x if label_counts[x] >= 5 else "OTHER")

    # Ensure only labeled data is used for splitting
    df_labeled = df[df["label"] != "UNKNOWN"]
   
    # Check again if there are any missing labels after cleaning
    if df_labeled["label"].isnull().sum() > 0:
        print("Warning: Some labels are still missing!")
    

    print(df_labeled)

    # Stratified split to maintain label distribution
    train, temp = train_test_split(df_labeled, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Save the splits
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print(f"Dataset split complete: {len(train)} train, {len(val)} validation, {len(test)} test.")

if __name__ == "__main__":
    # check_dataset()
    split_dataset()
