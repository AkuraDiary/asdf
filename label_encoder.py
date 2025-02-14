import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Paths
CSV_PATH = "src/data/augmented/train_augmented.csv"  # Adjust path if needed
SAVE_PATH = "src/models_checkpoint/label_encoder.pkl"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Fit Label Encoder
label_encoder = LabelEncoder()
labels = df["label"].astype(str).values
label_encoder.fit(labels)

# Save Label Encoder
with open(SAVE_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Label Encoder saved successfully!")
