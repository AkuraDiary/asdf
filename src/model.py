import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define input shape
IMG_WIDTH = 128
IMG_HEIGHT = 32

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Creates a Transformer Encoder block."""
    x = layers.LayerNormalization()(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(x, x)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([x, inputs])

    res = layers.LayerNormalization()(x)
    res = layers.Dense(ff_dim, activation="relu")(res)
    res = layers.Dropout(dropout)(res)
    res = layers.Dense(inputs.shape[-1])(res)
    x = layers.Add()([x, res])
    
    return x


def build_ocr_model(NUM_CLASSES):
    """Builds a CNN-Transformer model for handwriting recognition."""

    # Input layer
    input_img = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

    # CNN feature extraction
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape for Transformer
    new_shape = x.shape[1] * x.shape[2]  
    x = layers.Reshape(target_shape=(new_shape, x.shape[3]))(x)

    # Transformer Encoder Layers
    x = transformer_encoder(x, head_size=128, num_heads=4, ff_dim=256, dropout=0.1)
    x = transformer_encoder(x, head_size=128, num_heads=4, ff_dim=256, dropout=0.1)
    x = transformer_encoder(x, head_size=128, num_heads=4, ff_dim=256, dropout=0.1)

    # Fully connected output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    # Model compilation
    model = keras.Model(inputs=input_img, outputs=x, name="OCR_Transformer")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

if __name__ == "__main__":
    # Build the model
    NUM_CLASSES = 10000  # Adjust according to your vocabulary size
    ocr_model = build_ocr_model(NUM_CLASSES)
    ocr_model.summary()

