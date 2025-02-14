import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define input shape
IMG_WIDTH = 128
IMG_HEIGHT = 32

def build_ocr_model(NUM_CLASSES):
    """Builds a CNN-LSTM model for handwriting recognition."""

    # Input layer
    input_img = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

    # CNN feature extraction
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(input_img) 
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)  
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)


    # Reshape for LSTM
    new_shape = x.shape[1] * x.shape[2]  # Preserve total elements
    x = layers.Reshape(target_shape=(new_shape, x.shape[3]))(x)


    # LSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)  # New LSTM layer with fewer units

    # Attention Laye
    attention = layers.Attention()([x, x])  
    x = layers.Add()([x, attention])  # Combine attention with LSTM output

    # Fully connected layer
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    # x = layers.Dense(256, activation="relu")(x)
    
    # outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x) 

    # Model compilation
    model = keras.Model(inputs=input_img, outputs=x, name="Doctor_OCR_Model")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

if __name__ == "__main__":
    # Build the model
    ocr_model = build_ocr_model()
    ocr_model.summary()
