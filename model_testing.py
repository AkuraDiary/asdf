import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle

# Load label encoder
with open("src/models_checkpoint/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


def preprocess_image(image_path):
    """Converts image to grayscale and applies adaptive thresholding."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5,5), 0)  # Reduce noise
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)  # Invert colors
    return binary

def find_text_contours(binary_image):
    """Finds contours of text in the binary image."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_text_segments(image, contours):
    """Extracts and sorts text segments from an image."""
    segments = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 10:  # Ignore small noise
            segment = image[y:y+h, x:x+w]  # Crop
            segments.append((x, y, w, h, segment))

    # Sort by vertical (y) and horizontal (x) position
    segments = sorted(segments, key=lambda x: (x[1], x[0]))
    return segments

# Load trained model
ocr_model = load_model("src/models_checkpoint/bandungbondowoso.keras")

def predict_text(segment):
    """Resizes and predicts text from a single segment."""
    segment = cv2.resize(segment, (128, 32))  # Match model input size
    segment = segment.astype("float32") / 255.0  # Normalize
    segment = np.expand_dims(segment, axis=[0, -1])  # Add batch & channel dims

    prediction = ocr_model.predict(segment)
    return prediction  # Decode this into text!



def reconstruct_text(predictions):
    """Combines recognized characters into words or sentences."""
    predicted_indices = np.argmax(predictions, axis=-1)  # Get the highest prob. class per timestep

    if predicted_indices.ndim == 2:  # If it's a sequence (batch_size, seq_len)
        decoded_texts = []
        for seq in predicted_indices:
            decoded_text = label_encoder.inverse_transform(seq)  # Decode each sequence separately
            decoded_texts.append("".join(decoded_text))  # Join characters into words
        return " ".join(decoded_texts)  # Combine words into a sentence

    elif predicted_indices.ndim == 1:  # If it's already 1D
        decoded_text = label_encoder.inverse_transform(predicted_indices)
        return "".join(decoded_text)  # Join characters directly

    else:
        raise ValueError(f"Unexpected shape for predicted_indices: {predicted_indices.shape}")

if __name__ == "__main__":
    # Test
    image_path = "src/input_image/niat.jpeg"  # Replace with your image
    binary_image = preprocess_image(image_path)

    plt.imshow(binary_image, cmap="gray")
    plt.show()

    contours = find_text_contours(binary_image)
    output = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)  # Convert to color
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)  # Draw detected regions

    plt.imshow(output)
    plt.show()

    # Extracted segments
    text_segments = extract_text_segments(binary_image, contours)
    
    # Show a few extracted parts
    # Show first 5
    for i, (_, _, _, _, segment) in enumerate(text_segments[:5]):  
        plt.subplot(1, 5, i + 1)
        plt.imshow(segment, cmap="gray")
        plt.show()
    
    # part of code where we actually feed the data into model

    # Example: Run OCR on extracted segments
    # for _, _, _, _, segment in text_segments: 
    #     text = predict_text(segment)
    #     print("Predicted Text:", text)

    # Example usage
    full_text = reconstruct_text([predict_text(seg) for _, _, _, _, seg in text_segments[:10]])
    print("Final OCR Output:\n", full_text)