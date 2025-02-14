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

    # # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply median blur to remove small specks
    denoised = cv2.medianBlur(image, 3)

    # Apply morphological closing to remove noise (dilation followed by erosion)
    kernel = np.ones((1,1), np.uint8)  # Adjust kernel size if needed
    clean = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    blurred = cv2.GaussianBlur(clean, (5,5), 0)  # Reduce noise
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)  # Invert colors
    return binary

def find_text_contours(binary_image):
    """Finds contours of text in the binary image."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_text_segments(image, contours):
    """Extracts and sorts text segments from an image."""
    """Detects text regions but filters out notebook lines."""

    # **🔹 Ensure the image is grayscale**
    if len(image.shape) == 3:  # If it's RGB/BGR (3 channels)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Binarization

    # Alternative (try if OTSU is not working well):
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)  
    

    # **🔹 Morphological Operations (Dilation to Merge Broken Text)**
    kernel = np.ones((1, 3), np.uint8)  # Adjust kernel size based on text thickness
    thresh = cv2.dilate(thresh, kernel, iterations=1)  

    # Find contours of potential text segments
    
    text_segments = []
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # **🔹 Filter out horizontal lines**
        aspect_ratio = w / float(h)  # Width-to-height ratio
        if aspect_ratio > 10:  # If it's too long and thin, it's likely a line
            continue  

        # If it's too little it's likely a speck 
        # **🔹 Filter out specks and tiny noise**
        min_area = 30  # Adjust based on dataset (experiment!)
        if w * h < min_area:  # If the area is too small, it's likely noise
            continue  


        # **🔹 Resize segment slightly larger**
        padding = 5  # Increase the box size a bit
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        segment = image[y:y+h, x:x+w]
        text_segments.append(segment)
        bounding_boxes.append((x, y, w, h))


    # **🔹 Sorting by natural reading order (top-to-bottom, then left-to-right)**
    sorted_indices = sorted(range(len(bounding_boxes)), key=lambda i: (bounding_boxes[i][1], bounding_boxes[i][0]))
    sorted_text_segments = [text_segments[i] for i in sorted_indices]

    return sorted_text_segments

def display_segments(text_segments, cols=5):
    """Displays all detected text segments in a single grid."""
    num_segments = len(text_segments)
    
    if num_segments == 0:
        print("No text segments detected!")
        return

    # 🔹 Define grid size
    rows = (num_segments // cols) + int(num_segments % cols != 0)  # Round up

    # 🔹 Create a figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))  # Adjust size
    axes = axes.flatten()  # Flatten in case of a single row
    
    for i, segment in enumerate(text_segments):
        axes[i].imshow(segment, cmap="gray")
        axes[i].axis("off")  # Hide axis
    
    # Hide unused subplots (if any)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.show()


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
    image_path = "src/input_image/niat_cropped.jpeg"  # Replace with your image
    binary_image = preprocess_image(image_path)

    plt.imshow(binary_image, cmap="gray")
    plt.show()
    
    contours = find_text_contours(binary_image)
    output = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)  # Convert to color
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)  # Draw detected regions

    plt.imshow(output)
    plt.show()

    # # Extracted segments
    text_segments = extract_text_segments(binary_image, contours)
    
    # # Show a few extracted parts
    # # Show first 5
    print("Found text segment : ", len(text_segments))
    display_segments(text_segments, 10)
    # # for i, (_, _, _, _, segment) in enumerate(text_segments):  
    # #     plt.subplot(1, len(text_segments), i + 1)
    # #     plt.imshow(segment, cmap="gray")
    # #     plt.show()
    
    # # part of code where we actually feed the data into model

    # # Example: Run OCR on extracted segments
    # # for _, _, _, _, segment in text_segments: 
    # #     text = predict_text(segment)
    # #     print("Predicted Text:", text)

    # # Example usage
    # full_text = reconstruct_text([predict_text(seg) for seg in text_segments])

    # # full_text = reconstruct_text([predict_text(seg) for _, _, _, _, seg in text_segments])
    # print("Final OCR Output:\n", full_text)