import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle

# Load label encoder
with open("models_checkpoint/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


def preprocess_image(image_path):
    """Converts image to grayscale and applies adaptive thresholding."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.convertScaleAbs(image, alpha=1.5)

    # Apply median blur to remove small specks
    denoised = cv2.medianBlur(image, 1)

    # Apply morphological closing to remove noise (dilation followed by erosion)
    kernel = np.ones((1,1), np.uint8)  # Adjust kernel size if needed
    clean = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    blurred = cv2.GaussianBlur(clean, (1,1), 0)  # Reduce noise
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 13, 4)  

    # **Dilate the Text to Make It Thicker**
    # dilation_kernel = np.ones((1, 1), np.uint8)  # Adjust size if needed
    # thick_text = cv2.dilate(binary, dilation_kernel, iterations=1)

    # plt.imshow(thick_text, cmap="gray")
    # plt.show()

    return binary

def find_text_contours(binary_image):
    """Finds contours of potential text regions in the binary image."""
    # Find all external contours (ignore nested ones)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
import numpy as np

def sort_text_segments(text_segments):
    """Sorts text segments into natural reading order: top-to-bottom, then left-to-right."""

    if not text_segments:
        return []

    # Extract bounding box information while ensuring correct structure
    bounding_boxes = []
    for item in text_segments:
        if len(item) == 5:  # Ensure correct format
            bounding_boxes.append(item)
        else:
            print("Unexpected format in text_segments:", item)  # Debugging

    # Step 1: Sort primarily by y-coordinate (top to bottom)
    bounding_boxes.sort(key=lambda box: box[1])  

    # Estimate a dynamic threshold based on median line height
    median_line_height = np.median([h for _, _, _, h, _ in bounding_boxes])
    threshold = max(10, median_line_height * 0.5)  # Adaptive threshold

    # Step 2: Group bounding boxes into lines based on y-coordinate proximity
    lines = []
    current_line = [bounding_boxes[0]]

    for box in bounding_boxes[1:]:
        _, y, _, h, _ = box
        prev_y = current_line[-1][1]  # y of last element in current line

        if abs(y - prev_y) < threshold:  # If the text is close, treat as the same line
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]

    lines.append(current_line)  # Add the last line

    # Step 3: Sort each line by x-coordinate (left to right)
    for line in lines:
        line.sort(key=lambda box: box[0])  # Sort each line based on x

    # Step 4: Preserve full structure for debugging
    sorted_boxes = [tuple(box) for line in lines for box in line]

    # Extract only segments for OCR
    # sorted_segments = [segment for _, _, _, _, segment in sorted_boxes]

    return sorted_boxes

def extract_text_segments(image, contours):
    """Extracts text segments from an image while filtering out noise and lines."""
    
    text_segments = []
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # **🔹 Filter out very small noise**  
        min_area = 50  # Adjust this if needed
        if w * h < min_area:
            continue  # Ignore tiny specks
        
        # **🔹 Remove horizontal/vertical lines**
        aspect_ratio = w / float(h)  
        if aspect_ratio > 8 or aspect_ratio < 0.2:
            continue  # Likely a long line or a very tall shape (not text)

        # **🔹 Expand bounding box slightly for better segmentation**
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        segment = image[y:y+h, x:x+w]
        bounding_boxes.append((x, y, w, h, segment))  # Store bounding box info

    # **🔹 Sort in natural reading order**
    sorted_boxes = sort_text_segments(bounding_boxes)

    # # **🔹 Extract sorted text segments**
    text_segments = [segment for  _, _, _, _, segment in sorted_boxes]
    
    return text_segments

# def extract_text_segments(image, contours):
#     """Extracts and sorts text segments from an image."""
#     """Detects text regions but filters out notebook lines."""

#     # **🔹 Ensure the image is grayscale**
#     if len(image.shape) == 3:  # If it's RGB/BGR (3 channels)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image  # Already grayscale

#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Binarization

#     # Alternative (try if OTSU is not working well):
#     # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)  
    

#     # **🔹 Morphological Operations (Dilation to Merge Broken Text)**
#     # kernel = np.ones((1, 1), np.uint8)  # Adjust kernel size based on text thickness
#     # thresh = cv2.dilate(thresh, kernel, iterations=3)  

#     # Find contours of potential text segments
    
#     text_segments = []
#     bounding_boxes = []

#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)

#         # **🔹 Filter out horizontal lines**
#         aspect_ratio = w / float(h)  # Width-to-height ratio
#         if aspect_ratio > 8:  # If it's too long and thin, it's likely a line
#             continue  

#         # If it's too little it's likely a speck 
#         # **🔹 Filter out specks and tiny noise**
#         min_area = 50  # Adjust based on dataset (experiment!)
#         if w * h < min_area:  # If the area is too small, it's likely noise
#             continue  


#         # **🔹 Resize segment slightly larger**
#         padding = 5  # Increase the box size a bit
#         x = max(0, x - padding)
#         y = max(0, y - padding)
#         w = min(image.shape[1] - x, w + 2 * padding)
#         h = min(image.shape[0] - y, h + 2 * padding)

#         segment = image[y:y+h, x:x+w]
#         text_segments.append(segment)
#         bounding_boxes.append((x, y, w, h))

#   # 🔹 **Sort bounding boxes by natural reading order**
#     def sorting_criteria(box):
#         x, y, w, h = box
#         return (y // 10, x)  # Group by line (y) first, then left-to-right (x)

#     sorted_indices = sorted(range(len(bounding_boxes)), key=lambda i: sorting_criteria(bounding_boxes[i]))
#     sorted_text_segments = [text_segments[i] for i in sorted_indices]

#     return sorted_text_segments

def display_segments(text_segments, cols=5):
    """Displays all detected text segments in a single grid, ensuring they are valid images."""
    
    num_segments = len(text_segments)
    if num_segments == 0:
        print("No text segments detected!")
        return

    # 🔹 Extract images safely
    segment_images = []
    for seg in text_segments:
        if isinstance(seg, np.ndarray) and len(seg.shape) == 2:  # Ensure it's a 2D image
            segment_images.append(seg)
        else:
            print("Skipping invalid segment:", seg.shape if isinstance(seg, np.ndarray) else type(seg))  # Debugging
    
    num_segments = len(segment_images)  # Update count after filtering
    if num_segments == 0:
        print("No valid text segments to display!")
        return

    # 🔹 Define grid size
    rows = (num_segments // cols) + int(num_segments % cols != 0)  # Round up

    # 🔹 Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, segment in enumerate(segment_images):
        axes[i].imshow(segment, cmap="gray")
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.show()


def sort_segments_by_line(text_segments, line_threshold=15):
    """Sorts text segments into natural reading order (row-by-row, left-to-right)."""
    
    # 🔹 **Step 1: Sort by Y position (top to bottom)**
    text_segments.sort(key=lambda box: box[1])  

    # 🔹 **Step 2: Group into rows based on Y threshold**
    lines = []
    current_line = [text_segments[0]]  # Start with the first segment

    for i in range(1, len(text_segments)):
        x, y, w, h, segment = text_segments[i]

        # Check vertical distance from previous character
        _, prev_y, _, prev_h, _ = text_segments[i - 1]
        if abs(y - prev_y) < line_threshold:  
            current_line.append(text_segments[i])  # Same line
        else:
            lines.append(current_line)  # Save previous line
            current_line = [text_segments[i]]  # Start new line

    if current_line:
        lines.append(current_line)  # Add last line

    # 🔹 **Step 3: Sort each row left-to-right**
    for line in lines:
        line.sort(key=lambda box: box[0])  

    # 🔹 **Flatten the sorted list**
    sorted_segments = [seg for line in lines for seg in line]
    
    return sorted_segments


def segment_text_with_components(binary_image):
    """Segments text using Connected Components for better accuracy."""
    # Get connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    text_segments = []
    for i in range(1, num_labels):  # Start from 1 to ignore background
        x, y, w, h, area = stats[i]
        
        # 🔹 **Filter out small noise**
        if area < 30:  
            continue  

        # 🔹 **Extract and store the text segment**
        segment = binary_image[y:y+h, x:x+w]
        text_segments.append((x, y, w, h, segment))
    
    return text_segments

# Load trained model
ocr_model = load_model("models_checkpoint/bandungbondowoso.keras")

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
    image_path = "input_image/niat_cropped.jpeg"  # Replace with your image
    binary_image = preprocess_image(image_path)

    plt.imshow(binary_image, cmap="gray")
    plt.show()
    
    contours = find_text_contours(binary_image)
    output = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)  # Convert to color
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)  # Draw detected regions

    plt.imshow(output)
    plt.show()

    # # Extracted segments
    # text_segments = extract_text_segments(binary_image, contours)

    # 🔹 **Use Connected Components for better segmentation**
    text_segments = extract_text_segments(binary_image, contours)
    # 🔹 **Sort properly (line by line, then left to right)**
    # sorted_text_segments = sort_text_segments(text_segments)
    
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