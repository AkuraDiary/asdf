import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import text_segmentation_new as tsn
from pprint import pprint

image_path = "input_image/niat_cropped.jpeg"



def preprocess_image(image):
    """Converts an image to grayscale, applies thresholding, and removes noise."""
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)  # Thresholding
    kernel = np.ones((1, 1), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Noise removal
    return clean

def detect_words(image, min_word_height=5, min_word_ratio=1.5, max_word_ratio=5.0):
    """Detects words in a preprocessed image and returns a structured format preserving line breaks."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    words = []
    lines = []
    prev_y = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Get bounding box
        
        # Filter out "thin" lines (height threshold)
        # Calculate the width-to-height ratio
        aspect_ratio = w / h
            
        if h < min_word_height :
            continue  # Skip this bounding box as it's too thin to be a word
        
        if aspect_ratio > min_word_ratio:
                continue  # Skip this bounding box if it's too extreme (resembles a line)
        
        if prev_y is not None and abs(y - prev_y) > h * 1:
            words.append(lines)  # Append previous line
            lines = []  # Start a new line
        
        lines.append((x, y, w, h))  # Save word position
        prev_y = y  # Update previous Y position

    if lines:
        words.append(lines)  # Append last line
    
    img_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
    for line_idx, line in enumerate(words):
        color = colors[line_idx % len(colors)]  # Cycle colors for each line
        for (x, y, w, h) in line:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)

    # Show the detected words
    cv2.imshow("Detected Words", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return words  # Returns words structured in lines

def structure_text(words, space_threshold=15, vertical_threshold=1):
    """Converts bounding box data into a readable format for later processing."""
    structured_text = []
    
    for line in words:
        word_images = []
        current_cluster = []  # To hold words that should be merged together
        for i, (x, y, w, h) in enumerate(sorted(line, key=lambda b: b[0])):  # Sort words by x coordinate
            if not current_cluster:
                current_cluster.append((x, y, w, h))  # Add the first word to the cluster
            else:
                prev_x, prev_y, prev_w, prev_h = current_cluster[-1]
                if x - (prev_x + prev_w) <= space_threshold: #and y - (prev_y + prev_h) < vertical_threshold:  # Merge if space is small enough
                    # Merge the bounding boxes by extending the width of the current cluster
                    new_x = min(prev_x, x)
                    new_y = min(prev_y, y)
                    new_w = max(prev_x + prev_w, x + w) - new_x
                    new_h = max(prev_y + prev_h, y + h) - new_y
                    current_cluster[-1] = (new_x, new_y, new_w, new_h)
                else:
                    # If the gap is too large, finalize the current cluster and start a new one
                    word_images.append(current_cluster[-1])
                    current_cluster = [(x, y, w, h)]  # Start a new cluster

        if current_cluster:  # Don't forget to add the last cluster
            word_images.append(current_cluster[-1])
        
        structured_text.append(word_images)

    return structured_text



def process_words(image, structured_text):
    """Extracts words from the image and sends them to the character detection function."""
    for line in structured_text:
        for (x, y, w, h) in line:  # Directly unpack word bounding boxes
            word_img = image[y:y+h, x:x+w]  # Crop the word

            # Ensure valid crop
            if word_img.size == 0:
                continue

            cv2.imshow("Word to be fed into detection", word_img)
            cv2.waitKey(0)  # Show for 500ms
            # detect_and_plot_characters(word_img)  # Pass to your function

    cv2.destroyAllWindows()
    tsn.detect_and_plot_characters(word_img)  # Process the word

def flip_lines(structured_text):
    """Reverses the order of lines in the structured text while keeping words in order."""
    return structured_text[::-1]  # Reverse the list of lines


## PLOTTING AREA

def plot_structured_text(image, structured_text):
    img_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Different colors for each line

    for line_idx, line in enumerate(structured_text):
        color = colors[line_idx % len(colors)]  # Cycle colors for each line
        for (x, y, w, h) in line:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 1)

    cv2.imshow("Structured Text with Line Breaks", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
if __name__ == "__main__":
    image_original = cv2.imread(image_path)
    image_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image_grayscale)
    plt.title("Unprocessed Image")
    plt.show()
  
    
    image = preprocess_image(image_grayscale)
    plt.imshow(image)
    plt.title("Preprocessed Image")
    plt.show()

    
    words_detected = detect_words(image)
    
    print("words detected")
    pprint(words_detected)
    
    structured_text = structure_text(words_detected)
    print("structured text")
    pprint(structured_text)
    
    plot_structured_text(image, structured_text)
    
    process_words(image_original, flip_lines(structured_text))
    
    
  



