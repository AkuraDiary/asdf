import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

image_path = "input_image/random.jpeg"

# IMAGE PREPROCESSING
def preprocess_image(image):
    """Converts image to grayscale, applies thresholding, and removes noise."""
    
    # Step 1: Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Step 2: Adaptive Thresholding for binarization
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)

    # Step 3: Remove horizontal notebook lines using morphological operations
    kernel = np.ones((1, 50), np.uint8)  # Horizontal kernel
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 4: Subtract detected lines from the binary image
    cleaned = cv2.subtract(binary, lines)

    return cleaned

def add_padding_to_image(image, padding=10):
    """Adds black padding around the image."""
    
    # Add padding to each side of the image (top, bottom, left, right)
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_image

# TEXT AND CHARACTER DETECTION
def apply_dt_and_watershed(binary_image):
    """Uses distance transform & watershed to split joined characters."""
    
    # Step 0: Apply dilation to help connect nearby characters (such as dots on "i" and "j")
    dilated_image = cv2.dilate(binary_image, np.ones((3, 3), np.uint8), iterations=3)

    # Step 1: Compute distance transform
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L1, 0)
    # normalize the distance transform
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX) 
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    # Step 2: Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(binary_image, sure_fg)

    # Show intermediate results for debugging
    plt.imshow(sure_fg, cmap="gray")
    plt.title("sure fg")
    plt.show()

    # Show intermediate results for debugging
    plt.imshow(unknown, cmap="gray")
    plt.title("unknown")
    plt.show()
    

    # Step 3: Marker labelling
    _, markers = cv2.connectedComponents(sure_fg+unknown)
    markers += 1  # Ensure background is different
    markers[unknown == 255] = 0  # Mark unknown regions
    # markers[sure_fg == 255] = 0  # Mark sure_fg regions

    # Step 4: Apply Watershed
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_image, markers)

    
    # Mark boundaries in red
    color_image[markers == -1] = [0, 255, 0]

    # Convert markers to a binary image for contour detection (0 and 255)
    binary_watershed = np.uint8(markers == -1) * 255

    # Show intermediate results for debugging
    plt.imshow(binary_watershed, cmap="gray")
    plt.title("Binary Watershed")
    plt.show()

    plt.imshow(color_image)
    plt.title("Color Image with Watershed Boundaries")
    plt.show()

    return binary_watershed, color_image

def get_bounding_boxes(binary_watershed):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_watershed)
    bounding_boxes = []
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        bounding_boxes.append((x, y, w, h))

    return bounding_boxes
    
def filter_noise(bounding_boxes, min_size=5, max_aspect_ratio=10):
    """
    Removes noise by filtering out long horizontal/vertical lines and very small specks.

    Parameters:
    - bounding_boxes: List of (x, y, w, h) bounding boxes.
    - min_size: Minimum width or height to keep a bounding box.
    - max_aspect_ratio: Aspect ratio (width/height or height/width) threshold to filter out lines.

    Returns:
    - List of filtered bounding boxes.
    """
    filtered_boxes = []
    
    for x, y, w, h in bounding_boxes:
        aspect_ratio = max(w / (h + 1e-5), h / (w + 1e-5))  # Avoid division by zero

        # Remove very small noise
        if w < min_size and h < min_size:
            continue  

        # Remove long horizontal and vertical lines
        if aspect_ratio > max_aspect_ratio:
            continue  

        filtered_boxes.append((x, y, w, h))

    return filtered_boxes

def sort_bounding_boxes(bounding_boxes, line_threshold=10):
    """Sorts bounding boxes by lines (top to bottom) and left to right."""
    
    # Step 1: Sort by Y first (top to bottom)
    bounding_boxes.sort(key=lambda box: box[1])

    # Step 2: Group into rows based on Y proximity
    lines = []
    current_line = [bounding_boxes[0]]

    for i in range(1, len(bounding_boxes)):
        x, y, w, h = bounding_boxes[i]
        _, prev_y, _, prev_h = bounding_boxes[i - 1]

        if abs(y - prev_y) < line_threshold:  # If close in Y, they belong to the same row
            current_line.append(bounding_boxes[i])
        else:
            lines.append(current_line)
            current_line = [bounding_boxes[i]]

    if current_line:
        lines.append(current_line)

    # Step 3: Sort each row by X (left to right)
    for line in lines:
        line.sort(key=lambda box: box[0])

    # Flatten sorted lines back into a list
    sorted_boxes = [char for line in lines for char in line]
    return lines, sorted_boxes

def merge_close_bounding_boxes(bounding_boxes, merge_threshold=3):
    """
    Merges small noise bounding boxes into the nearest larger bounding box.

    Parameters:
    - bounding_boxes: List of (x, y, w, h) bounding boxes.
    - merge_threshold: Maximum distance to merge small boxes into larger ones.

    Returns:
    - List of merged bounding boxes.
    """
    if not bounding_boxes:
        return []

    # Sort bounding boxes from left to right
    bounding_boxes.sort(key=lambda box: (box[1], box[0]))

    merged_boxes = []
    while bounding_boxes:
        # Start with the first bounding box
        x, y, w, h = bounding_boxes.pop(0)

        to_merge = []  # List of boxes to merge into this one

        for i, (nx, ny, nw, nh) in enumerate(bounding_boxes):
            # Check if this bounding box is "small" (potential noise)
            is_noise = nw * nh < 50  # Adjust this threshold as needed

            # Check if the box is within the merge_threshold distance
            if is_noise and abs(nx - (x + w)) <= merge_threshold and abs(ny - y) <= merge_threshold:
                to_merge.append((nx, ny, nw, nh))

        # Expand the main bounding box to include all merged ones
        for (mx, my, mw, mh) in to_merge:
            x = min(x, mx)
            y = min(y, my)
            w = max(x + w, mx + mw) - x
            h = max(y + h, my + mh) - y
            bounding_boxes.remove((mx, my, mw, mh))  # Remove merged boxes from list

        merged_boxes.append((x, y, w, h))

    return merged_boxes

def merge_vertical_bounding_boxes(bounding_boxes, vertical_threshold=3):
    """
    Merges vertically close bounding boxes to prevent character splits.

    Parameters:
    - bounding_boxes: List of (x, y, w, h) bounding boxes.
    - vertical_threshold: Max distance (in pixels) between bounding boxes to merge.

    Returns:
    - List of merged bounding boxes.
    """
    if not bounding_boxes:
        return []

    # Sort bounding boxes from top to bottom (primary), left to right (secondary)
    bounding_boxes.sort(key=lambda box: (box[0], box[1]))

    merged_boxes = []
    while bounding_boxes:
        x, y, w, h = bounding_boxes.pop(0)
        
        to_merge = []
        for i, (nx, ny, nw, nh) in enumerate(bounding_boxes):
            # Check if bounding boxes are close in vertical space
            vertical_close = abs(ny - (y + h)) <= vertical_threshold or abs(y - (ny + nh)) <= vertical_threshold
            
            horizontal_overlap = not (nx > (x + w) or (nx + nw) < x)  # Check if X ranges overlap

            if vertical_close and horizontal_overlap:
                to_merge.append((nx, ny, nw, nh))

        # Expand the bounding box to include all vertically close ones
        for (mx, my, mw, mh) in to_merge:
            x = min(x, mx)
            y = min(y, my)
            w = max(x + w, mx + mw) - x
            h = max(y + h, my + mh) + mh-(my-h)
            bounding_boxes.remove((mx, my, mw, mh))  # Remove merged boxes

    
        merged_boxes.append((x, y, w, h))

    return merged_boxes

def draw_bounding_boxes(image, bounding_boxes_coordinate):
    """Draw bounding boxes on the image."""
    for (x, y, w, h) in bounding_boxes_coordinate:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return image

# TEXT AND CHARACTER SEGMENTATION
def prepare_segments(image, bounding_boxes, target_size=(32, 32)):
    """
    Extracts and preprocesses character segments from the image.

    Parameters:
    - image: The original grayscale image (before binarization).
    - bounding_boxes: List of (x, y, w, h) bounding boxes.
    - target_size: Tuple (width, height) representing the model's input size.

    Returns:
    - segments: List of preprocessed character images.
    """
    segments = []
    
    for x, y, w, h in bounding_boxes:
        # Crop the character region
        char_crop = image[y:y+h, x:x+w]

        # Resize while maintaining aspect ratio
        char_resized = resize_with_aspect_ratio(char_crop, target_size)

        # Normalize pixel values (0 to 1)
        char_resized = char_resized.astype(np.float32) / 255.0

        segments.append(char_resized)

    return segments

def resize_with_aspect_ratio(image, target_size, padding=3):
    """
    Resizes an image to fit within target_size while maintaining aspect ratio.
    Pads with black (zero) if needed.

    Parameters:
    - image: Input character image (grayscale).
    - target_size: (width, height) of output image.

    Returns:
    - Resized and padded image.
    """

    padded_image = add_padding_to_image(image, 4)

    h, w = padded_image.shape
    target_w, target_h = target_size

    # Compute scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize while maintaining aspect ratio
    resized = cv2.resize(padded_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank (black) target image
    output = np.zeros((target_h, target_w), dtype=np.uint8)

    # Center the resized character in the target image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return output

def plot_segments(segments, max_cols=10):
    """
    Dynamically plots extracted character segments in sequential order.
    
    Parameters:
    - segments: List of preprocessed character images (grayscale).
    - max_cols: Maximum number of columns (default 10).
    """
    num_chars = len(segments)
    
    # Dynamically determine the number of columns (not exceeding max_cols)
    cols = min(max_cols, math.ceil(math.sqrt(num_chars)))
    
    # Calculate rows dynamically
    rows = math.ceil(num_chars / cols)

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle("Character Segments in Sequential Order", fontsize=14)

    # Flatten axes for easy iteration
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes):
        if i < num_chars:
            ax.imshow(segments[i], cmap="gray")
            ax.set_title(f"#{i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    bin_img = preprocess_image(image)
    plt.imshow(bin_img, cmap="gray")
    plt.title("Preprocessed Image")
    plt.show()

    # Add padding to the image to ensure characters near edges are not cut off
    padded_image = add_padding_to_image(bin_img, padding=4)
    plt.imshow(padded_image, cmap="gray")
    plt.title("Padded Image")
    plt.show()

    # Apply Distance Transform and Watershed to split characters
    bin_img, color_img = apply_dt_and_watershed(padded_image)

    # Assuming binary_watershed is the segmented image from watershed
    bounding_boxes = get_bounding_boxes(bin_img)

    image_bounded_before = draw_bounding_boxes(color_img.copy(), bounding_boxes)
    plt.imshow(image_bounded_before)
    plt.title("Bounding Boxes on Color Image before Processed")
    plt.show()

    print(f"Before merging: {len(bounding_boxes)}")

    lines, sorted_boxes = sort_bounding_boxes(bounding_boxes)
    # image_bounded_sorted = draw_bounding_boxes(color_img.copy(), sorted_boxes[:5])
    # plt.imshow(image_bounded_sorted)
    # plt.title("Bounding Boxes Sorted")
    # plt.show()

    merged_boxes = merge_close_bounding_boxes(sorted_boxes, merge_threshold=3)

    mergeclose_image = draw_bounding_boxes(color_img.copy(), merged_boxes)
    plt.imshow(mergeclose_image)
    plt.title("Merge close Bounding Boxes ")
    plt.show()

    merged_boxes = filter_noise(merged_boxes, 2)
    merged_boxes = merge_vertical_bounding_boxes(merged_boxes, vertical_threshold=3)
    mergevert_image = draw_bounding_boxes(color_img.copy(), merged_boxes)
    plt.imshow(mergevert_image)
    plt.title("Merge vert Bounding Boxes ")
    plt.show()

    final_boxes = filter_noise(merged_boxes)
    
    print(f"After merging: {len(final_boxes)}")
    # Draw bounding boxes based on the binary watershed result
    final_image = draw_bounding_boxes(color_img.copy(), final_boxes)

    plt.imshow(final_image)
    plt.title("Bounding Boxes on Color Image after Processed")
    plt.show()

    segments = prepare_segments(padded_image, final_boxes, target_size=(64, 64))
    plot_segments(segments)
