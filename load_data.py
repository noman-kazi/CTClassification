import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time

# Helper function to load an image with a simulated timeout (Windows-compatible)
def load_image_with_timeout(image_path, timeout=5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            image = Image.open(image_path)  # Using PIL to open the image
            return img_to_array(image)  # Convert image to array format
        except Exception as e:
            print(f"Warning: Attempt to load image '{image_path}' failed with error {str(e)}.")
            return None
    print(f"Warning: Timeout reached for image '{image_path}'. Skipping.")
    return None

# Main function with enhanced loading
def load_data(label_path, data_path):
    data_x, data_y, data_z, labels = [], [], [], []
    print("Starting data loading...")

    # Check if label file and directories exist
    if not os.path.exists(label_path):
        print(f"Error: Label file '{label_path}' not found.")
        return None, None, None, None
    if not all(os.path.exists(os.path.join(data_path, d)) for d in ['x', 'y', 'z']):
        print(f"Error: Missing one or more image directories ('x', 'y', 'z') in '{data_path}'.")
        return None, None, None, None

    # Load labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        file_name, label = line.strip().split()
        label = int(label)

        for direction, data_list in zip(['x', 'y', 'z'], [data_x, data_y, data_z]):
            image_path = os.path.join(data_path, direction, file_name)
            if not os.path.exists(image_path):
                print(f"Warning: Image '{image_path}' not found. Skipping.")
                continue

            # Attempt loading the image with a custom timeout
            image = load_image_with_timeout(image_path)
            if image is not None:
                data_list.append(image)

        labels.append(label)

    print("Data loading complete.")
    return np.array(labels), np.array(data_x) / 255.0, np.array(data_y) / 255.0, np.array(data_z) / 255.0
