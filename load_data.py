import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import signal

# Helper function to enforce timeout on image loading
def timeout_handler(signum, frame):
    raise TimeoutError("Image loading timed out.")

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

    # Set a signal for timeout
    signal.signal(signal.SIGALRM, timeout_handler)

    for line in lines:
        file_name, label = line.strip().split()
        label = int(label)

        for direction, data_list in zip(['x', 'y', 'z'], [data_x, data_y, data_z]):
            image_path = os.path.join(data_path, direction, file_name)
            if not os.path.exists(image_path):
                print(f"Warning: Image '{image_path}' not found. Skipping.")
                continue

            # Attempt loading with timeout
            try:
                signal.alarm(5)  # Set a 5-second timeout
                image = Image.open(image_path)  # Using PIL
                image = img_to_array(image)
                data_list.append(image)
                signal.alarm(0)  # Reset alarm after successful load

            except TimeoutError:
                print(f"Warning: Timeout reached while loading '{image_path}'. Skipping.")
                signal.alarm(0)  # Reset alarm if timeout occurs

            except Exception as e:
                print(f"Warning: Failed to load image '{image_path}' due to {str(e)}. Skipping.")

        labels.append(label)

    print("Data loading complete.")
    return np.array(labels), np.array(data_x) / 255.0, np.array(data_y) / 255.0, np.array(data_z) / 255.0
