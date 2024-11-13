import numpy as np
import cv2
import os
import nibabel as nib
import pandas as pd
import re

# Paths
base_dir = r"C:\Users\noman\Downloads\CTClassification\dataset\VOIs\image"  # Path to main folder with .nii.gz files
save_dir = r"C:\Users\noman\Downloads\CTClassification\processed_data"  # Directory to save processed images
metadata_path = r"C:\Users\noman\Downloads\CTClassification\dataSet\MetadatabyNoduleMaxVoting.xlsx"  # Metadata file path

# Load metadata
metadata = pd.read_excel(metadata_path, engine="openpyxl")
metadata = metadata.set_index(['patient_id', 'nodule_id'])  # Use patient_id and nodule_id as a composite index

# Normalize Hounsfield units to [0,1]
def normalize_hu(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return np.clip(image, 0, 1)

# Save slices of each image
def save_slices(image, label, index):
    train_dirs = ['z', 'x', 'y']
    for directory in train_dirs:
        os.makedirs(os.path.join(save_dir, directory), exist_ok=True)

    # Save label in label.txt
    with open(os.path.join(save_dir, 'label.txt'), 'a') as txtfile:
        txtfile.write(f"{index}.jpg {label}\n")

    # Extract middle slices and save
    z_slice = image[:, :, image.shape[2] // 2]
    x_slice = image[image.shape[0] // 2, :, :]
    y_slice = image[:, image.shape[1] // 2, :]

    for direction, slice_data in zip(['z', 'x', 'y'], [z_slice, x_slice, y_slice]):
        resized_slice = cv2.resize(slice_data * 255, (50, 50)).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, direction, f"{index}.jpg"), resized_slice)

# Main function to process .nii.gz files
def main():
    print("Starting to process .nii.gz files...")

    index = 0
    total_files = 0
    for root, dirs, files in os.walk(base_dir):  # Walk through all subdirectories
        for file in files:
            if file.endswith('.nii.gz'):
                dicom_name = file
                total_files += 1

                # Use regex to extract patient_id and nodule_id
                match = re.match(r"(LIDC-IDRI-\d+)_R_(\d+)", dicom_name)
                if match:
                    patient_id = match.group(1)  # Extracts the patient ID (e.g., LIDC-IDRI-0870)
                    nodule_id = int(match.group(2))  # Extracts the nodule ID as an integer (e.g., 9)
                else:
                    continue  # Skip files with unrecognized naming format

                # Load the .nii.gz file
                path_dicom = os.path.join(root, dicom_name)
                try:
                    img = nib.load(path_dicom)
                    vol = img.get_fdata()
                except Exception as e:
                    print(f"Error loading {dicom_name}: {e}")
                    continue

                # Retrieve the label using the composite index
                if (patient_id, nodule_id) in metadata.index:
                    label = metadata.loc[(patient_id, nodule_id), 'Diagnosis_value']  # Use Diagnosis_value as label

                    # Normalize and save slices
                    normalized_image = normalize_hu(vol)
                    save_slices(normalized_image, label, index)
                    index += 1

                    # Print progress every 100 files
                    if index % 100 == 0:
                        print(f"Processed {index} files so far...")
                else:
                    continue  # Skip if no metadata found

    print(f"Processing complete. Total files processed: {index}/{total_files}")

if __name__ == "__main__":
    main()
