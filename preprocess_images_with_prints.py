import os
import cv2
import numpy as np

# Path to your dataset folder
dataset_path = 'C:/Users/UDAYK/OneDrive/Desktop/udayproject/dataset'

# Resize and normalize images
def preprocess_images(folder_path, target_size=(224, 224)):
    for folder_name in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_full_path):
            print(f'Processing folder: {folder_name}')
            for img_name in os.listdir(folder_full_path):
                img_full_path = os.path.join(folder_full_path, img_name)
                img = cv2.imread(img_full_path)
                if img is not None:
                    # Resize image
                    img_resized = cv2.resize(img, target_size)
                    # Normalize pixel values
                    img_normalized = img_resized / 255.0
                    # Save preprocessed image
                    cv2.imwrite(img_full_path, img_normalized * 255)
                    print(f'Processed image: {img_name}')
                else:
                    print(f'Error reading image: {img_name}')

preprocess_images(dataset_path)
print('Preprocessing completed!')
