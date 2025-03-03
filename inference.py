# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:21:07 2024

@author: ferra
"""

import os
import glob
import numpy as np
import tifffile as tiff
import tensorflow as tf
from tensorflow.keras.models import load_model

#Let's use CPU for inference
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disables GPU visibility

# Path to the directory containing new dataset images
input_directory = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20211018_tiles_woverlap"

output_directory = os.path.join(input_directory, "predicted_masks")

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load the trained model (assuming no need for the optimizer, set compile=False if not adjusting weights)
#Load the model without expecting the optimizer state: (to avoid incompatibility with models trained on successive versions of tensorflow)
model = load_model(r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\weights_unet\U-Net-Weights-BFCE_IOWA2020.h5", compile=False)

# Function to load and preprocess images
def load_and_preprocess_images(directory):
    image_paths = glob.glob(os.path.join(directory, "*.tif"))
    images = [tiff.imread(fp) for fp in image_paths]
    images = np.array(images)

    # Normalize the images (ensure this matches training preprocessing)
    images[:,:,:,0] = images[:,:,:,0] / 10000.0  # Normalize B2 channel
    images[:,:,:,1] = images[:,:,:,1] / 65535.0  # Normalize NDVI channel
    images[:,:,:,2] = images[:,:,:,2] / 65535.0  # Normalize NDWI channel
    
    return images, image_paths

# Load and preprocess new images
new_images, image_paths = load_and_preprocess_images(input_directory)

# Perform inference
predictions = model.predict(new_images)

# Convert predictions to binary masks
predicted_masks = (predictions > 0.5).astype(np.uint8)

# Save each predicted mask
for i, predicted_mask in enumerate(predicted_masks):
    filename = os.path.basename(image_paths[i])
    output_path = os.path.join(output_directory, f"predicted_{filename}")
    tiff.imwrite(output_path, predicted_mask)

print(f"Inference done. Predicted masks saved to: {output_directory}")