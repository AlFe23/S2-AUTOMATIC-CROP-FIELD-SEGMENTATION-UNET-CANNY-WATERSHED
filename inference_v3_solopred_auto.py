"""
Automated Batch Inference Script for Sentinel-2 Crop Segmentation (v3 Solo Prediction)

This version is optimized for **automated batch inference**, allowing users to process  
**multiple folders of Sentinel-2 image tiles** without manual intervention.  
It runs **inference only** without evaluation.

Key Features:
- **Processes multiple directories of Sentinel-2 tiles automatically.**
- **Runs inference without requiring ground truth labels.**
- **Batch Processing:** Scans a base directory and processes all subfolders.
- **Saves predicted masks** in separate output folders.
- **Supports both `.keras` and `.h5` model formats**.

Differences from the Previous Version:
- **Fully Automated:** Iterates over multiple folders without manual input.
- **No Evaluation Step:** Only generates predictions, without ground truth comparison.
- **Batch Directory Support:** Processes all datasets in a given base directory.
- **Optimized Model Loading:** Ensures compatibility with `.keras` and `.h5` models.

Quick User Guide:
1. Set `base_directory` to the location of your Sentinel-2 tile folders.
2. Ensure the **U-Net model** (either `.keras` or `.h5`) is available at `model_path`.
3. Run the script:
       python inference_v3_solopred_auto.py
4. The output masks will be saved in separate `_predicted/` folders inside each tile directory.

Dependencies:
Python packages: numpy, gdal (from osgeo), glob, os, tifffile, tensorflow (keras)

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

"""


import os
import glob
import numpy as np
import tifffile as tiff
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K  # Used only for safe backend functions


# Disabilita la visibilità della GPU per eseguire l'inferenza su CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Define a smoothing factor for the Dice coefficient calculation
smooth = 1

def dice_coef(y_true, y_pred):
    """
    Calculate Dice Coefficient using TensorFlow backend.

    Parameters:
    - y_true: Ground truth labels.
    - y_pred: Predicted labels.

    Returns:
    - Dice Coefficient score.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def load_trained_model(model_path):
    """
    Load the pre-trained model from the specified path.

    Supports both `.keras` and `.h5` formats.

    Parameters:
    - model_path (str): Path to the model file.

    Returns:
    - model (tf.keras.Model): Loaded and compiled model.
    """
    if model_path.endswith('.keras'):
        model = load_model(model_path, compile=False)
    elif model_path.endswith('.h5'):
        from keras.layers import Conv2DTranspose
        
        # Custom layer handling for older .h5 models
        class CustomConv2DTranspose(Conv2DTranspose):
            def __init__(self, *args, **kwargs):
                if 'groups' in kwargs:
                    kwargs.pop('groups')
                super(CustomConv2DTranspose, self).__init__(*args, **kwargs)

        model = load_model(model_path, custom_objects={'Conv2DTranspose': CustomConv2DTranspose}, compile=False)
    else:
        raise ValueError("Unsupported model format. Only '.keras' and '.h5' formats are supported.")
    
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='binary_crossentropy',  # Loss function used during training
        metrics=[dice_coef]
    )
    
    return model

def load_and_preprocess_images(image_directory):
    """
    Load and preprocess images for inference.

    Parameters:
    - image_directory (str): Directory containing image tiles.

    Returns:
    - images (np.ndarray): Array of preprocessed images.
    - image_paths (list): List of image file paths.
    """
    image_paths = glob.glob(os.path.join(image_directory, "*.tif"))
    images = [tiff.imread(image_path) for image_path in image_paths]
    images = np.array(images)

    # Normalize the images (ensure it matches the preprocessing during training)
    images[:,:,:,0] = images[:,:,:,0] / 10000.0  # Normalize B2 channel
    images[:,:,:,1] = images[:,:,:,1] / 65535.0  # Normalize NDVI channel
    images[:,:,:,2] = images[:,:,:,2] / 65535.0  # Normalize NDWI channel

    return images, image_paths

def run_inference_on_directory(model, input_directory, output_directory):
    """
    Run inference on all image tiles in a directory and save predicted masks.

    Parameters:
    - model (tf.keras.Model): Pre-trained model for inference.
    - input_directory (str): Directory containing input image tiles.
    - output_directory (str): Directory where predicted masks will be saved.

    Returns:
    - None
    """
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Load and preprocess images
    images, image_paths = load_and_preprocess_images(input_directory)

    # Run inference
    predictions = model.predict(images)

    # Convert predictions to binary masks
    predicted_masks = (predictions > 0.5).astype(np.uint8)

    # Save each predicted mask
    for i, predicted_mask in enumerate(predicted_masks):
        filename = os.path.basename(image_paths[i])
        output_path = os.path.join(output_directory, f"predicted_{filename}")
        tiff.imwrite(output_path, predicted_mask)

    print(f"Inference completed for directory: {input_directory}. Predicted masks saved in: {output_directory}")

def main():
    """
    Main function to run the inference workflow for Sentinel-2 image tiles.
    """
    # Base directory containing multiple folders of processed tiles from sub-tiling step
    base_directory = '/mnt/h/Alvise/S2_samples'  # Replace with your actual tiles base directory path
    
    # Path to the pre-trained model
    model_path = '/mnt/h/Alvise/training_DS_A/weights_unet/BFC-2024-05-16-204727/U-Net-Weights-BFCE.h5'  # Replace with your model path

    # Load the pre-trained model
    print("Loading trained model...")
    model = load_trained_model(model_path)
    print("Model loaded successfully.")

    # Iterate over each folder in the base directory
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)

        if os.path.isdir(subdir_path):  # Ensure it's a directory
            print(f"Starting inference on tiles in directory: {subdir_path}")

            # Create output directory in the same directory as input, with a suffix '_predicted'
            output_directory = os.path.join(base_directory, f"{subdir}_predicted")

            # Run inference on all image tiles in the current directory
            run_inference_on_directory(model, subdir_path, output_directory)

    print("Inference complete for all folders. Predicted masks are saved.")

if __name__ == "__main__":
    main()
