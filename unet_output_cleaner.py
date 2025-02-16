# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:34:29 2024

@author: Alvise Ferrari

Una volta ottenuto l'output unet e rimosaicata l'immagine integrale, prima di applicare l'algoritmo watershed è necessario effettuare alcune operazioni morfologiche per ripulire
la maschera binaria da elementi isolati assimilabili a rumore e non bordi effettivi.

Questo script è un estratto dello script 'canny_cleaner_v3.py', e parte direttamente dalla maschera binaria (a valori 0-1 )per applicare 'opening' ed il 'remove_small_objects'.

Input:
    Maschera binaria output unet dove 0==bordo, 1==non-bordo.
Output:
    Maschera binaria 0-255 ripulita da elementi indesiderati.
"""

"""
U-Net Output Cleaner for Sentinel-2 Crop Segmentation

This script cleans the **binary segmentation masks** produced by the U-Net model,  
removing **noise and small artifacts** before applying the **watershed algorithm**  
for final crop field segmentation. The cleaning process includes **morphological opening,  
small object removal, and hole filling** to enhance segmentation quality.

Key Features:
- **Removes small noise** from binary masks using morphological operations.
- **Fills small holes** to create cleaner field boundaries.
- **Ensures consistency** before applying watershed segmentation.
- Supports **customizable filtering parameters**.

Quick User Guide:
1. Set `input_path` to the **binary mask** to be cleaned.
2. Set `output_path` to save the **cleaned mask**.
3. Adjust optional parameters:
   - `apply_opening=True`: Enables morphological noise removal.
   - `min_size=9`: Removes objects smaller than this value.
   - `area_threshold=50`: Fills holes smaller than this value.
4. Run the script:
       python unet_output_cleaner.py
5. The cleaned mask will be saved as a **georeferenced GeoTIFF**.

Dependencies:
Python packages: numpy, gdal (from osgeo), skimage (morphology), os

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

"""



import os
import numpy as np
from osgeo import gdal
from skimage.morphology import disk, opening, remove_small_objects, remove_small_holes
import matplotlib.pyplot as plt

def read_geotiff(input_path):
    """Read a GeoTIFF file and return the array, geotransform, and projection."""
    try:
        dataset = gdal.Open(input_path)
        if dataset is None:
            raise IOError(f"Unable to open {input_path}")
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray()
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        return array, geotransform, projection
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return None, None, None

def write_geotiff(output_path, array, geotransform, projection):
    """Write an array to a GeoTIFF file with specified geotransform and projection."""
    try:
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = array.shape
        options = ['COMPRESS=LZW']
        dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte, options=options)
        if dataset is None:
            raise IOError(f"Could not create {output_path}")
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(projection)
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.FlushCache()
    except Exception as e:
        print(f"Error writing {output_path}: {e}")

def process_mask(input_path, output_path, apply_opening=False, min_size=9, area_threshold=50):
    """
    Process the binary mask to remove noise and fill holes, then save the result.
    
    Parameters:
    - input_path: str, path to the input GeoTIFF file.
    - output_path: str, path to save the processed GeoTIFF file.
    - apply_opening: bool, whether to apply the morphological opening operation.
    - min_size: int, minimum size for objects to keep in the mask.
    - area_threshold: int, maximum size for holes to fill in the mask.
    """
    # Read the GeoTIFF
    canny, geotransform, projection = read_geotiff(input_path)
    if canny is None:
        return
    
    # Ensure the input is binary (0 and 1), convert if necessary
    canny = (canny > 0).astype(np.uint8)

    # Apply morphological opening to remove small noise (erosion followed by dilation), if specified
    if apply_opening:
        footprint = disk(1)  # Use a larger, non-zero radius for the disk
        canny = opening(canny, footprint)
    
    # Remove small objects to clean noise
    cleaned_mask = remove_small_objects(canny, min_size=min_size, connectivity=2).astype('uint8')
    
    # Fill small holes within the regions
    filled_mask = remove_small_holes(cleaned_mask, area_threshold=area_threshold).astype('uint8')
    
    # Write the final processed mask to a new file
    write_geotiff(output_path, filled_mask, geotransform, projection)



#######################################################################################



# Example usage with new parameters
# input_path = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/2018_T33TXF_B2_NDVI_NDWI_predicted_mask_combined.tif'
# output_path = input_path.replace('.tif', '_cleaned.tif')

# Set optional parameters
# apply_opening = True  # Set to False if you don't want to apply opening
# min_size = 9  # Adjust minimum object size to remove (actually the maximum size of white holes to be filled)
# area_threshold = 60  # Adjust maximum hole size to fill (actually the maximum size of isolated black objects to be removed  from white backgrounds!)

# Call the function with new arguments
# process_mask(input_path, output_path, apply_opening=apply_opening, min_size=min_size, area_threshold=area_threshold)






