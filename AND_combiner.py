#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:57:45 2024

@author: Alvise Ferrari

AND Combiner for Sentinel-2 Crop Segmentation Masks

This script merges multiple **remosaicked predicted masks** into a single  
binary mask using a **logical AND operation**. This approach ensures that  
only consistent segmentations across multiple predictions are retained.

Key Features:
- **Automatically detects and processes all predicted masks** in a given directory.
- Uses **logical AND** to combine multiple masks into a final segmentation.
- Preserves **geospatial information** (geotransform and projection).
- Saves the **final combined mask as a GeoTIFF**.

Quick User Guide:
1. Set `predicted_mask_dir` to the directory containing **remosaicked predicted masks**.
2. Define the output path (`output_file`) for the final combined mask.
3. Run the script:
       python AND_combiner.py
4. The final **binary mask** will be saved at `output_file`.

Dependencies:
Python packages: numpy, gdal (from osgeo), glob, os

License:
This code is released under the MIT License.




"""

# from osgeo import gdal
# import numpy as np

# # Paths to the input GeoTIFF files
# file1 = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180420T094031_20180420T094644_T33TXF_B2_NDVI_NDWI_predicted_mask.tif'
# file2 = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180525T094029_20180525T094824_T33TXF_B2_NDVI_NDWI_predicted_mask.tif'
# file3 = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180719T094031_20180719T094430_T33TXF_B2_NDVI_NDWI_predicted_mask.tif'

# # Function to read a GeoTIFF file and return the mask array and geotransform
# def read_mask(file_path):
#     dataset = gdal.Open(file_path)
#     mask = dataset.GetRasterBand(1).ReadAsArray()
#     geotransform = dataset.GetGeoTransform()
#     projection = dataset.GetProjection()
#     return mask, geotransform, projection

# # Read the three masks
# mask1, geotransform, projection = read_mask(file1)
# mask2, _, _ = read_mask(file2)
# mask3, _, _ = read_mask(file3)


# # Apply the inverted 'or' logic (equivalent to 'and' logic for binary masks)  - ricordiamo che il dato che vogliamo sommare ü il bordo, identificato dai pixel di valore nullo (False)
# combined_mask = np.logical_and(np.logical_and(mask1, mask2), mask3).astype(np.uint8)

# # Path for the output GeoTIFF
# output_file = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/2018_T33TXF_B2_NDVI_NDWI_predicted_mask_combined.tif'

# # Get driver for GeoTIFF
# driver = gdal.GetDriverByName('GTiff')

# # Create the output dataset
# out_dataset = driver.Create(output_file, mask1.shape[1], mask1.shape[0], 1, gdal.GDT_Byte)

# # Set the geotransform and projection on the output dataset
# out_dataset.SetGeoTransform(geotransform)
# out_dataset.SetProjection(projection)

# # Write the combined mask to the output dataset
# out_band = out_dataset.GetRasterBand(1)
# out_band.WriteArray(combined_mask)

# # Close the output dataset
# out_dataset = None

# print(f"Combined mask written to {output_file}")


from osgeo import gdal
import numpy as np
import glob
import os

def combine_predicted_masks(predicted_mask_dir, output_file):
    """
    Combine all remosaicked predicted masks into a unique binary mask using logical AND.

    Parameters:
    - predicted_mask_dir (str): Directory containing remosaicked predicted masks.
    - output_file (str): Path to save the combined binary mask GeoTIFF.

    Returns:
    - None
    """

    # List all remosaicked predicted masks in the directory
    predicted_mask_files = glob.glob(os.path.join(predicted_mask_dir, "*_predicted_full.tif"))

    if not predicted_mask_files:
        print(f"No remosaicked predicted masks found in {predicted_mask_dir}.")
        return

    print(f"Found {len(predicted_mask_files)} predicted mask files to combine.")

    # Initialize the combined mask with the first mask
    first_mask_path = predicted_mask_files[0]
    dataset = gdal.Open(first_mask_path)

    if dataset is None:
        print(f"Error: Could not open the file {first_mask_path}")
        return

    combined_mask = dataset.GetRasterBand(1).ReadAsArray()
    if combined_mask is None:
        print(f"Error: Could not read the data from the file {first_mask_path}")
        return

    combined_mask = combined_mask.astype(bool)
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # Iterate over remaining masks and apply logical AND
    for mask_file in predicted_mask_files[1:]:
        print(f"Processing mask file: {mask_file}")
        mask_dataset = gdal.Open(mask_file)
        
        if mask_dataset is None:
            print(f"Error: Could not open the file {mask_file}")
            continue

        mask_data = mask_dataset.GetRasterBand(1).ReadAsArray()
        if mask_data is None:
            print(f"Error: Could not read the data from the file {mask_file}")
            continue
        
        mask_data = mask_data.astype(bool)
        combined_mask = np.logical_and(combined_mask, mask_data)

    # Convert combined mask to uint8 format
    combined_mask = combined_mask.astype(np.uint8)

    # Get driver for GeoTIFF
    driver = gdal.GetDriverByName('GTiff')

    # Create the output dataset
    out_dataset = driver.Create(output_file, combined_mask.shape[1], combined_mask.shape[0], 1, gdal.GDT_Byte)

    # Set the geotransform and projection on the output dataset
    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)

    # Write the combined mask to the output dataset
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(combined_mask)

    # Close the output dataset
    out_dataset = None

    print(f"Combined mask written to {output_file}")
