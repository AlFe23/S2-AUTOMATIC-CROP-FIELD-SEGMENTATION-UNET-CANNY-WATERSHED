#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:57:45 2024

@author: Alvise Ferrari
"""

from osgeo import gdal
import numpy as np

# Paths to the input GeoTIFF files
file1 = 'dataset_gee_2/MODESTO/20200424T184921_20200424T185418_T10SFG_B2_NDVI_NDWI_predicted_mask.tif'
file2 = 'dataset_gee_2/MODESTO/20200703T184921_20200703T185812_T10SFG_B2_NDVI_NDWI_predicted_mask.tif'
file3 = 'dataset_gee_2/MODESTO/20200812T184921_20200812T185701_T10SFG_B2_NDVI_NDWI_predicted_mask.tif'

# Function to read a GeoTIFF file and return the mask array and geotransform
def read_mask(file_path):
    dataset = gdal.Open(file_path)
    mask = dataset.GetRasterBand(1).ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    return mask, geotransform, projection

# Read the three masks
mask1, geotransform, projection = read_mask(file1)
mask2, _, _ = read_mask(file2)
mask3, _, _ = read_mask(file3)


# Apply the inverted 'or' logic (equivalent to 'and' logic for binary masks)  - ricordiamo che il dato che vogliamo sommare ü il bordo, identificato dai pixel di valore nullo (False)
combined_mask = np.logical_and(np.logical_and(mask1, mask2), mask3).astype(np.uint8)

# Path for the output GeoTIFF
output_file = 'dataset_gee_2/MODESTO/2020_T10SFG_combined_predicted_mask.tif'

# Get driver for GeoTIFF
driver = gdal.GetDriverByName('GTiff')

# Create the output dataset
out_dataset = driver.Create(output_file, mask1.shape[1], mask1.shape[0], 1, gdal.GDT_Byte)

# Set the geotransform and projection on the output dataset
out_dataset.SetGeoTransform(geotransform)
out_dataset.SetProjection(projection)

# Write the combined mask to the output dataset
out_band = out_dataset.GetRasterBand(1)
out_band.WriteArray(combined_mask)

# Close the output dataset
out_dataset = None

print(f"Combined mask written to {output_file}")
