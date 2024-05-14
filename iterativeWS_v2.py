# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:41:55 2024

@author: ferra
"""

import os
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import cv2 as cv
from osgeo import gdal, osr

def read_geotiff(input_path):
    dataset = gdal.Open(input_path)
    if dataset is None:
        raise IOError(f"Unable to open {input_path}")
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    return array, geotransform, projection

# def write_geotiff(output_path, array, geotransform, projection, dtype=gdal.GDT_UInt16, nodata_value=0):
#     driver = gdal.GetDriverByName('GTiff')
#     rows, cols = array.shape
#     options = ['COMPRESS=LZW']
#     dataset = driver.Create(output_path, cols, rows, 1, dtype, options=options)
#     if dataset is None:
#         raise IOError(f"Could not create {output_path}")
#     dataset.SetGeoTransform(geotransform)
#     dataset.SetProjection(projection)
#     band = dataset.GetRasterBand(1)
#     band.WriteArray(array)
#     band.SetNoDataValue(nodata_value)
#     dataset.FlushCache()
#     dataset = None  # Close the dataset to ensure CRS info is saved



def write_geotiff(output_path, array, geotransform, projection, dtype=gdal.GDT_UInt16, nodata_value=0):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    options = ['COMPRESS=LZW']
    dataset = driver.Create(output_path, cols, rows, 1, dtype, options=options)
    if dataset is None:
        raise IOError(f"Could not create {output_path}")
    dataset.SetGeoTransform(geotransform)
    
    # Check if projection information is present
    if projection is None or projection == "":
        # Set projection manually
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # Assuming EPSG 4326, change if needed
        dataset.SetProjection(srs.ExportToWkt())
    else:
        dataset.SetProjection(projection)
    
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata_value)
    dataset.FlushCache()
    dataset = None  # Close the dataset to ensure CRS info is saved



# Define the base directory and input filename
base_dir = r'D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020'
input_filename = "IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh_dil_closed_thinned_filledobj50p_objrem50p.tif"
input_canny_mask_dir = os.path.join(base_dir, input_filename)

# Reading the GeoTIFF binary mask
binary_mask, geotransform, projection = read_geotiff(input_canny_mask_dir)

# Preparing for the iterative watershed
ref_image_mask = np.copy(binary_mask)
watershed_labels = np.copy(binary_mask).astype(np.uint32) * 0
number = 0  # Initialize a counter for unique label tracking

# Define min_distance for each iterative round of watershed segmentation
min_distances = [60, 30, 30, 20, 20, 15, 15, 10]

for round_idx, min_distance in enumerate(min_distances, start=1):
    # Apply distance transform
    distance = ndimage.distance_transform_edt(ref_image_mask)

    # Find local maxima
    localMax = peak_local_max(distance, indices=False, min_distance=min_distance, labels=ref_image_mask)

    # Connected component analysis on the local peaks
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

    # Watershed algorithm
    labels = watershed(-distance, markers, mask=ref_image_mask)
    print(f"[INFO] Round {round_idx}: {len(np.unique(labels)) - 1} unique segments found")

    # Update masks and labels
    ref_image_mask[labels > 0] = 0
    watershed_labels[labels > 0] = labels[labels > 0] + number

    # Update the number for next iteration
    number = np.max(watershed_labels)

    # Write out intermediate results (optional)
    output_path = os.path.join(base_dir, f"{input_filename}_WS_round_{round_idx}.tif")
    write_geotiff(output_path, watershed_labels, geotransform, projection, nodata_value=0)

print("All watershed segmentation rounds completed.")
