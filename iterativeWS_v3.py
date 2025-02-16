# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:41:55 2024

@author: Alvise Ferrari

Rispetto alla v2, questa versione risolve il problema della mancanza dell'argomento 'indeces' per la funzione 'peak_local_max' di SK-Image. 
In questo caso la conversione dall'array di coordinate dei massimi locali alla maschera bolleana viene implemetata manualmente'


Iterative Watershed Segmentation for Sentinel-2 Crop Masks (v3)

This version improves upon **iterativeWS_v2.py** by fixing an issue  
with **peak detection in the `peak_local_max` function**.  
It ensures robust detection of **segmentation markers**.

Key Features:
- **Fixes `peak_local_max` argument issue**, ensuring correct marker detection.
- **Converts peak coordinates to a boolean mask** for better watershed accuracy.
- **Refines segmentation iteratively**, applying multiple watershed passes.
- **Writes georeferenced segmentation results** after each iteration.

Differences from the Previous Version:
- **Bug Fix:** Corrects `peak_local_max` implementation to avoid shape mismatch errors.
- **More Robust Marker Definition:** Converts detected peak coordinates into a boolean mask.
- **Improved Compatibility:** Ensures consistency across different versions of `skimage`.

Quick User Guide:
1. Set `input_canny_mask_dir` to the **binary mask file** for segmentation.
2. Adjust `min_distances` for watershed iterations (default: `[60, 30, 30, 20, 20, 15, 15, 10, 5]`).
3. Run the script:
       python iterativeWS_v3.py
4. Output **GeoTIFF segmentation maps** will be saved after each iteration.

Dependencies:
Python packages: numpy, gdal (from osgeo), skimage (segmentation), scipy (ndimage), cv2, os

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

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

def write_geotiff(output_path, array, geotransform, projection, dtype=gdal.GDT_UInt16, nodata_value=0):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    options = ['COMPRESS=LZW']
    dataset = driver.Create(output_path, cols, rows, 1, dtype, options=options)
    if dataset is None:
        raise IOError(f"Could not create {output_path}")
    dataset.SetGeoTransform(geotransform)
    
    if projection is None or projection == "":
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
    else:
        dataset.SetProjection(projection)
    
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata_value)
    dataset.FlushCache()
    dataset = None

input_canny_mask_dir = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/2018_T33TXF_B2_NDVI_NDWI_predicted_mask_combined_cleaned.tif'

binary_mask, geotransform, projection = read_geotiff(input_canny_mask_dir)

ref_image_mask = np.copy(binary_mask)
watershed_labels = np.copy(binary_mask).astype(np.uint32) * 0
number = 0

min_distances = [60, 30, 30, 20, 20, 15, 15, 10, 5]

for round_idx, min_distance in enumerate(min_distances, start=1):
    distance = ndimage.distance_transform_edt(ref_image_mask)
    localMax = peak_local_max(distance, min_distance=min_distance, labels=ref_image_mask)

    # Convert localMax to boolean mask if it returns coordinates
    if localMax.dtype == np.bool_:
        localMax_mask = localMax
    else:
        localMax_mask = np.zeros_like(ref_image_mask, dtype=bool)
        localMax_mask[tuple(localMax.T)] = True

    # Ensure localMax_mask has the same shape as distance and ref_image_mask
    if localMax_mask.shape != distance.shape:
        raise ValueError(f"Shape mismatch: localMax_mask shape {localMax_mask.shape} and distance shape {distance.shape}")

    markers, _ = ndimage.label(localMax_mask, structure=np.ones((3, 3)))

    # Ensure markers has the same shape as distance and ref_image_mask
    if markers.shape != distance.shape:
        raise ValueError(f"Shape mismatch: markers shape {markers.shape} and distance shape {distance.shape}")

    labels = watershed(-distance, markers, mask=ref_image_mask)
    
    print(f"[INFO] Round {round_idx}: {len(np.unique(labels)) - 1} unique segments found")

    ref_image_mask[labels > 0] = 0
    watershed_labels[labels > 0] = labels[labels > 0] + number

    number = np.max(watershed_labels)

    base_dir = os.path.dirname(input_canny_mask_dir)
    input_filename = os.path.splitext(os.path.basename(input_canny_mask_dir))[0]
    output_path = os.path.join(base_dir, f"{input_filename}_WS_round_{round_idx}.tif")
    write_geotiff(output_path, watershed_labels, geotransform, projection, nodata_value=0)

print("All watershed segmentation rounds completed.")
