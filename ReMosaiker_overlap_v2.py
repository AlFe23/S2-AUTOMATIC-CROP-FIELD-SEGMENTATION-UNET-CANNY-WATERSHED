# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:02:30 2024

@author: Alvise Ferrari

'ReMosaiker_wOverlap_v2.py' differs from version 1 for:
    
    - The final reconstructed mask is georeferenced on the base of the same image that has been used for prediction of mask_subtiles.
    
"""

import os
import glob
from osgeo import gdal, osr
import numpy as np

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def read_geotiff(input_path):
    dataset = gdal.Open(input_path)
    if dataset is None:
        raise IOError(f"Unable to open {input_path}")
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    return array, geotransform, projection

def write_geotiff(output_path, array, geotransform, projection, num_channels, dtype=gdal.GDT_Float32):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols, channels = array.shape
    options = ['COMPRESS=LZW']
    dataset = driver.Create(output_path, cols, rows, num_channels, dtype, options=options)
    if dataset is None:
        raise IOError(f"Could not create {output_path}")
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    for channel in range(num_channels):
        band = dataset.GetRasterBand(channel + 1)
        band.WriteArray(array[:, :, channel])
    dataset.FlushCache()
    dataset = None  # Close the dataset to ensure CRS info is saved

def reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file, original_geotiff):
    # Read the original geotiff to get geotransform and projection
    _, geotransform, projection = read_geotiff(original_geotiff)

    # Find all subtiles in the specified folder
    subtiles_files = glob.glob(os.path.join(subtiles_folder, "*.tif"))
    if not subtiles_files:
        print("Error: No subtiles found in the specified folder")
        return

    # Get the indices of subtiles
    subtiles_indices = [os.path.splitext(os.path.basename(file))[0].split("_")[3:] for file in subtiles_files]
    subtiles_indices = np.array(subtiles_indices, dtype=int)

    # Calculate the number of rows and columns in the final image
    num_rows_tiles = subtiles_indices[:, 0].max() + 1
    num_cols_tiles = subtiles_indices[:, 1].max() + 1

    # Calculate the total dimensions of the final image considering overlaps
    num_rows = num_rows_tiles * (tile_size - overlap_size) + overlap_size
    num_cols = num_cols_tiles * (tile_size - overlap_size) + overlap_size

    # Create an empty array to store the reconstructed image
    num_channels = gdal.Open(subtiles_files[0]).RasterCount
    reconstructed_image = np.zeros((num_rows, num_cols, num_channels), dtype=np.float32)

    # Reconstruct the image from subtiles
    for subtiles_file in subtiles_files:
        filename = os.path.splitext(os.path.basename(subtiles_file))[0]
        indices = filename.split("_")[3:]
        i, j = map(int, indices)

        start_row = i * (tile_size - overlap_size)
        start_col = j * (tile_size - overlap_size)
        end_row = start_row + tile_size
        end_col = start_col + tile_size

        if i == num_rows_tiles - 1:
            end_row = num_rows
        if j == num_cols_tiles - 1:
            end_col = num_cols

        subtiles_dataset = gdal.Open(subtiles_file)
        for channel in range(num_channels):
            band_data = subtiles_dataset.GetRasterBand(channel + 1).ReadAsArray()
            reconstructed_image[start_row:end_row, start_col:end_col, channel] = band_data

    # Write the reconstructed image to the output file with georeferencing
    write_geotiff(output_file, reconstructed_image, geotransform, projection, num_channels)

    print("Image reconstruction and georeferencing completed!")

# Example usage:
subtiles_folder = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20211018_tiles_woverlap\predicted_masks"
original_geotiff = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20211018.tif"
output_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20211018_predicted_mask.tif"
tile_size = 256
overlap_size = 32
reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file, original_geotiff)
