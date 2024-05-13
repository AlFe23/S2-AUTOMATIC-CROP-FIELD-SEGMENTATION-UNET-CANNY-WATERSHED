# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:52:29 2024

@author: ferra
"""

import os
from osgeo import gdal
import numpy as np

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def extract_subtiles_with_overlap(input_file, tile_size, overlap_size):
    # Open the input geotiff file
    dataset = gdal.Open(input_file)
    if dataset is None:
        print("Error: Could not open input file")
        return
    
    num_channels = dataset.RasterCount
    num_rows = dataset.RasterYSize
    num_cols = dataset.RasterXSize
    
    # Calculate the number of tiles in rows and columns considering overlap
    num_rows_tiles = (num_rows - overlap_size) // (tile_size - overlap_size)
    num_cols_tiles = (num_cols - overlap_size) // (tile_size - overlap_size)
    
    # Create a folder to save the output tiles
    # output_folder = os.path.join(os.path.dirname(input_file), "output_tiles_with_overlap")
    # create_folder(output_folder)
    output_folder = os.path.splitext(input_file)[0] + "_tiles_woverlap"
    create_folder(output_folder)
    
    # Extract subtiles with overlap
    for i in range(num_rows_tiles):
        for j in range(num_cols_tiles):
            start_row = i * (tile_size - overlap_size)
            start_col = j * (tile_size - overlap_size)
            end_row = start_row + tile_size
            end_col = start_col + tile_size
            
            # Read data for all channels
            subtile_data = []
            for channel in range(num_channels):
                band = dataset.GetRasterBand(channel + 1)
                band_data = band.ReadAsArray(start_col, start_row, tile_size, tile_size)
                band_data = np.nan_to_num(band_data, nan=0.0)  # Replace NaN with zeros
                subtile_data.append(band_data)
            
            # Create output file name
            output_file = os.path.join(output_folder, f"subtile_{i}_{j}.tif")
            
            # Write subtile to output file
            driver = gdal.GetDriverByName("GTiff")
            subtile_dataset = driver.Create(output_file, tile_size, tile_size, num_channels, gdal.GDT_Float32)
            for channel, channel_data in enumerate(subtile_data):
                subtile_dataset.GetRasterBand(channel + 1).WriteArray(channel_data)
            
            subtile_dataset = None

    dataset = None
    print("Subtiles extraction with overlap completed!")

# Example usage
#input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\dataset_unet\Cordoba_east_20JML\verano_2019_2020\20JML_19feb2020.tif"
#input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\dataset_unet\Cordoba_east_20JML\verano_2019_2020\canny_masks\20JML_canny_verano1920_wclMask_NDVIth015_NDWIth015_sigma1dot5_6months_optimized_thresh.tif"

#IOWA 2020
# input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif"
# input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IMG_IOWA_15TWG_20200710.tif"
# input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IMG_IOWA_15TWG_20200519.tif"
# input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IMG_IOWA_15TWG_20201008.tif"

#IOWA 2021
# input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif"
# input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210526.tif"
# input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210615.tif"
# input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210814.tif"
# input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210918.tif"
# input_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20211018.tif"


# tile_size = 256
# overlap_size = 32
# extract_subtiles_with_overlap(input_file, tile_size, overlap_size)


# List of input files
input_files = [
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IMG_IOWA_15TWG_20200710.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IMG_IOWA_15TWG_20200519.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IMG_IOWA_15TWG_20201008.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210526.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210615.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210814.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210918.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20211018.tif"
]

tile_size = 256
overlap_size = 32

# Process each file
for file in input_files:
    extract_subtiles_with_overlap(file, tile_size, overlap_size)
    print(f"Processed {file}")
