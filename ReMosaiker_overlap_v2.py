"""
ReMosaiker Script for Sentinel-2 Crop Segmentation (v2)

This script reconstructs a full image from **sub-tiles with overlap**,  
produced during the segmentation pipeline. The **final output is georeferenced**,  
ensuring that the reconstructed image aligns with the original Sentinel-2 L2A data.

Key Features:
- **Reconstructs full images** from segmented sub-tiles.
- **Applies georeferencing** using metadata from the original GeoTIFF.
- Supports **multi-channel** images (e.g., B2, NDVI, NDWI).
- Uses **LZW compression** for optimized GeoTIFF storage.
- **Handles edge cases**, ensuring alignment with the original image size.

Differences from the Previous Version:
- **Georeferencing Enhancement:** The reconstructed mask is aligned using metadata  
  from the same image used for prediction.
- **Improved Filename Handling:** Extracts tile indices more reliably.
- **Better Error Handling:** Skips incorrectly formatted filenames.
- **Optimized Memory Management:** Ensures efficient array allocation.

Quick User Guide:
1. Set `subtiles_folder` to the directory containing segmented sub-tiles.
2. Specify the path to the **original GeoTIFF** (`original_geotiff`).
3. Set the output file path (`output_file`).
4. Run the script:
       python ReMosaiker_overlap_v2.py
5. The reconstructed image will be saved as a **georeferenced GeoTIFF**.

Dependencies:
Python packages: numpy, gdal (from osgeo), glob, os

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

"""


# # -*- coding: utf-8 -*-
# """
# Created on Tue May 14 10:02:30 2024

# @author: Alvise Ferrari

# 'ReMosaiker_wOverlap_v2.py' differs from version 1 for:
    
#     - The final reconstructed mask is georeferenced on the base of the same image that has been used for prediction of mask_subtiles.
    
# """

# import os
# import glob
# from osgeo import gdal, osr
# import numpy as np

# def create_folder(folder_path):
#     """Create a folder if it does not exist."""
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

# def read_geotiff(input_path):
#     """
#     Read a GeoTIFF file and return its array, geotransform, and projection.

#     Parameters:
#     - input_path (str): Path to the GeoTIFF file.

#     Returns:
#     - array (numpy.ndarray): Array of image data.
#     - geotransform (tuple): Geotransform data of the GeoTIFF.
#     - projection (str): Projection information of the GeoTIFF.
#     """
#     dataset = gdal.Open(input_path)
#     if dataset is None:
#         raise IOError(f"Unable to open {input_path}")
#     band = dataset.GetRasterBand(1)
#     array = band.ReadAsArray()
#     geotransform = dataset.GetGeoTransform()
#     projection = dataset.GetProjection()
#     return array, geotransform, projection

# def write_geotiff(output_path, array, geotransform, projection, num_channels, dtype=gdal.GDT_Float32):
#     """
#     Write a numpy array to a GeoTIFF file.

#     Parameters:
#     - output_path (str): Path to save the GeoTIFF file.
#     - array (numpy.ndarray): Array of image data.
#     - geotransform (tuple): Geotransform data to assign to the GeoTIFF.
#     - projection (str): Projection information to assign to the GeoTIFF.
#     - num_channels (int): Number of channels in the image.
#     - dtype: Data type of the output image.
#     """
#     driver = gdal.GetDriverByName('GTiff')
#     rows, cols, channels = array.shape
#     options = ['COMPRESS=LZW']
#     dataset = driver.Create(output_path, cols, rows, num_channels, dtype, options=options)
#     if dataset is None:
#         raise IOError(f"Could not create {output_path}")
#     dataset.SetGeoTransform(geotransform)
#     dataset.SetProjection(projection)
#     for channel in range(num_channels):
#         band = dataset.GetRasterBand(channel + 1)
#         band.WriteArray(array[:, :, channel])
#     dataset.FlushCache()
#     dataset = None  # Close the dataset to ensure CRS info is saved

# # def reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file, original_geotiff):
# #     """
# #     Reconstruct a full image from sub-tiles with overlap.

# #     Parameters:
# #     - subtiles_folder (str): Directory containing the sub-tiles.
# #     - tile_size (int): Size of each tile in pixels.
# #     - overlap_size (int): Overlap size between tiles in pixels.
# #     - output_file (str): Path to save the reconstructed GeoTIFF.
# #     - original_geotiff (str): Path to the original GeoTIFF for geotransform and projection.
# #     """
# #     # Read the original geotiff to get geotransform and projection
# #     _, geotransform, projection = read_geotiff(original_geotiff)

# #     # Find all subtiles in the specified folder
# #     subtiles_files = glob.glob(os.path.join(subtiles_folder, "*.tif"))
# #     if not subtiles_files:
# #         print("Error: No subtiles found in the specified folder")
# #         return

# #     # Get the indices of subtiles
# #     subtiles_indices = [os.path.splitext(os.path.basename(file))[0].split("_")[4:] for file in subtiles_files]  #Supponendo che le subtile da rimosaicare sono chiamate come: 'predicted_tilename_data_subtile_xx_yy'
# #     subtiles_indices = np.array(subtiles_indices, dtype=int)

# #     # Calculate the number of rows and columns in the final image
# #     num_rows_tiles = subtiles_indices[:, 0].max() + 1
# #     num_cols_tiles = subtiles_indices[:, 1].max() + 1

# #     # Calculate the total dimensions of the final image considering overlaps
# #     num_rows = num_rows_tiles * (tile_size - overlap_size) + overlap_size
# #     num_cols = num_cols_tiles * (tile_size - overlap_size) + overlap_size

# #     # Create an empty array to store the reconstructed image
# #     num_channels = gdal.Open(subtiles_files[0]).RasterCount
# #     reconstructed_image = np.zeros((num_rows, num_cols, num_channels), dtype=np.float32)

# #     # Reconstruct the image from subtiles
# #     for subtiles_file in subtiles_files:
# #         filename = os.path.splitext(os.path.basename(subtiles_file))[0]
# #         indices = filename.split("_")[4:] #Supponendo che le subtile da rimosaicare sono chiamate come: 'predicted_tilename_data_subtile_xx_yy'
# #         i, j = map(int, indices)

# #         start_row = i * (tile_size - overlap_size)
# #         start_col = j * (tile_size - overlap_size)
# #         end_row = start_row + tile_size
# #         end_col = start_col + tile_size

# #         if i == num_rows_tiles - 1:
# #             end_row = num_rows
# #         if j == num_cols_tiles - 1:
# #             end_col = num_cols

# #         subtiles_dataset = gdal.Open(subtiles_file)
# #         for channel in range(num_channels):
# #             band_data = subtiles_dataset.GetRasterBand(channel + 1).ReadAsArray()
# #             reconstructed_image[start_row:end_row, start_col:end_col, channel] = band_data

# #     # Write the reconstructed image to the output file with georeferencing
# #     write_geotiff(output_file, reconstructed_image, geotransform, projection, num_channels)

# #     print("Image reconstruction and georeferencing completed!")


# def reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file, original_geotiff):
#     """
#     Reconstruct a full image from sub-tiles with overlap.

#     Parameters:
#     - subtiles_folder (str): Directory containing the sub-tiles.
#     - tile_size (int): Size of each tile in pixels.
#     - overlap_size (int): Overlap size between tiles in pixels.
#     - output_file (str): Path to save the reconstructed GeoTIFF.
#     - original_geotiff (str): Path to the original GeoTIFF for geotransform and projection.
#     """
#     # Read the original geotiff to get geotransform and projection
#     _, geotransform, projection = read_geotiff(original_geotiff)

#     # Find all subtiles in the specified folder
#     subtiles_files = glob.glob(os.path.join(subtiles_folder, "*.tif"))
#     if not subtiles_files:
#         print("Error: No subtiles found in the specified folder")
#         return

#     # Initialize list to store indices
#     subtiles_indices = []

#     # Corrected: Get the indices of subtiles
#     for file in subtiles_files:
#         filename = os.path.splitext(os.path.basename(file))[0]
#         parts = filename.split("_")
#         try:
#             # Extract indices based on the known filename structure
#             i = int(parts[-2])  # Row index
#             j = int(parts[-1])  # Column index
#             subtiles_indices.append((i, j))
#         except ValueError:
#             print(f"Warning: Filename {filename} does not follow expected format. Skipping.")
#             continue

#     if not subtiles_indices:
#         print("Error: No valid subtiles indices found. Cannot reconstruct image.")
#         return

#     subtiles_indices = np.array(subtiles_indices)

#     # Calculate the number of rows and columns in the final image
#     num_rows_tiles = subtiles_indices[:, 0].max() + 1
#     num_cols_tiles = subtiles_indices[:, 1].max() + 1

#     # Calculate the total dimensions of the final image considering overlaps
#     num_rows = num_rows_tiles * (tile_size - overlap_size) + overlap_size
#     num_cols = num_cols_tiles * (tile_size - overlap_size) + overlap_size

#     # Create an empty array to store the reconstructed image
#     num_channels = gdal.Open(subtiles_files[0]).RasterCount
#     reconstructed_image = np.zeros((num_rows, num_cols, num_channels), dtype=np.float32)

#     # Reconstruct the image from subtiles
#     for subtiles_file in subtiles_files:
#         filename = os.path.splitext(os.path.basename(subtiles_file))[0]
#         parts = filename.split("_")
#         i = int(parts[-2])  # Row index
#         j = int(parts[-1])  # Column index

#         start_row = i * (tile_size - overlap_size)
#         start_col = j * (tile_size - overlap_size)
#         end_row = start_row + tile_size
#         end_col = start_col + tile_size

#         if i == num_rows_tiles - 1:
#             end_row = num_rows
#         if j == num_cols_tiles - 1:
#             end_col = num_cols

#         subtiles_dataset = gdal.Open(subtiles_file)
#         for channel in range(num_channels):
#             band_data = subtiles_dataset.GetRasterBand(channel + 1).ReadAsArray()
#             reconstructed_image[start_row:end_row, start_col:end_col, channel] = band_data

#     # Write the reconstructed image to the output file with georeferencing
#     write_geotiff(output_file, reconstructed_image, geotransform, projection, num_channels)

#     print(f"Image reconstruction and georeferencing completed for: {output_file}")


# # # Example usage:
# # subtiles_folder = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180719T094031_20180719T094430_T33TXF_B2_NDVI_NDWI_33TXF_20180719_tiles_woverlap/predicted_masks'
# # original_geotiff = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180719T094031_20180719T094430_T33TXF_B2_NDVI_NDWI.tif'
# # output_file = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180719T094031_20180719T094430_T33TXF_B2_NDVI_NDWI_predicted_mask.tif'
# # tile_size = 256
# # overlap_size = 32
# # reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file, original_geotiff)


# # os.path.isfile(output_file)

import os
import glob
from osgeo import gdal, osr
import numpy as np

def create_folder(folder_path):
    """Create a folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def read_geotiff(input_path):
    """
    Read a GeoTIFF file and return its array, geotransform, and projection.
    """
    dataset = gdal.Open(input_path)
    if dataset is None:
        raise IOError(f"Unable to open {input_path}")
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    return array, geotransform, projection

def write_geotiff(output_path, array, geotransform, projection, num_channels, dtype=gdal.GDT_Float32):
    """
    Write a numpy array to a GeoTIFF file.
    """
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
    """
    Reconstruct a full image from sub-tiles with overlap.
    """
    # Read the original geotiff to get geotransform and projection
    _, geotransform, projection = read_geotiff(original_geotiff)
    dataset = gdal.Open(original_geotiff)
    num_rows = dataset.RasterYSize
    num_cols = dataset.RasterXSize
    dataset = None  # Close the dataset

    # Find all subtiles in the specified folder
    subtiles_files = glob.glob(os.path.join(subtiles_folder, "*.tif"))
    if not subtiles_files:
        print("Error: No subtiles found in the specified folder")
        return

    # Get number of channels from one of the predicted subtiles
    sample_subtile = gdal.Open(subtiles_files[0])
    num_channels = sample_subtile.RasterCount
    sample_subtile = None  # Close the dataset

    # Create an empty array to store the reconstructed image
    reconstructed_image = np.zeros((num_rows, num_cols, num_channels), dtype=np.float32)

    # Process each subtile
    for subtiles_file in subtiles_files:
        filename = os.path.splitext(os.path.basename(subtiles_file))[0]
        parts = filename.split("_")
        try:
            i = int(parts[-2])  # Row index
            j = int(parts[-1])  # Column index
        except ValueError:
            print(f"Warning: Filename {filename} does not follow expected format. Skipping.")
            continue

        start_row = i * (tile_size - overlap_size)
        start_col = j * (tile_size - overlap_size)
        end_row = start_row + tile_size
        end_col = start_col + tile_size

        # Adjust end_row and end_col if they exceed image dimensions
        write_end_row = min(end_row, num_rows)
        write_end_col = min(end_col, num_cols)
        write_height = write_end_row - start_row
        write_width = write_end_col - start_col

        subtiles_dataset = gdal.Open(subtiles_file)

        for channel in range(num_channels):
            band_data = subtiles_dataset.GetRasterBand(channel + 1).ReadAsArray()
            # Extract the valid portion of the subtile
            band_data = band_data[:write_height, :write_width]
            # Place the valid data into the reconstructed image
            reconstructed_image[start_row:write_end_row, start_col:write_end_col, channel] = band_data

        subtiles_dataset = None  # Close the subtile dataset

    # Write the reconstructed image to the output file with georeferencing
    write_geotiff(output_file, reconstructed_image, geotransform, projection, num_channels)

    print(f"Image reconstruction and georeferencing completed for: {output_file}")

# Example usage:
def main():
    subtiles_folder = '/path/to/your/subtiles_folder'  # Replace with your subtiles directory
    original_geotiff = '/path/to/your/original_image.tif'  # Replace with the path to your original GeoTIFF
    output_file = '/path/to/your/output_image.tif'  # Replace with your desired output file path
    tile_size = 256
    overlap_size = 32
    reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file, original_geotiff)

if __name__ == "__main__":
    main()
