"""
Subtiling Script for Sentinel-2 GeoTIFFs (Original Version)

This script extracts **256x256 sub-tiles with overlap** from Sentinel-2 **10m resolution L2A images**.  
It is designed to preprocess input data for deep learning models by ensuring that each tile has a  
consistent size and includes partial overlap to improve segmentation accuracy.

Key Features:
- Extracts sub-tiles from **GeoTIFF images** while maintaining georeferencing.
- Uses **fixed overlap size** to avoid boundary artifacts.
- Saves sub-tiles in a dedicated folder named `<input_filename>_tiles_woverlap`.
- Processes all `.tif` files within the specified input directory.
- Automatically creates necessary output directories.

Quick User Guide:
1. Modify `input_dir` and `output_dir` in the script to match your dataset.
2. Adjust `tile_size` (default: 256) and `overlap_size` (default: 32) if needed.
3. Run the script:
       python subtiler_wOverlap_auto.py
4. The processed sub-tiles will be stored in `<output_dir>/<filename>_tiles_woverlap/`.

Dependencies:
Python packages: numpy, gdal (from osgeo), os

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

"""



# import os
# from osgeo import gdal
# import numpy as np

# def create_folder(folder_path):
#     """Create a folder if it does not exist."""
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

# # def extract_subtiles_with_overlap(input_dir, output_dir, tile_size, overlap_size):
# #     """
# #     Extract sub-tiles with overlap from all GeoTIFF files in the input directory.



# #     This function reads each GeoTIFF file in the input directory, extracts sub-tiles with
# #     specified tile size and overlap, and saves the sub-tiles to the output directory.

# #     Parameters:
# #     - input_dir (str): Directory containing input GeoTIFF files.
# #     - output_dir (str): Directory where sub-tiles will be saved.
# #     - tile_size (int): Size of each tile (in pixels).
# #     - overlap_size (int): Size of the overlap between adjacent tiles (in pixels).

# #     Returns:
# #     - None
# #     """
# #     # Ensure output directory exists
# #     create_folder(output_dir)

# #     # List all GeoTIFF files in the input directory
# #     input_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

# #     for input_file in input_files:
# #         input_path = os.path.join(input_dir, input_file)
        
# #         # Generate prefix from the input filename (first three elements separated by underscores)
# #         filename_parts = input_file.split('_')
# #         if len(filename_parts) >= 3:
# #             prefix = '_'.join(filename_parts[:3])
# #         else:
# #             print(f"Warning: Unexpected file name structure for {input_file}. Using full filename as prefix.")
# #             prefix = os.path.splitext(input_file)[0]

# #         # Open the input GeoTIFF file
# #         dataset = gdal.Open(input_path)
# #         if dataset is None:
# #             print(f"Error: Could not open input file {input_file}")
# #             continue
        
# #         num_channels = dataset.RasterCount
# #         num_rows = dataset.RasterYSize
# #         num_cols = dataset.RasterXSize
        
# #         # Calculate the number of tiles in rows and columns considering overlap
# #         num_rows_tiles = (num_rows - overlap_size) // (tile_size - overlap_size)
# #         num_cols_tiles = (num_cols - overlap_size) // (tile_size - overlap_size)
        
# #         # Create sub-directory for each file's sub-tiles
# #         output_folder = os.path.join(output_dir, f"{prefix}_tiles_woverlap")
# #         create_folder(output_folder)
        
# #         for i in range(num_rows_tiles):
# #             for j in range(num_cols_tiles):
# #                 start_row = i * (tile_size - overlap_size)
# #                 start_col = j * (tile_size - overlap_size)
# #                 end_row = start_row + tile_size
# #                 end_col = start_col + tile_size
                
# #                 subtile_data = []
# #                 for channel in range(num_channels):
# #                     band = dataset.GetRasterBand(channel + 1)
# #                     band_data = band.ReadAsArray(start_col, start_row, tile_size, tile_size)
# #                     band_data = np.nan_to_num(band_data, nan=0.0)
# #                     subtile_data.append(band_data)
                
# #                 output_file = os.path.join(output_folder, f"{prefix}_subtile_{i}_{j}.tif")
                
# #                 driver = gdal.GetDriverByName("GTiff")
# #                 subtile_dataset = driver.Create(output_file, tile_size, tile_size, num_channels, gdal.GDT_Float32)
# #                 for channel, channel_data in enumerate(subtile_data):
# #                     subtile_dataset.GetRasterBand(channel + 1).WriteArray(channel_data)
                
# #                 subtile_dataset = None

# #         print(f"Sub-tiles extraction with overlap completed for {prefix}!")

# def extract_subtiles_with_overlap(input_dir, output_dir, tile_size, overlap_size):
#     """
#     Extract sub-tiles with overlap from all GeoTIFF files in the input directory.

#     This function reads each GeoTIFF file in the input directory, extracts sub-tiles with
#     specified tile size and overlap, and saves the sub-tiles to the output directory.

#     Parameters:
#     - input_dir (str): Directory containing input GeoTIFF files.
#     - output_dir (str): Directory where sub-tiles will be saved.
#     - tile_size (int): Size of each tile (in pixels).
#     - overlap_size (int): Size of the overlap between adjacent tiles (in pixels).

#     Returns:
#     - None
#     """
#     # Ensure output directory exists
#     create_folder(output_dir)

#     # List all GeoTIFF files in the input directory
#     input_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

#     for input_file in input_files:
#         input_path = os.path.join(input_dir, input_file)
        
#         # Generate prefix from the input filename (first three elements separated by underscores)
#         filename_parts = input_file.split('_')
#         if len(filename_parts) >= 3:
#             prefix = '_'.join(filename_parts[:3])
#         else:
#             print(f"Warning: Unexpected file name structure for {input_file}. Using full filename as prefix.")
#             prefix = os.path.splitext(input_file)[0]

#         # Open the input GeoTIFF file
#         dataset = gdal.Open(input_path)
#         if dataset is None:
#             print(f"Error: Could not open input file {input_file}")
#             continue
        
#         num_channels = dataset.RasterCount
#         num_rows = dataset.RasterYSize
#         num_cols = dataset.RasterXSize
        
#         # Calculate the number of tiles in rows and columns considering overlap
#         num_rows_tiles = (num_rows - overlap_size) // (tile_size - overlap_size)
#         num_cols_tiles = (num_cols - overlap_size) // (tile_size - overlap_size)
        
#         # Create sub-directory for each file's sub-tiles
#         output_folder = os.path.join(output_dir, f"{prefix}_tiles_woverlap")
#         create_folder(output_folder)
        
#         for i in range(num_rows_tiles):
#             for j in range(num_cols_tiles):
#                 start_row = i * (tile_size - overlap_size)
#                 start_col = j * (tile_size - overlap_size)
#                 end_row = start_row + tile_size
#                 end_col = start_col + tile_size
                
#                 subtile_data = []
#                 for channel in range(num_channels):
#                     band = dataset.GetRasterBand(channel + 1)
#                     band_data = band.ReadAsArray(start_col, start_row, tile_size, tile_size)
#                     band_data = np.nan_to_num(band_data, nan=0.0)
#                     subtile_data.append(band_data)
                
#                 # Revised consistent sub-tile naming
#                 output_file = os.path.join(output_folder, f"{prefix}_subtile_{i:02d}_{j:02d}.tif")
                
#                 driver = gdal.GetDriverByName("GTiff")
#                 subtile_dataset = driver.Create(output_file, tile_size, tile_size, num_channels, gdal.GDT_Float32)
#                 for channel, channel_data in enumerate(subtile_data):
#                     subtile_dataset.GetRasterBand(channel + 1).WriteArray(channel_data)
                
#                 subtile_dataset = None

#         print(f"Sub-tiles extraction with overlap completed for {prefix}!")

# def main():
#     """
#     Main function to run the sub-tiling process for Sentinel-2 preprocessed GeoTIFF images.
#     """
#     # Input directory containing the processed GeoTIFF files from Step 1
#     input_dir = '/mnt/h/Alvise/S2_samples'  # Replace with your actual input directory path

#     # Output directory where sub-tiles will be saved
#     output_dir = '/mnt/h/Alvise/S2_samples'  # Replace with your desired output directory path

#     # Parameters for sub-tiling
#     tile_size = 256
#     overlap_size = 32

#     # Run the sub-tiling process
#     print("Starting sub-tiling of GeoTIFF images...")
#     extract_subtiles_with_overlap(input_dir, output_dir, tile_size, overlap_size)
#     print("Sub-tiling complete. Tiles are saved in the output directory.")

# if __name__ == "__main__":
#     main()


import os
from osgeo import gdal
import numpy as np

def create_folder(folder_path):
    """Create a folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def extract_subtiles_with_overlap(input_dir, output_dir, tile_size, overlap_size):
    """
    Extract sub-tiles with overlap from all GeoTIFF files in the input directory.
    """
    # Ensure output directory exists
    create_folder(output_dir)

    # List all GeoTIFF files in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        
        # Generate prefix from the input filename
        filename_parts = input_file.split('_')
        if len(filename_parts) >= 3:
            prefix = '_'.join(filename_parts[:3])
        else:
            print(f"Warning: Unexpected file name structure for {input_file}. Using full filename as prefix.")
            prefix = os.path.splitext(input_file)[0]

        # Open the input GeoTIFF file
        dataset = gdal.Open(input_path)
        if dataset is None:
            print(f"Error: Could not open input file {input_file}")
            continue
        
        num_channels = dataset.RasterCount
        num_rows = dataset.RasterYSize
        num_cols = dataset.RasterXSize
        
        # Calculate the number of tiles in rows and columns considering overlap
        stride = tile_size - overlap_size
        num_rows_tiles = int(np.ceil((num_rows - overlap_size) / stride))
        num_cols_tiles = int(np.ceil((num_cols - overlap_size) / stride))
        
        # Create sub-directory for each file's sub-tiles
        output_folder = os.path.join(output_dir, f"{prefix}_tiles_woverlap")
        create_folder(output_folder)
        
        for i in range(num_rows_tiles):
            for j in range(num_cols_tiles):
                start_row = i * stride
                start_col = j * stride
                end_row = start_row + tile_size
                end_col = start_col + tile_size
                
                # Adjust end_row and end_col if they exceed image dimensions
                read_end_row = min(end_row, num_rows)
                read_end_col = min(end_col, num_cols)
                read_height = read_end_row - start_row
                read_width = read_end_col - start_col

                subtile_data = []
                for channel in range(num_channels):
                    band = dataset.GetRasterBand(channel + 1)
                    band_data = band.ReadAsArray(start_col, start_row, read_width, read_height)
                    band_data = np.nan_to_num(band_data, nan=0.0)
                    # Pad the sub-tile if it's smaller than tile_size
                    if read_height < tile_size or read_width < tile_size:
                        padded_data = np.zeros((tile_size, tile_size), dtype=band_data.dtype)
                        padded_data[:read_height, :read_width] = band_data
                        band_data = padded_data
                    subtile_data.append(band_data)
                
                # Revised consistent sub-tile naming
                output_file = os.path.join(output_folder, f"{prefix}_subtile_{i:02d}_{j:02d}.tif")
                
                driver = gdal.GetDriverByName("GTiff")
                subtile_dataset = driver.Create(output_file, tile_size, tile_size, num_channels, gdal.GDT_Float32)
                for channel, channel_data in enumerate(subtile_data):
                    subtile_dataset.GetRasterBand(channel + 1).WriteArray(channel_data)
                
                subtile_dataset = None

        print(f"Sub-tiles extraction with overlap completed for {prefix}!")

def main():
    """
    Main function to run the sub-tiling process for Sentinel-2 preprocessed GeoTIFF images.
    """
    # Input directory containing the processed GeoTIFF files
    input_dir = '/path/to/your/input_directory'  # Replace with your actual input directory path

    # Output directory where sub-tiles will be saved
    output_dir = '/path/to/your/output_directory'  # Replace with your desired output directory path

    # Parameters for sub-tiling
    tile_size = 256
    overlap_size = 32

    # Run the sub-tiling process
    print("Starting sub-tiling of GeoTIFF images...")
    extract_subtiles_with_overlap(input_dir, output_dir, tile_size, overlap_size)
    print("Sub-tiling complete. Tiles are saved in the output directory.")

if __name__ == "__main__":
    main()
