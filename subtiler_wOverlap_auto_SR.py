"""
Subtiling Script for Sentinel-2 GeoTIFFs (Superresolved Version)

This version is designed for **superresolved Sentinel-2 L2A images**, allowing users  
to extract sub-tiles while selecting specific image channels. It improves efficiency  
by handling **memory allocation more effectively** and ensuring edge padding  
for incomplete sub-tiles.

Key Features:
- **Supports Superresolved Images:** Optimized for **higher resolution Sentinel-2 data**.
- **Channel Selection:** Users can specify which bands to process (e.g., `[1, 5, 6]`).
- **Memory-Optimized Processing:** Reduces unnecessary memory usage for large datasets.
- **Edge Handling:** Applies **zero-padding** to ensure all sub-tiles have uniform size.
- **Structured Folder Organization:** Saves sub-tiles in clearly named output directories.

Differences from the Previous Version:
- **Superresolution Support:** Adapts to higher resolution Sentinel-2 data.
- **Custom Channel Selection:** Users can specify specific bands instead of using all.
- **Improved Memory Management:** Reduces memory footprint during processing.
- **Enhanced Edge Handling:** Pads sub-tiles at image borders to maintain consistency.
- **Refined File Output:** Ensures better directory organization and naming.

Quick User Guide:
1. Modify `input_dir` and `output_dir` in the script.
2. Adjust `tile_size` (default: 256) and `overlap_size` (default: 32).
3. Define `channels_to_use` (e.g., `[1, 5, 6]` for RGB-NDVI-NDWI).
4. Run the script:
       python subtiler_wOverlap_auto_SR.py
5. Processed sub-tiles will be stored in `<output_dir>/<filename>_tiles_woverlap/`.

Dependencies:
Python packages: numpy, gdal (from osgeo), os

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

"""


import os
from osgeo import gdal
import numpy as np

def create_folder(folder_path):
    """Create a folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def extract_subtiles_with_overlap(input_dir, output_dir, tile_size, overlap_size, channels_to_use=None):
    """
    Extract sub-tiles with overlap from all GeoTIFF files in the input directory,
    reading only the specified channels (1-based indices).
    
    Parameters
    ----------
    input_dir : str
        Directory containing input GeoTIFF files.
    output_dir : str
        Directory where sub-tiles will be saved.
    tile_size : int
        Size of each tile (in pixels).
    overlap_size : int
        Size of the overlap between adjacent tiles (in pixels).
    channels_to_use : list of int, optional
        1-based channel indices to read from each GeoTIFF. 
        If None, read all available channels.
    """
    # Ensure output directory exists
    create_folder(output_dir)

    # List all GeoTIFF files in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        
        # Generate prefix from the input filename
        prefix = os.path.splitext(input_file)[0]


        # Open the input GeoTIFF file
        dataset = gdal.Open(input_path)
        if dataset is None:
            print(f"Error: Could not open input file {input_file}")
            continue
        
        # If no channels specified, read all channels
        if channels_to_use is None:
            channels_to_use = list(range(1, dataset.RasterCount + 1))

        # Basic info
        num_rows = dataset.RasterYSize
        num_cols = dataset.RasterXSize

        # Prepare tiling parameters
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
                
                # Adjust end_* if they exceed image dimensions
                read_end_row = min(end_row, num_rows)
                read_end_col = min(end_col, num_cols)
                read_height = read_end_row - start_row
                read_width  = read_end_col - start_col

                # Read only the channels we care about
                subtile_data = []
                for channel_idx in channels_to_use:
                    band = dataset.GetRasterBand(channel_idx)
                    band_data = band.ReadAsArray(start_col, start_row, read_width, read_height)
                    band_data = np.nan_to_num(band_data, nan=0.0)

                    # If tile is near the bottom/right edge, the tile might be smaller.
                    # Pad sub-tile to 'tile_size' if needed.
                    if (read_height < tile_size) or (read_width < tile_size):
                        padded_data = np.zeros((tile_size, tile_size), dtype=band_data.dtype)
                        padded_data[:read_height, :read_width] = band_data
                        band_data = padded_data

                    subtile_data.append(band_data)

                # Each sub-tile has shape [num_channels, tile_size, tile_size]
                # We need to write a multi-band GeoTIFF with len(channels_to_use) channels
                num_channels = len(channels_to_use)

                # Revised consistent sub-tile naming
                output_file = os.path.join(output_folder, f"{prefix}_subtile_{i:02d}_{j:02d}.tif")
                
                driver = gdal.GetDriverByName("GTiff")
                subtile_dataset = driver.Create(output_file, tile_size, tile_size, num_channels, gdal.GDT_Float32)
                for k, channel_data in enumerate(subtile_data):
                    subtile_dataset.GetRasterBand(k + 1).WriteArray(channel_data)
                
                # Close the sub-tile dataset
                subtile_dataset = None

        print(f"Sub-tiles extraction with overlap completed for {prefix}!")
        
        # Explicitly close the main dataset
        dataset = None

def main():
    """
    Main function to run the sub-tiling process.
    Modify 'input_dir', 'output_dir', 'tile_size', 'overlap_size', and 'channels_to_use' as needed.
    """
    # Input directory containing the GeoTIFF files
    input_dir = '/path/to/your/input_directory'  # Replace with your actual input directory path

    # Output directory where sub-tiles will be saved
    output_dir = '/path/to/your/output_directory'  # Replace with your desired output directory path

    # Parameters for sub-tiling
    tile_size = 256
    overlap_size = 32

    # Suppose you only want channels 1, 5, and 6 from a 6-band GeoTIFF
    channels_to_use = [1, 5, 6]

    # Run the sub-tiling process
    print("Starting sub-tiling of GeoTIFF images...")
    extract_subtiles_with_overlap(
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap_size=overlap_size,
        channels_to_use=channels_to_use
    )
    print("Sub-tiling complete. Tiles are saved in the output directory.")

if __name__ == "__main__":
    main()
