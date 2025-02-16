import os
import numpy as np
from osgeo import gdal
import glob

def calculate_ndvi(nir, red):
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) from NIR and Red bands.

    NDVI is calculated as (NIR - Red) / (NIR + Red) and is used to monitor vegetation health.

    Parameters:
    - nir (numpy.ndarray): Array representing the Near-Infrared (NIR) band (B8).
    - red (numpy.ndarray): Array representing the Red band (B4).

    Returns:
    - numpy.ndarray: Calculated NDVI values.
    """
    ndvi = (nir - red) / (nir + red)
    return ndvi

def calculate_ndwi(green, nir):
    """
    Calculate the Normalized Difference Water Index (NDWI) from Green and NIR bands.

    NDWI is calculated as (Green - NIR) / (Green + NIR) and is used to monitor water content in vegetation.

    Parameters:
    - green (numpy.ndarray): Array representing the Green band (B3).
    - nir (numpy.ndarray): Array representing the Near-Infrared (NIR) band (B8).

    Returns:
    - numpy.ndarray: Calculated NDWI values.
    """
    ndwi = (green - nir) / (green + nir)
    return ndwi

def scale_index_to_uint16(index):
    """
    Scale index values (such as NDVI or NDWI) to the uint16 range (0, 65535).

    This function converts index values from the range (-1, 1) to the range (0, 65535),
    suitable for saving as a GeoTIFF.

    Parameters:
    - index (numpy.ndarray): Array of index values to be scaled (e.g., NDVI or NDWI).

    Returns:
    - numpy.ndarray: Scaled index values as uint16.
    """
    scaled_index = (index * 32767.5) + 32767.5
    scaled_index_uint16 = scaled_index.astype(np.uint16)
    return scaled_index_uint16

def find_band_files(directory):
    """
    Locate .tif files for specified bands within the provided directory.

    This function searches for the bands 'B2', 'B3', 'B4', and 'B8' in the given directory.

    Parameters:
    - directory (str): Path to the directory containing the .tif files.

    Returns:
    - dict: A dictionary with band names as keys and corresponding file paths as values.

    Raises:
    - FileNotFoundError: If any specified band file is not found.
    """
    band_files = {}
    bands = ['B2', 'B3', 'B4', 'B8']
    
    # Search for each band in the provided directory
    for band in bands:
        band_file_path = glob.glob(os.path.join(directory, f'SENTINEL2A_*_FRE_{band}.tif'))
        if len(band_file_path) > 0:
            band_files[band] = band_file_path[0]
        else:
            raise FileNotFoundError(f"Band {band} file not found in {directory}.")
    
    return band_files

def process_sentinel_image(directory, output_dir):
    """
    Process a Sentinel-2 directory: Calculate NDVI, NDWI, normalize, and save as a GeoTIFF.

    This function reads the specified bands, calculates NDVI and NDWI indices, 
    scales them to uint16, and saves a 3-band GeoTIFF containing B2, NDVI, and NDWI.

    Parameters:
    - directory (str): Path to the directory containing the .tif files.
    - output_dir (str): Directory where the processed GeoTIFF should be saved.

    Returns:
    - None
    """
    # Ensure that we are working with the correct directory
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    # Define output file path
    base_name = os.path.basename(directory)
    output_file = os.path.join(output_dir, f"{base_name}_B2_NDVI_NDWI.tif")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    # Find band files from the root directory
    bands = find_band_files(directory)  # This will now search in the root directory

    # Open the bands using GDAL
    b2 = gdal.Open(bands['B2']).ReadAsArray().astype(np.float32)
    b3 = gdal.Open(bands['B3']).ReadAsArray().astype(np.float32)
    b4 = gdal.Open(bands['B4']).ReadAsArray().astype(np.float32)
    b8 = gdal.Open(bands['B8']).ReadAsArray().astype(np.float32)

    # Ensure arrays are non-empty and have valid shapes
    if b2 is None or b3 is None or b4 is None or b8 is None:
        raise ValueError("One or more input bands are invalid (empty or None).")

    # Calculate NDVI and NDWI
    ndvi = calculate_ndvi(b8, b4)
    ndwi = calculate_ndwi(b3, b8)

    # Scale NDVI and NDWI to uint16
    scaled_ndvi = scale_index_to_uint16(ndvi)
    scaled_ndwi = scale_index_to_uint16(ndwi)

    # Get GeoTransform and Projection from one of the input bands
    dataset = gdal.Open(bands['B2'])
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # Create output GeoTIFF file with 3 bands: B2, NDVI, NDWI
    driver = gdal.GetDriverByName('GTiff')
    if driver is None:
        raise RuntimeError("GDAL driver for GeoTIFF is not available.")
    
    out_raster = driver.Create(output_file, b2.shape[1], b2.shape[0], 3, gdal.GDT_UInt16)
    
    # If the output raster could not be created, raise an error
    if out_raster is None:
        raise RuntimeError(f"Failed to create the output file {output_file}.")
    
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(projection)
    
    out_raster.GetRasterBand(1).WriteArray(b2.astype(np.uint16))
    out_raster.GetRasterBand(2).WriteArray(scaled_ndvi)
    out_raster.GetRasterBand(3).WriteArray(scaled_ndwi)

    # Close the dataset
    out_raster.FlushCache()
    out_raster = None
    print(f"Processed and saved: {output_file}")

def process_directory(input_dir, output_dir):
    """
    Process a single Sentinel-2 directory: Calculate NDVI, NDWI, and save as a GeoTIFF.

    This function processes the root directory (not subdirectories), and saves the 
    processed results in the specified output directory.

    Parameters:
    - input_dir (str): Directory containing Sentinel-2 data.
    - output_dir (str): Directory where processed GeoTIFFs should be saved.

    Returns:
    - None
    """
    # This will only process the root directory, not any subfolders.
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"The input directory {input_dir} does not exist.")
    
    # Process the root directory
    process_sentinel_image(input_dir, output_dir)

# Example usage
input_dir = '/mnt/h/S2_THEIA_test/SENTINEL2B_20210520-105858-515_L2A_T31TCJ_C_V3-0'
output_dir = '/mnt/h/S2_THEIA_test/Processed'
process_directory(input_dir, output_dir)
