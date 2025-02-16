import os
import zipfile
import numpy as np
from osgeo import gdal
import glob
import shutil

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

def find_band_files(safe_path, band_numbers):
    """
    Locate .jp2 files for specified band numbers within a Sentinel-2 .SAFE directory.

    This function searches for the specified bands (e.g., B2, B3, B4, B8) in the 
    provided .SAFE directory path.

    Parameters:
    - safe_path (str): Path to the .SAFE directory containing Sentinel-2 data.
    - band_numbers (list of str): List of band numbers to locate (e.g., ['02', '03', '04', '08']).

    Returns:
    - dict: A dictionary with band numbers as keys and corresponding file paths as values.

    Raises:
    - FileNotFoundError: If any specified band file is not found.
    """
    band_files = {}
    for band in band_numbers:
        band_file_path = glob.glob(os.path.join(safe_path, f'GRANULE/*/IMG_DATA/R10m/*_B{band}_10m.jp2'))
        if len(band_file_path) > 0:
            band_files[band] = band_file_path[0]
        else:
            raise FileNotFoundError(f"Band {band} file not found in {safe_path}.")
    return band_files

def process_sentinel_image(safe_path, output_dir):
    """
    Process a Sentinel-2 .SAFE folder: Calculate NDVI, NDWI, normalize, and save as a GeoTIFF.

    This function reads the specified bands, calculates NDVI and NDWI indices, 
    scales them to uint16, and saves a 3-band GeoTIFF containing B2, NDVI, and NDWI.

    Parameters:
    - safe_path (str): Path to the .SAFE folder containing Sentinel-2 data.
    - output_dir (str): Directory where the processed GeoTIFF should be saved.

    Returns:
    - None
    """
    # Define output file path
    base_name = os.path.basename(safe_path).replace('.SAFE', '')
    output_file = os.path.join(output_dir, f"{base_name}_B2_NDVI_NDWI.tif")

    # Find band files
    bands = find_band_files(safe_path, ['02', '03', '04', '08'])
    
    # Open the bands using GDAL
    b2 = gdal.Open(bands['02']).ReadAsArray().astype(np.float32)
    b3 = gdal.Open(bands['03']).ReadAsArray().astype(np.float32)
    b4 = gdal.Open(bands['04']).ReadAsArray().astype(np.float32)
    b8 = gdal.Open(bands['08']).ReadAsArray().astype(np.float32)

    # Calculate NDVI and NDWI
    ndvi = calculate_ndvi(b8, b4)
    ndwi = calculate_ndwi(b3, b8)

    # Scale NDVI and NDWI to uint16
    scaled_ndvi = scale_index_to_uint16(ndvi)
    scaled_ndwi = scale_index_to_uint16(ndwi)

    # Get GeoTransform and Projection from one of the input bands
    dataset = gdal.Open(bands['02'])
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # Create output GeoTIFF file with 3 bands: B2, NDVI, NDWI
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(output_file, b2.shape[1], b2.shape[0], 3, gdal.GDT_UInt16)
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(projection)
    out_raster.GetRasterBand(1).WriteArray(b2.astype(np.uint16))
    out_raster.GetRasterBand(2).WriteArray(scaled_ndvi)
    out_raster.GetRasterBand(3).WriteArray(scaled_ndwi)

    # Close the dataset
    out_raster.FlushCache()
    out_raster = None
    print(f"Processed and saved: {output_file}")

def extract_and_process_zip(zip_path, extract_dir, output_dir):
    """
    Extract and process Sentinel-2 images from a zip file.

    This function extracts .SAFE folders from a given zip file, processes them to 
    generate 3-band GeoTIFFs, and then deletes the extracted folders to save space.

    Parameters:
    - zip_path (str): Path to the zip file containing Sentinel-2 data.
    - extract_dir (str): Directory where the zip file contents will be extracted.
    - output_dir (str): Directory where the processed GeoTIFFs should be saved.

    Returns:
    - None
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print(f"Extracted: {zip_path}")

    # Process each extracted .SAFE folder
    for safe_folder in os.listdir(extract_dir):
        if safe_folder.endswith('.SAFE'):
            safe_path = os.path.join(extract_dir, safe_folder)
            process_sentinel_image(safe_path, output_dir)
            
            # Clean up the .SAFE folder after processing
            shutil.rmtree(safe_path)
            print(f"Deleted extracted folder {safe_folder} after processing.")

    # Remove the extracted directory itself if it is empty
    if os.path.exists(extract_dir) and not os.listdir(extract_dir):
        shutil.rmtree(extract_dir)
        print(f"Deleted empty extraction directory {extract_dir}.")

def process_zip_directory(input_dir, output_dir):
    """
    Process all zipped Sentinel-2 images in the input directory.

    This function iterates over all zip files in the input directory, extracts and processes 
    each one to produce 3-band GeoTIFFs, and saves the results in the output directory.

    Parameters:
    - input_dir (str): Directory containing zip files of Sentinel-2 data.
    - output_dir (str): Directory where processed GeoTIFFs should be saved.

    Returns:
    - None
    """
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.zip'):
            zip_path = os.path.join(input_dir, file_name)
            extract_dir = os.path.join(input_dir, file_name.replace('.zip', ''))
            
            # Process the zip file
            extract_and_process_zip(zip_path, extract_dir, output_dir)


# # Example usage
# input_dir = '/mnt/h/Alvise/S2_samples'
# output_dir = '/mnt/h/Alvise/S2_samples'
# process_zip_directory(input_dir, output_dir)
