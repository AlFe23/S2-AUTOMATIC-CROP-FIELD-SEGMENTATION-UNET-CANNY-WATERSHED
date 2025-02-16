# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:22:09 2024

@author: Alvise Ferrari


Watershed Polygonization for Sentinel-2 Crop Segmentation (v1)

This script converts **watershed-segmented raster masks** into **vector polygon shapefiles**,  
allowing for further spatial analysis of crop fields.

Key Features:
- **Converts labeled raster masks** into vector polygons.
- **Preserves spatial reference** using the input raster’s projection.
- **Ignores background pixels** (zero values) to extract valid field boundaries.
- **Outputs ESRI Shapefile format** for compatibility with GIS software.

Quick User Guide:
1. Set `input_raster_path` to the **watershed-segmented raster**.
2. Define `output_vector_path` for the **output shapefile**.
3. Run the script:
       python polygonize_watershed.py
4. The vectorized crop field polygons will be saved as a **Shapefile (.shp)**.

Dependencies:
Python packages: gdal (from osgeo), ogr, os

License:
This code is released under the MIT License.

Author: Alvise Ferrari  


"""

from osgeo import gdal, ogr, osr
import os

def polygonize_watershed_with_mask(input_raster_path, output_vector_path):
    # Open the labeled raster (watershed mask)
    src_ds = gdal.Open(input_raster_path)
    if src_ds is None:
        raise IOError(f"Could not open the input raster: {input_raster_path}")

    src_band = src_ds.GetRasterBand(1)

    # Create a mask band that ignores zero values
    mask_band = src_band.GetMaskBand()

    # Get the spatial reference from the input raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(src_ds.GetProjection())

    # Create the output vector file (Shapefile format)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(output_vector_path):
        driver.DeleteDataSource(output_vector_path)
    out_ds = driver.CreateDataSource(output_vector_path)

    # Create the output layer with the same spatial reference as the input raster
    out_layer = out_ds.CreateLayer(
        os.path.splitext(os.path.basename(output_vector_path))[0], 
        srs=raster_srs, 
        geom_type=ogr.wkbMultiPolygon
    )

    # Create a field to store the unique label values (identifiers)
    new_field = ogr.FieldDefn('Label', ogr.OFTInteger)
    out_layer.CreateField(new_field)

    # Perform the polygonization (label field index is 0 since it's the first field)
    gdal.Polygonize(src_band, mask_band, out_layer, 0, [], callback=None)

    # Cleanup resources
    out_ds = None
    src_ds = None
    print(f"Polygonized watershed segmentation saved to {output_vector_path}")

# # Specify paths to your watershed mask raster and the output shapefile
# input_raster = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/2018_T33TXF_B2_NDVI_NDWI_predicted_mask_combined_cleaned_WS_round_9.tif'
# output_shapefile = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/2018_T33TXF_B2_NDVI_NDWI_predicted_mask_combined_cleaned_WS_round_9.shp'

# # Run the polygonization function
# polygonize_watershed_with_mask(input_raster, output_shapefile)
