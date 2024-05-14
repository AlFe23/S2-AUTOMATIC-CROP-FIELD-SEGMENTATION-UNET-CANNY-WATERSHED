# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:22:09 2024

@author: Alvise Ferrari

"""

from osgeo import gdal, ogr
import os

def polygonize_watershed_with_mask(input_raster_path, output_vector_path):
    # Open the labeled raster (watershed mask)
    src_ds = gdal.Open(input_raster_path)
    if src_ds is None:
        raise IOError(f"Could not open the input raster: {input_raster_path}")

    src_band = src_ds.GetRasterBand(1)

    # Create a mask band that ignores zero values
    mask_band = src_band.GetMaskBand()

    # Create the output vector file (Shapefile format)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(output_vector_path):
        driver.DeleteDataSource(output_vector_path)
    out_ds = driver.CreateDataSource(output_vector_path)
    out_layer = out_ds.CreateLayer(os.path.splitext(os.path.basename(output_vector_path))[0], geom_type=ogr.wkbMultiPolygon)

    # Create a field to store the unique label values (identifiers)
    new_field = ogr.FieldDefn('Label', ogr.OFTInteger)
    out_layer.CreateField(new_field)

    # Perform the polygonization (label field index is 0 since it's the first field)
    gdal.Polygonize(src_band, mask_band, out_layer, 0, [], callback=None)

    # Cleanup resources
    out_ds = None
    src_ds = None
    print(f"Polygonized watershed segmentation saved to {output_vector_path}")

# Specify paths to your watershed mask raster and the output shapefile
input_raster = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh_dil_closed_thinned_filledobj50p_objrem50p.tif_WS_round_1.tif"
output_shapefile = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IOWA_15TWG_canny_2020_output_polygon.shp"

# Run the polygonization function
polygonize_watershed_with_mask(input_raster, output_shapefile)

