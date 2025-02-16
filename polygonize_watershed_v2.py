# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:22:09 2024

@author: Alvise

Watershed Polygonization for Sentinel-2 Crop Segmentation (v2)

This version extends `polygonize_watershed.py` by adding an **optional polygon simplification step**,  
which reduces vertex count and smooths boundaries while **preserving topology**.

Key Features:
- **Converts labeled raster masks** into vector polygons.
- **Preserves spatial reference** using the input raster’s projection.
- **Ignores background pixels** (zero values) to extract valid field boundaries.
- **Supports polygon simplification** with adjustable `simplify_tolerance` (default: 0).
- **Outputs ESRI Shapefile format** for compatibility with GIS software.

Differences from the Previous Version:
- **Polygon Simplification Added:** Allows smoothing of boundaries to reduce file size and complexity.
- **Customizable `simplify_tolerance` Parameter:** Higher values simplify more aggressively.
- **Topology Preservation:** Uses `SimplifyPreserveTopology()` to maintain valid geometries.

Quick User Guide:
1. Set `input_raster_path` to the **watershed-segmented raster**.
2. Define `output_vector_path` for the **output shapefile**.
3. Optionally set `simplify_tolerance` (default: 0, no simplification).
4. Run the script:
       python polygonize_watershed_v2.py
5. The vectorized crop field polygons will be saved as a **Shapefile (.shp)**.

Dependencies:
Python packages: gdal (from osgeo), ogr, os

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

"""

import os
from osgeo import gdal, ogr, osr

def polygonize_watershed_with_mask(
    input_raster_path,
    output_vector_path,
    simplify_tolerance=0.0
):
    """
    Converts a labeled raster into polygons, optionally simplifying geometries.

    Differences from previous version:
      - Added an optional 'simplify_tolerance' parameter that, if > 0,
        will simplify the polygons using GDAL/OGR's SimplifyPreserveTopology() method.
      - This step reduces vertex count and smooths boundaries while preserving topology.

    :param input_raster_path: Path to the input raster file containing watershed labels.
    :type input_raster_path: str
    :param output_vector_path: Path to the output vector file (Shapefile).
    :type output_vector_path: str
    :param simplify_tolerance: Tolerance distance for simplifying polygon geometries.
                               A value of 0.0 (default) means no simplification.
                               Larger values remove more vertices.
    :type simplify_tolerance: float
    """
    # Open the labeled raster (watershed mask)
    src_ds = gdal.Open(input_raster_path)
    if src_ds is None:
        raise IOError(f"Could not open the input raster: {input_raster_path}")

    src_band = src_ds.GetRasterBand(1)

    # Create a mask band that ignores zero (background) values
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
    layer_name = os.path.splitext(os.path.basename(output_vector_path))[0]
    out_layer = out_ds.CreateLayer(layer_name, srs=raster_srs, geom_type=ogr.wkbMultiPolygon)

    # Create a field to store the unique label values (identifiers)
    new_field = ogr.FieldDefn('Label', ogr.OFTInteger)
    out_layer.CreateField(new_field)

    # Perform the polygonization (label field index is 0 since it's the first field)
    gdal.Polygonize(src_band, mask_band, out_layer, 0, [], callback=None)

    # Optional: Simplify geometries if tolerance is set
    if simplify_tolerance > 0.0:
        print(f"Simplifying geometries with tolerance = {simplify_tolerance}")
        # Loop through each feature and simplify
        for feature in out_layer:
            geom = feature.GetGeometryRef()
            if geom is not None:
                # Use SimplifyPreserveTopology to help maintain overall topology
                simplified_geom = geom.SimplifyPreserveTopology(simplify_tolerance)
                feature.SetGeometry(simplified_geom)
                out_layer.SetFeature(feature)
        # Ensure layer changes are written to disk
        out_layer.SyncToDisk()

    # Cleanup resources
    out_ds = None
    src_ds = None

    print(f"Polygonized watershed segmentation saved to {output_vector_path}")
    if simplify_tolerance > 0.0:
        print("Geometries have been simplified.")
