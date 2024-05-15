# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:34:29 2024

@author: Alvise Ferrari

Una volta ottenuto l'output unet e rimosaicata l'immagine integrale, prima di applicare l'algoritmo watershed è necessario effettuare alcune operazioni morfologiche per ripulire
la maschera binaria da elementi isolati assimilabili a rumore e non bordi effettivi.

Questo script è un estratto dello script 'canny_cleaner_v3.py', e parte direttamente dalla maschera binaria (a valori 0-1 )per applicare 'opening' ed il 'remove_small_objects'.

Input:
    Maschera binaria output unet dove 0==bordo, 1==non-bordo.
Output:
    Maschera binaria 0-255 ripulita da elementi indesiderati.
"""




import os
from matplotlib import pyplot as plt
from PIL import Image
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import skimage
from scipy import ndimage
import matplotlib
import numpy as np
import argparse
import imutils
import cv2 as cv
from osgeo import gdal
from skimage.morphology import binary_dilation, disk, closing, opening, remove_small_objects, remove_small_holes, thin


def read_geotiff(input_path):
    dataset = gdal.Open(input_path)
    if dataset is None:
        raise IOError(f"Unable to open {input_path}")
    
    # Get raster data
    band = dataset.GetRasterBand(1)  # assuming you need the first band
    array = band.ReadAsArray()
    
    # Get geotransform and projection
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    
    return array, geotransform, projection



def write_geotiff(output_path, array, geotransform, projection):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    #Define creation options for LZW compression
    options = ['COMPRESS=LZW']
    
    dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte, options=options)
    
    if dataset is None:
        raise IOError(f"Could not create {output_path}")
    
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.





# %% INPUT
# Set the PROJ_LIB environment variable
os.environ['PROJ_LIB'] = 'C:\\Program Files\\envs\\gisenv\\Library\\share\\proj'

# Define the base directory
base_dir = r'D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021'
os.chdir(base_dir)
# Define the filename
name = "IMG_IOWA_15TWG_20211018_predicted_mask.tif"


# Generate the full path
ee_canny_edge_detection_path = os.path.join(base_dir, name)


# Reading the GeoTIFF
canny, geotransform, projection = read_geotiff(ee_canny_edge_detection_path)




# %% TRASFORMAZIONI MORFOLOGICHE


################################################################################################################################################


#Opening
#The morphological opening of an image is defined as an erosion followed by a dilation. Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks. This tends to “open” up (dark) gaps between (bright) features.

footprint_opening  = disk(2.5)
opened_thresh = opening(canny, footprint_opening )
output_path_closed = name.replace('.tif', '_thresh_opened.tif')
write_geotiff(output_path_closed, opened_thresh, geotransform, projection) 


#REMOVE SMALL OBJECTS

#riempiamo i buchi bianchi della maschera bianchi (attenzione rimuove anche contorni...da rivedere)
# skimage.morphology.remove_small_objects(ar, min_size=64, connectivity=1, *, out=None)[source]
# Remove objects smaller than the specified size.
# Expects ar to be an array with labeled objects, and removes objects smaller than min_size. If ar is bool, the image is first labeled. This leads to potentially different behavior for bool and 0-and-1 arrays.

thresh_filled=remove_small_objects(opened_thresh.astype(bool),min_size=200,connectivity=2).astype('uint8')
output_path_filled = name.replace('.tif', '_thresh_opened_filledobj50p.tif')
write_geotiff(output_path_filled, thresh_filled, geotransform, projection)



# thresh_filled_objrem=remove_small_holes(thresh_filled.astype(bool),area_threshold=50,connectivity=2).astype('uint8')
# output_path_filled_objrem = name.replace('.tif', '_thresh_opened_filledobj50p_objrem50p.tif')
# write_geotiff(output_path_filled, thresh_filled_objrem, geotransform, projection)

#REMOVE SMALL OBJECTS NEGATIVE(applicato al negativo per rimuovere rumore di bordi incompleti)
# Calculate the negative
thresh_filled_inv = np.copy(thresh_filled)
thresh_filled_inv[thresh_filled == 1] = 0
thresh_filled_inv[thresh_filled == 0] = 255
# output_path_filled_inv = name.replace('.tif', '_thresh_opened_filledobj50p_inv.tif')
# write_geotiff(output_path_filled_inv, thresh_filled_inv, geotransform, projection)

# Fill white holes in the negative mask of the previous one
thresh_filled_inv_objrem = remove_small_objects(thresh_filled_inv.astype(bool), min_size=80, connectivity=2)
thresh_filled_inv_objrem = thresh_filled_inv_objrem.astype('uint8')

# Calculate the final processed mask by applying the inverse operation again if needed
# Assuming you need to invert it again, otherwise adjust according to your needs
thresh_filled_objrem_inv = np.copy(thresh_filled_inv_objrem)
thresh_filled_objrem_inv[thresh_filled_inv_objrem == 1] = 0
thresh_filled_objrem_inv[thresh_filled_inv_objrem == 0] = 255
output_path_objrem_inv = name.replace('.tif', '_thresh_dil_closed_filledobj_objrem.tif')
write_geotiff(output_path_objrem_inv, thresh_filled_objrem_inv, geotransform, projection)




