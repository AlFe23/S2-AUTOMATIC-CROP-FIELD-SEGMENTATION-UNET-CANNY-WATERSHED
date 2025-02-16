# -*- coding: utf-8 -*-
"""
Created on Mon May 13 07:30:48 2024

@author: Alvise Ferrari

Una volta ottenute le Maschere multitemporali di Canny, è necessario binarizzarle prima di poterle utilizzare per l'addestramento della UNet'

Questo script è un estratto dello script 'canny_cleaner_v3.py', solo la prima parter dove viene effettuato il thresholding è mantenuta in questa parte.

Input:
    Maschera Canny Multitemporale uint8 in scala di grigi
Output:
    Maschera di Canny binarizzata (0 == bordo ; 255 == non bordo)
"""

# %% Import Libraries
from time import time
t_start = time()



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
#import imutils
import cv2 as cv
from osgeo import gdal
from skimage.morphology import remove_small_objects,remove_small_holes
from skimage.morphology import binary_dilation, disk
from skimage.morphology import closing
from skimage.morphology import binary_dilation, disk, closing, remove_small_objects, remove_small_holes, thin


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
#os.environ['PROJ_LIB'] = 'C:\\Program Files\\envs\\gisenv\\Library\\share\\proj'

# Define the base directory
#base_dir = 'dataset_gee'# r'D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021'
#os.chdir(base_dir)
# Define the filename
#name = "IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized.tif"


# Generate the full path
# ee_canny_edge_detection_path = '/mnt/ssd3/unet/dataset_gee/ALBURY/ALBURY_50SMC_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized.tif'
# ee_canny_edge_detection_path = '/mnt/ssd3/unet/dataset_gee/CORDOBA/Cordoba_20JML_canny_verano1920_NDVIth025_sigma1dot5_NDWIth025_sigmaNDWI1dot5_optimized.tif'
# ee_canny_edge_detection_path = '/mnt/ssd3/unet/dataset_gee/NORWICH/NORWICH_30UYD_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized.tif'
# ee_canny_edge_detection_path = '/mnt/ssd3/unet/dataset_gee/IOWA/2020/IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized.tif'
# ee_canny_edge_detection_path = '/mnt/ssd3/unet/dataset_gee/IOWA/2021/IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized.tif'
# ee_canny_edge_detection_path = '/mnt/ssd3/unet/dataset_gee/SUZHOU/SUZHOU_50SMC_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized.tif'
# ee_canny_edge_detection_path = 'dataset_gee_2/MODESTO/MODESTO_10SFG_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized.tif'
ee_canny_edge_detection_path = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/T33TXF_canny_2018_NDVIth020_sigma1dot7_NDWIth020_sigmaNDWI2dot5_optimized.tif'


# Reading the GeoTIFF
canny, geotransform, projection = read_geotiff(ee_canny_edge_detection_path)

# %% TRASFORMAZIONI MORFOLOGICHE

# Thresholding
thresh = cv.threshold(canny, 50, 255,cv.THRESH_BINARY_INV)[1]
thresh_norm = np.zeros(np.shape(thresh))
thresh_norm[thresh == 0] = 0
thresh_norm[thresh == 255] = 1
output_path_thresh = ee_canny_edge_detection_path.replace('.tif', '_thresh.tif')
write_geotiff(output_path_thresh, thresh, geotransform, projection)


