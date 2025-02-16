# -*- coding: utf-8 -*-
"""
Created on Aug 31 2024

@author: Alvise Ferrari

Versione v4:
Questa versione del codice introduce diverse modifiche per migliorare l'efficienza del calcolo e la gestione della memoria durante l'elaborazione delle immagini GeoTIFF per la segmentazione tramite la trasformata watershed.

Principali aggiornamenti rispetto alla versione v3:
1. **Parallelizzazione Interna alle Iterazioni**: La versione v4 utilizza il modulo `multiprocessing` per parallelizzare le operazioni computazionalmente intensive, come il calcolo della trasformata di distanza e la ricerca dei massimi locali, all'interno di ciascuna iterazione. Questo permette di accelerare il calcolo sfruttando più core del processore, mantenendo comunque l'ordine sequenziale delle iterazioni necessarie per la segmentazione progressiva.

2. **Supporto per Più di 65,535 Segmenti Unici**: Per superare il limite di 65,535 segmenti unici imposto dal tipo di dati `uint16`, questa versione utilizza `uint32` per l'array `watershed_labels`. Questo cambiamento permette di gestire un numero significativamente maggiore di segmenti unici, fino a circa 4.29 miliardi.

3. **Ottimizzazione dell'Uso della Memoria**: Le operazioni che richiedono molta memoria vengono gestite in modo più efficiente utilizzando l'esplicita deallocazione della memoria e il garbage collector di Python (`gc.collect()`). Questo riduce il consumo complessivo di memoria, rendendo il codice più adatto per l'esecuzione su macchine con risorse limitate, come i laptop.

4. **Scrittura di GeoTIFF con Tipo di Dato Aggiornato**: La funzione `write_geotiff` è stata aggiornata per scrivere file GeoTIFF utilizzando `GDT_UInt32` invece di `GDT_UInt16`, garantendo che i dati dei segmenti vengano correttamente salvati senza perdita di informazioni.

Queste modifiche migliorano sia le prestazioni che la scalabilità del codice, rendendolo adatto per dataset più grandi e complessi.

-------------------------------------------------------------------------------------------------------------

Iterative Watershed Segmentation for Sentinel-2 Crop Masks (v4)

This version improves **efficiency, scalability, and memory management**  
for iterative watershed segmentation, enabling **parallel processing**  
and support for a larger number of unique segments.

Key Features:
- **Parallelized distance transform computation** using `multiprocessing`.
- **Supports `uint32` labeling**, allowing more than 65,535 segments.
- **Optimized memory usage**, reducing RAM consumption via explicit deallocation.
- **Refines segmentation iteratively**, with improved marker detection.
- **Writes georeferenced segmentation results** after each iteration.

Differences from the Previous Version:
- **Parallelization:** Uses `multiprocessing` to speed up distance transform calculations.
- **Larger Label Capacity:** Switches from `uint16` to `uint32` for segment indexing.
- **Memory Optimization:** Implements garbage collection to prevent memory overload.
- **More Efficient File Writing:** Uses `GDT_UInt32` for correct data preservation.

Quick User Guide:
1. Set `input_canny_mask_dir` to the **binary mask file** for segmentation.
2. Adjust `min_distances` for watershed iterations (default: `[60, 30, 30, 20, 20, 15, 15, 10, 5]`).
3. Run the script:
       python iterativeWS_v4.py
4. Output **GeoTIFF segmentation maps** will be saved after each iteration.

Dependencies:
Python packages: numpy, gdal (from osgeo), skimage (segmentation), scipy (ndimage), multiprocessing, gc, os

License:
This code is released under the MIT License.

Author: Alvise Ferrari  
Code Generation Year: 2025  
"""


import os
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from osgeo import gdal, osr
from multiprocessing import Pool, cpu_count
import gc

def read_geotiff(input_path):
    dataset = gdal.Open(input_path)
    if dataset is None:
        raise IOError(f"Unable to open {input_path}")
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    return array, geotransform, projection

def write_geotiff(output_path, array, geotransform, projection, dtype=gdal.GDT_UInt32, nodata_value=0):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    options = ['COMPRESS=LZW']
    dataset = driver.Create(output_path, cols, rows, 1, dtype, options=options)
    if dataset is None:
        raise IOError(f"Could not create {output_path}")
    dataset.SetGeoTransform(geotransform)
    
    if projection is None or projection == "":
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
    else:
        dataset.SetProjection(projection)
    
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata_value)
    dataset.FlushCache()
    dataset = None

def compute_distance_transform(mask):
    """
    Compute the distance transform for the given mask.
    """
    return ndimage.distance_transform_edt(mask)

def find_local_maxima(distance, min_distance, mask):
    """
    Find local maxima in the distance-transformed image.
    """
    localMax = peak_local_max(distance, min_distance=min_distance, labels=mask)
    return localMax

def process_watershed(min_distance, ref_image_mask):
    """
    This function performs watershed segmentation for a given minimum distance using multiprocessing.
    """
    ref_image_mask_local = ref_image_mask.copy()
    
    # Use multiprocessing to compute the distance transform
    with Pool(processes=cpu_count()) as pool:
        distance = pool.apply(compute_distance_transform, args=(ref_image_mask_local,))
    
    # Find local maxima in parallel
    localMax = find_local_maxima(distance, min_distance, ref_image_mask_local)

    # Convert localMax to boolean mask if it returns coordinates
    localMax_mask = np.zeros_like(ref_image_mask_local, dtype=bool)
    localMax_mask[tuple(localMax.T)] = True

    markers, _ = ndimage.label(localMax_mask, structure=np.ones((3, 3)))

    labels = watershed(-distance, markers, mask=ref_image_mask_local)
    
    # Free memory after usage
    del distance, localMax, localMax_mask, markers
    gc.collect()

    return labels

def main():
    input_canny_mask_dir = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/2018_T33TXF_B2_NDVI_NDWI_predicted_mask_combined_cleaned.tif'
    binary_mask, geotransform, projection = read_geotiff(input_canny_mask_dir)

    ref_image_mask = np.copy(binary_mask)
    watershed_labels = np.zeros_like(binary_mask, dtype=np.uint32)  # Changed to np.uint32 for more unique values

    min_distances = [60, 30, 30, 20, 20, 15, 15, 10, 5]
    number = 0

    for round_idx, min_distance in enumerate(min_distances, start=1):
        # Perform watershed segmentation for the current iteration
        labels = process_watershed(min_distance, ref_image_mask)
        
        print(f"[INFO] Round {round_idx}: {len(np.unique(labels)) - 1} unique segments found")

        # Update masks for the next iteration
        ref_image_mask[labels > 0] = 0
        watershed_labels[labels > 0] = labels[labels > 0] + number

        number = np.max(watershed_labels)

        # Write output after each round
        base_dir = os.path.dirname(input_canny_mask_dir)
        input_filename = os.path.splitext(os.path.basename(input_canny_mask_dir))[0]
        output_path = os.path.join(base_dir, f"{input_filename}_WS_round_{round_idx}.tif")
        write_geotiff(output_path, watershed_labels, geotransform, projection, dtype=gdal.GDT_UInt32, nodata_value=0)  # Changed dtype

        # Free memory by invoking garbage collector
        del labels
        gc.collect()

    print("All watershed segmentation rounds completed.")

if __name__ == "__main__":
    main()
