"""
Created on Wed May 15 15:33:32 2024

@author: Alvise Ferrari

Rispetto alla v1, questa versione aggiunge automaticamente al nome delle tile in prefisso fornito in una lista di prefissi corrispettivi alla lista delle immagini di input.
e.g:
    input_files = [
        input_img_path_1,
        input_img_path_2, 
        input_img_path_3, 
        input_img_path_4,
    ]

    prefix_name_list = ['prefix1_', 'prefix2_', 'prefix3_', 'prefix4_']
    
Una volta ottenuta la maschera binarizzata con il codice 'canny_binarizer.py', sia le immagini di input (3ch) che la maschera di output(1ch) possono essere preparate in forma di tile 256x256, 
così da diventare input idonei al training U-Net.

Questo script (new_subtiler_wOverlap.py) può ricevere in input sia le immagini a 3 canali, che quelle a 1 canale, e generare le subtile all'interno di una nuova cartella automaticamente generata
nella stessa directory dell'immagine/i di input; all'interno di questa nuova cartella le subtile verranno nominate come:
    -subtile_0_0
    -...
    -subtile_0_N
    -...
    - subtile_M_0
    -...
    - subtile_M_N
dove N è pari al numero di colonne dell'immagine diviso per 256, mentre M è pari al numero di righe dell'immagine di input diviso per 256.

Al fine di accelerare il processo di generazione di un dataset composto da diverse immagini di input, questo script accetta come input una lista di immagini, sia a 1ch che a 3ch.
"""

import os
from osgeo import gdal
import numpy as np

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def extract_subtiles_with_overlap(input_file, tile_size, overlap_size, prefix):
    # Open the input geotiff file
    dataset = gdal.Open(input_file)
    if dataset is None:
        print("Error: Could not open input file")
        return
    
    num_channels = dataset.RasterCount
    num_rows = dataset.RasterYSize
    num_cols = dataset.RasterXSize
    
    # Calculate the number of tiles in rows and columns considering overlap
    num_rows_tiles = (num_rows - overlap_size) // (tile_size - overlap_size)
    num_cols_tiles = (num_cols - overlap_size) // (tile_size - overlap_size)
    
    output_folder = os.path.splitext(input_file)[0] + "_tiles_woverlap"
    create_folder(output_folder)
    
    for i in range(num_rows_tiles):
        for j in range(num_cols_tiles):
            start_row = i * (tile_size - overlap_size)
            start_col = j * (tile_size - overlap_size)
            end_row = start_row + tile_size
            end_col = start_col + tile_size
            
            subtile_data = []
            for channel in range(num_channels):
                band = dataset.GetRasterBand(channel + 1)
                band_data = band.ReadAsArray(start_col, start_row, tile_size, tile_size)
                band_data = np.nan_to_num(band_data, nan=0.0)
                subtile_data.append(band_data)
            
            output_file = os.path.join(output_folder, f"{prefix}subtile_{i}_{j}.tif")
            
            driver = gdal.GetDriverByName("GTiff")
            subtile_dataset = driver.Create(output_file, tile_size, tile_size, num_channels, gdal.GDT_Float32)
            for channel, channel_data in enumerate(subtile_data):
                subtile_dataset.GetRasterBand(channel + 1).WriteArray(channel_data)
            
            subtile_dataset = None

    print(f"Subtiles extraction with overlap completed for {prefix.strip('_')}!")

tile_size = 256
overlap_size = 32

input_files = [
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IMG_IOWA_15TWG_20200710.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IMG_IOWA_15TWG_20200519.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IMG_IOWA_15TWG_20201008.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210526.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210615.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210814.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20210918.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20211018.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif",
    r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif",
]

prefix_name_list = ['20200710_', '20200519_', '20201008_', '20200710_', '20200519_', '20201008_', '20210526_', '20210615_', '20210814_', '20210918_', '20211018_', '20210526_', '20210615_', '20210814_', '20210918_', '20211018_' ]

# Process each file with corresponding prefix
for file, prefix in zip(input_files, prefix_name_list):
    extract_subtiles_with_overlap(file, tile_size, overlap_size, prefix)
    print(f"Processed {file} with prefix {prefix}")
