"""
Subtiling Script for Sentinel-2 GeoTIFFs (v2)

This version extends the original subtiling script by **adding support for multiple input images**  
and **customizable file prefixing**. It enables processing datasets consisting of multiple images  
by allowing the user to assign a unique prefix to each file.

Key Features:
- Supports **batch processing** of multiple images in one execution.
- Allows **prefix assignment** for sub-tiles, improving dataset organization.
- Supports both **3-channel images** and **1-channel binary masks** (for U-Net training).
- Uses the same **256x256 tile size with overlap** approach.
- Outputs sub-tiles with a structured naming convention: `<prefix>_subtile_X_Y.tif`.

Differences from the Previous Version:
- **Multiple Image Support:** Can process a list of input files instead of just one directory.
- **Prefix-Based Naming:** Users can assign custom prefixes to sub-tiles.
- **Binary Mask Processing:** Supports **1-channel masks** alongside standard 3-channel images.
- **More Flexible File Management:** Generates sub-tiles inside a structured folder system.

Quick User Guide:
1. Define `input_files` (list of image paths) and `prefix_name_list` in the script.
2. Adjust `tile_size` (default: 256) and `overlap_size` (default: 32) if needed.
3. Run the script:
       python subtiler_wOverlap_v2.py
4. The processed sub-tiles will be saved in `<output_dir>/<prefix>_tiles_woverlap/`.

Dependencies:
Python packages: numpy, gdal (from osgeo), os

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

"""



# -*- coding: utf-8 -*-
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
    
    output_folder = os.path.splitext(input_file)[0] + f"_{prefix}_tiles_woverlap"
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
            
            output_file = os.path.join(output_folder, f"{prefix}_subtile_{i}_{j}.tif")
            
            driver = gdal.GetDriverByName("GTiff")
            subtile_dataset = driver.Create(output_file, tile_size, tile_size, num_channels, gdal.GDT_Float32)
            for channel, channel_data in enumerate(subtile_data):
                subtile_dataset.GetRasterBand(channel + 1).WriteArray(channel_data)
            
            subtile_dataset = None

    print(f"Subtiles extraction with overlap completed for {prefix.strip('_')}!")

tile_size = 256
overlap_size = 32

# ## CORDOBA INPUT

# input_files = ['/mnt/ssd3/unet/dataset_gee/CORDOBA/20191007T141051_20191007T141048_T20JML_B2_NDVI_NDWI.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/20191206T141041_20191206T142043_T20JML_B2_NDVI_NDWI.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/20200105T141041_20200105T142039_T20JML_B2_NDVI_NDWI.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/20200204T140641_20200204T141733_T20JML_B2_NDVI_NDWI.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/20200305T141041_20200305T142041_T20JML_B2_NDVI_NDWI.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/20200404T141041_20200404T141042_T20JML_B2_NDVI_NDWI.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/20200424T141051_20200424T142022_T20JML_B2_NDVI_NDWI.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/Cordoba_20JML_canny_verano1920_NDVIth025_sigma1dot5_NDWIth025_sigmaNDWI1dot5_optimized_thresh.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/Cordoba_20JML_canny_verano1920_NDVIth025_sigma1dot5_NDWIth025_sigmaNDWI1dot5_optimized_thresh.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/Cordoba_20JML_canny_verano1920_NDVIth025_sigma1dot5_NDWIth025_sigmaNDWI1dot5_optimized_thresh.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/Cordoba_20JML_canny_verano1920_NDVIth025_sigma1dot5_NDWIth025_sigmaNDWI1dot5_optimized_thresh.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/Cordoba_20JML_canny_verano1920_NDVIth025_sigma1dot5_NDWIth025_sigmaNDWI1dot5_optimized_thresh.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/Cordoba_20JML_canny_verano1920_NDVIth025_sigma1dot5_NDWIth025_sigmaNDWI1dot5_optimized_thresh.tif',
#                '/mnt/ssd3/unet/dataset_gee/CORDOBA/Cordoba_20JML_canny_verano1920_NDVIth025_sigma1dot5_NDWIth025_sigmaNDWI1dot5_optimized_thresh.tif']
               

    
# prefix_name_list = ['20JML_20191007', 
#                     '20JML_20191206',
#                     '20JML_20200105',
#                     '20JML_20200204',
#                     '20JML_20200305',
#                     '20JML_20200404',
#                     '20JML_20200424',
#                     '20JML_20191007', 
#                     '20JML_20191206',
#                     '20JML_20200105',
#                     '20JML_20200204',
#                     '20JML_20200305',
#                     '20JML_20200404',
#                     '20JML_20200424'
#                     ]


# ## ALBURI INPUT

# input_files = ['/mnt/ssd3/unet/dataset_gee/ALBURY/20210417T001101_20210417T001103_T55HDA_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/20210522T001109_20210522T001109_T55HDA_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/20210621T001109_20210621T001109_T55HDA_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/20210721T001109_20210721T001111_T55HDA_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/20210825T001111_20210825T001111_T55HDA_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/20210914T001111_20210914T001110_T55HDA_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/ALBURY_55HDA_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/ALBURY_55HDA_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/ALBURY_55HDA_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/ALBURY_55HDA_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/ALBURY_55HDA_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/ALBURY/ALBURY_55HDA_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif'
#                 ]
               

    
# prefix_name_list = ['55HDA_20210417', 
#                     '55HDA_20210522', 
#                     '55HDA_20210621', 
#                     '55HDA_20210721',
#                     '55HDA_20210825', 
#                     '55HDA_20210914', 
#                     '55HDA_20210417', 
#                     '55HDA_20210522', 
#                     '55HDA_20210621', 
#                     '55HDA_20210721',
#                     '55HDA_20210825', 
#                     '55HDA_20210914'
#                     ]

# ## NORWICH INPUT

# input_files = ['/mnt/ssd3/unet/dataset_gee/NORWICH/20200416T110621_20200416T111043_T30UYD_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/NORWICH/20200506T110621_20200506T110845_T30UYD_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/NORWICH/20200625T110631_20200625T111446_T30UYD_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/NORWICH/NORWICH_30UYD_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/NORWICH/NORWICH_30UYD_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/NORWICH/NORWICH_30UYD_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 ]
               

    
# prefix_name_list = ['30UYD_20200416', 
#                     '30UYD_20200506', 
#                     '30UYD_20200625', 
#                     '30UYD_20200416', 
#                     '30UYD_20200506', 
#                     '30UYD_20200625' 
#                     ]

# ## IOWA 2020 INPUT

# input_files = ['/mnt/ssd3/unet/dataset_gee/IOWA/2020/IMG_IOWA_15TWG_20200519.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2020/IMG_IOWA_15TWG_20200710.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2020/IMG_IOWA_15TWG_20201008.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2020/IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2020/IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2020/IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif'
#                 ]
               

    
# prefix_name_list = ['15TWG_20200519', 
#                     '15TWG_20200710', 
#                     '15TWG_20201008',
#                     '15TWG_20200519', 
#                     '15TWG_20200710', 
#                     '15TWG_20201008'
#                     ]

# ## IOWA 2021 INPUT

# input_files = ['/mnt/ssd3/unet/dataset_gee/IOWA/2021/IMG_IOWA_15TWG_20210526.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2021/IMG_IOWA_15TWG_20210615.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2021/IMG_IOWA_15TWG_20210814.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2021/IMG_IOWA_15TWG_20210918.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2021/IMG_IOWA_15TWG_20211018.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2021/IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2021/IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2021/IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2021/IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/IOWA/2021/IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh.tif'
#                 ]
               

    
# prefix_name_list = ['15TWG_20210526', 
#                     '15TWG_20210615', 
#                     '15TWG_20210814',
#                     '15TWG_20210918', 
#                     '15TWG_20211018',
#                     '15TWG_20210526', 
#                     '15TWG_20210615', 
#                     '15TWG_20210814',
#                     '15TWG_20210918', 
#                     '15TWG_20211018'
#                     ]

# ## SUZHOU 2021 INPUT

# input_files = ['/mnt/ssd3/unet/dataset_gee/SUZHOU/20210911T025551_20210911T030736_T50SMC_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/SUZHOU/20210921T025551_20210921T030738_T50SMC_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/SUZHOU/20211001T025551_20211001T030740_T50SMC_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/SUZHOU/20211110T025941_20211110T025937_T50SMC_B2_NDVI_NDWI.tif',
#                 '/mnt/ssd3/unet/dataset_gee/SUZHOU/SUZHOU_50SMC_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/SUZHOU/SUZHOU_50SMC_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/SUZHOU/SUZHOU_50SMC_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 '/mnt/ssd3/unet/dataset_gee/SUZHOU/SUZHOU_50SMC_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif'
#                 ]
               

    
# prefix_name_list = ['50SMC_20210911', 
#                     '50SMC_20210921', 
#                     '50SMC_20211001',
#                     '50SMC_20211110',
#                     '50SMC_20210911', 
#                     '50SMC_20210921', 
#                     '50SMC_20211001',
#                     '50SMC_20211110'
#                     ]



# ## PUNJAB 2019

# input_files = ['dataset_gee_2/PUNJAB/20190423T053651_20190423T054423_T43REP_B2_NDVI_NDWI.tif',
#                 'dataset_gee_2/PUNJAB/20190602T053641_20190602T054415_T43REP_B2_NDVI_NDWI.tif',
#                 'dataset_gee_2/PUNJAB/20191015T053809_20191015T054753_T43REP_B2_NDVI_NDWI.tif',
#                 'dataset_gee_2/PUNJAB/PUNJAB_43REP_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 'dataset_gee_2/PUNJAB/PUNJAB_43REP_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 'dataset_gee_2/PUNJAB/PUNJAB_43REP_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif'
#                 ]
               

    
# prefix_name_list = ['43REP_20190423', 
#                     '43REP_20190602', 
#                     '43REP_20191015',
#                     '43REP_20190423', 
#                     '43REP_20190602', 
#                     '43REP_20191015'
#                     ]    

# ## LES MANS 2021

# input_files = ['dataset_gee_2/LESMANS/20190723T105629_20190723T105718_T30UYU_B2_NDVI_NDWI.tif',
#                 'dataset_gee_2/LESMANS/20190916T105701_20190916T110551_T30UYU_B2_NDVI_NDWI.tif',
#                 'dataset_gee_2/LESMANS/20190921T105739_20190921T110218_T30UYU_B2_NDVI_NDWI.tif',
#                 'dataset_gee_2/LESMANS/LESMANS_30UYU_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 'dataset_gee_2/LESMANS/LESMANS_30UYU_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 'dataset_gee_2/LESMANS/LESMANS_30UYU_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif'
#                 ]
               

    
# prefix_name_list = ['T30UYU_20190723', 
#                     'T30UYU_20190916', 
#                     'T30UYU_20190921',
#                     'T30UYU_20190723', 
#                     'T30UYU_20190916', 
#                     'T30UYU_20190921'
#                     ]       

## MODESTO 2020

# input_files = ['dataset_gee_2/MODESTO/20200424T184921_20200424T185418_T10SFG_B2_NDVI_NDWI.tif',
#                 'dataset_gee_2/MODESTO/20200703T184921_20200703T185812_T10SFG_B2_NDVI_NDWI.tif',
#                 'dataset_gee_2/MODESTO/20200812T184921_20200812T185701_T10SFG_B2_NDVI_NDWI.tif',
#                 'dataset_gee_2/MODESTO/MODESTO_10SFG_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 'dataset_gee_2/MODESTO/MODESTO_10SFG_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif',
#                 'dataset_gee_2/MODESTO/MODESTO_10SFG_canny_2021_NDVIth025_sigma1dot5_NDWIth020_sigmaNDWI1dot5_optimized_thresh.tif'
#                 ]
               

    
# prefix_name_list = ['10SFG_20200424', 
#                     '10SFG_20200703', 
#                     '10SFG_20200812',
#                     '10SFG_20200424', 
#                     '10SFG_20200703', 
#                     '10SFG_20200812'
#                     ]     

# 33TXF 2018

input_files = ['/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180420T094031_20180420T094644_T33TXF_B2_NDVI_NDWI.tif',
                '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180525T094029_20180525T094824_T33TXF_B2_NDVI_NDWI.tif',
                '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180719T094031_20180719T094430_T33TXF_B2_NDVI_NDWI.tif',
                '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/T33TXF_canny_2018_NDVIth020_sigma1dot7_NDWIth020_sigmaNDWI2dot5_optimized_thresh.tif',
                '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/T33TXF_canny_2018_NDVIth020_sigma1dot7_NDWIth020_sigmaNDWI2dot5_optimized_thresh.tif',
                '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/T33TXF_canny_2018_NDVIth020_sigma1dot7_NDWIth020_sigmaNDWI2dot5_optimized_thresh.tif'
                ]
               

    
prefix_name_list = ['33TXF_20180420', 
                    '33TXF_20180525', 
                    '33TXF_20180719',
                    '33TXF_20180420', 
                    '33TXF_20180525', 
                    '33TXF_20180719'
                    ]          

# Process each file with corresponding prefix
for file, prefix in zip(input_files, prefix_name_list):
    extract_subtiles_with_overlap(file, tile_size, overlap_size, prefix)
    print(f"Processed {file} with prefix {prefix}")
