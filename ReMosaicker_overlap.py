import os
import glob
from osgeo import gdal
import numpy as np

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file):
    # Find all subtiles in the specified folder
    subtiles_files = glob.glob(os.path.join(subtiles_folder, "*.tif"))
    
    if not subtiles_files:
        print("Error: No subtiles found in the specified folder")
        return
    
    # Get the indices of subtiles
    subtiles_indices = [os.path.splitext(os.path.basename(file))[0].split("_")[2:] for file in subtiles_files] #sostituire con [1:]se le immagini sono chiamate come: 'subtile_xx_yy'
    subtiles_indices = np.array(subtiles_indices, dtype=int)
    
    # Calculate the number of rows and columns in the final image
    num_rows_tiles = subtiles_indices[:, 0].max() + 1
    num_cols_tiles = subtiles_indices[:, 1].max() + 1
    
    # Calculate the number of rows and columns in the final image considering overlaps
    num_rows = num_rows_tiles * (tile_size - overlap_size) + overlap_size
    num_cols = num_cols_tiles * (tile_size - overlap_size) + overlap_size
    
    # Create an empty array to store the reconstructed image
    num_channels = gdal.Open(subtiles_files[0]).RasterCount
    reconstructed_image = np.zeros((num_rows, num_cols, num_channels), dtype=np.float32)
    
    # Reconstruct the image from subtiles
    for subtiles_file in subtiles_files:
        # Extract row and column indices from the subtiles filename
        filename = os.path.splitext(os.path.basename(subtiles_file))[0]
        indices = filename.split("_")[2:] #sostituire con [1:]se le immagini sono chiamate come: 'subtile_xx_yy'
        i, j = map(int, indices)
        
        # Calculate the start and end indices for this subtiles
        start_row = i * (tile_size - overlap_size)
        start_col = j * (tile_size - overlap_size)
        end_row = start_row + tile_size
        end_col = start_col + tile_size
        
        # Adjust end indices for subtiles on the border
        if i == num_rows_tiles - 1:
            end_row = num_rows
        if j == num_cols_tiles - 1:
            end_col = num_cols
        
        # Read subtiles data
        subtiles_dataset = gdal.Open(subtiles_file)
        for channel in range(num_channels):
            band_data = subtiles_dataset.GetRasterBand(channel + 1).ReadAsArray()
            
            # Copy band_data into the reconstructed image
            reconstructed_image[start_row:end_row, start_col:end_col, channel] = band_data
        
        subtiles_dataset = None
    
    # Write reconstructed image to output file
    driver = gdal.GetDriverByName("GTiff")
    #output_dataset = driver.Create(output_file, num_cols, num_rows, num_channels, gdal.GDT_Float32)
    output_dataset = driver.Create(output_file, int(num_cols), int(num_rows), int(num_channels), gdal.GDT_Float32)

    for channel in range(num_channels):
        output_dataset.GetRasterBand(channel + 1).WriteArray(reconstructed_image[:,:,channel])
    
    output_dataset = None
    print("Image reconstruction completed!")

# # Example usage
# subtiles_folder = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\dataset_unet\Cordoba_east_20JML\verano_2019_2020\canny_masks\20JML_canny_tiles_woverlap"
# output_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\dataset_unet\Cordoba_east_20JML\verano_2019_2020\canny_masks\reconstructed_image.tif"
# tile_size = 256
# overlap_size = 64
# reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file)


# # Example usage
# subtiles_folder = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\dataset_unet\Cordoba_east_20JML\verano_2019_2020\20JML_19feb2020_tiles_woverlap"
# output_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\dataset_unet\Cordoba_east_20JML\verano_2019_2020\reconstructed_image.tif"
# tile_size = 256
# overlap_size = 64
# reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file)


subtiles_folder = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20211018_tiles_woverlap\predicted_masks"
output_file = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IMG_IOWA_15TWG_20211018_tiles_woverlap\predicted_masks\reconstructed_image.tif"
tile_size = 256
overlap_size = 32
reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file)
