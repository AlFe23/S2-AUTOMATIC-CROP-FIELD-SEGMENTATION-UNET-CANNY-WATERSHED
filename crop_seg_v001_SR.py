
"""
CropSeg Processing Script (v001_SR)

This script is an updated version of the original crop field segmentation pipeline,  
designed to process **superresolved Sentinel-2 L2A images** at higher spatial resolution.  
It performs inference using a pre-trained **ResUNet model**, trained with the **Canny multitemporal approach**,  
similar to the methodology described in the cited paper.

Key Features (Compared to v001):
- Supports **superresolved Sentinel-2 L2A images**.
- Uses **enhanced tiling and remosaicking** for better segmentation accuracy.
- Applies **iterative watershed segmentation** to refine crop field boundaries.
- Generates **final crop field polygons** from segmentation results.

Quick User Guide:
1. Setup the Environment  
   Ensure all required dependencies (see below) are installed in your Python environment.

2. Modify Input Paths  
   Edit the `input_dir`, `output_dir`, and `model_path` variables in the script to match your dataset locations.

3. Run the Script  
   Execute the script from the command line:
       python crop_seg_v001_SR.py
   The output includes segmentation masks, watershed results, and final crop field polygons.

4. Output Files  
   - _predicted_full.tif: The fully reconstructed segmentation mask.
   - combined_predicted_mask.tif: The merged binary mask.
   - _WS_final.tif: The final watershed-segmented image.
   - .shp files: The vectorized crop field polygons.

Dependencies:
Python packages: numpy, gdal (from osgeo), glob, gc, sys, os  
External modules: s2zip_reader, subtiler_wOverlap_auto_SR, inference_v3_solopred_auto, ReMosaiker_overlap_v2,  
AND_combiner, unet_output_cleaner, iterativeWS_v4, polygonize_watershed  

Citation:
A. Ferrari, S. Saquella, G. Laneve and V. Pampanoni,  
"Automating Crop-Field Segmentation in High-Resolution Satellite Images:  
A U-Net Approach with Optimized Multitemporal Canny Edge Detection,"  
IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium,  
Athens, Greece, 2024, pp. 4094-4098, doi: 10.1109/IGARSS53475.2024.10641103.  

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

[https://github.com/AlFe23/S2-AUTOMATIC-CROP-FIELD-SEGMENTATION-UNET-CANNY-WATERSHED]
"""


import os
import glob
import numpy as np
import gc
from osgeo import gdal, osr
import s2zip_reader  
import subtiler_wOverlap_auto_SR
import inference_v3_solopred_auto
import ReMosaiker_overlap_v2
import AND_combiner
import unet_output_cleaner
import iterativeWS_v4  
import polygonize_watershed 


def main():
    """
    Main function to run the preprocessing workflow for Sentinel-2 images.

    This function sets up input and output directories, processes all zipped Sentinel-2 images,
    generates GeoTIFF files containing bands B2, NDVI, and NDWI, runs inference, reconstructs full predicted masks,
    combines them into a single binary mask, cleans the mask, and applies the Watershed algorithm iteratively.
    """
    # test_Canada_geom2_SR
    #input_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_tests/test_Canada_Geom2_test/Canada_geom2_SR'  
    #output_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_tests/test_Canada_Geom2_test/Canada_geom2_SR'  
    #model_path = '/mnt/h/Alvise/weights_unet/BFC-2024-05-16-204727/U-Net-Weights-BFCE.h5'  
    #model_path = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/weights_unet/BFC-2024-12-25-013641/U-Net-Weights-BFCE.keras'  

    # test_Canada_geom2_S2original
    #input_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_tests/test_Canada_Geom2_test/Canada_geom2_S2original'  
    #output_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_tests/test_Canada_Geom2_test/Canada_geom2_S2original'  
    #model_path = '/mnt/h/Alvise/weights_unet/BFC-2024-05-16-204727/U-Net-Weights-BFCE.h5'  
    #model_path = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/weights_unet/BFC-2024-12-25-013641/U-Net-Weights-BFCE.keras'  

    # test_China_tile11_50SMC
    #input_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_tests/test_China_tile11_50SMC/tile_11'  
    #output_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_tests/test_China_tile11_50SMC/tile_11'  
    #model_path = '/mnt/h/Alvise/weights_unet/BFC-2024-05-16-204727/U-Net-Weights-BFCE.h5' 
    #model_path = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/weights_unet/BFC-2024-12-25-013641/U-Net-Weights-BFCE.keras' 


    # test_Punjab_tile11_43REP
    #input_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_tests/test_punjab_tile11_43REP/tile_11'  
    #output_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_tests/test_punjab_tile11_43REP/tile_11'  
    #model_path = '/mnt/h/Alvise/weights_unet/BFC-2024-05-16-204727/U-Net-Weights-BFCE.h5' 
    #model_path = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/weights_unet/BFC-2024-12-25-013641/U-Net-Weights-BFCE.keras' 

    # test_Fucino
    input_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_tests/test_Fucino/SR_coreg'  
    output_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_tests/test_Fucino/SR_coreg'  
    #model_path = '/mnt/h/Alvise/weights_unet/BFC-2024-05-16-204727/U-Net-Weights-BFCE.h5' 
    model_path = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/weights_unet/BFC-2024-12-25-013641/U-Net-Weights-BFCE.keras' 

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # # Step 1: Process Sentinel-2 images and generate 3-channel GeoTIFFs
    # print("Starting preprocessing of Sentinel-2 images...")
    # s2zip_reader.process_zip_directory(input_dir, output_dir)
    # print("Preprocessing complete. GeoTIFF files are saved in the output directory.")
    
    # Step 2: Sub-tiling
    tile_size = 256
    overlap_size = 32

    print("Starting sub-tiling of GeoTIFF images...")
    subtiler_wOverlap_auto_SR.extract_subtiles_with_overlap(
    input_dir,
    output_dir,
    tile_size,
    overlap_size,
    #channels_to_use=[2, 13, 14])
    channels_to_use=[1, 5, 6])
    print("Sub-tiling complete. Tiles are saved in the output directory.")

    # Step 3: Inference
    print("Loading trained model...")
    model = inference_v3_solopred_auto.load_trained_model(model_path)
    print("Model loaded successfully.")

    # Iterate over each folder in the base directory for inference
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        if os.path.isdir(subdir_path):  # Ensure it's a directory
            print(f"Starting inference on tiles in directory: {subdir_path}")

            # Create output directory in the same directory as input, with a suffix '_predicted'
            output_directory = os.path.join(input_dir, f"{subdir}_predicted")

            # Run inference on all image tiles in the current directory
            inference_v3_solopred_auto.run_inference_on_directory(model, subdir_path, output_directory)

    print("Inference complete for all folders. Predicted masks are saved.")

    # Step 4: Reconstruct Full Predicted Masks
    for subdir in os.listdir(input_dir):
        predicted_folder = os.path.join(input_dir, subdir)

        if os.path.isdir(predicted_folder) and subdir.endswith('_tiles_woverlap_predicted'):  # Check for predicted folders
            # Extract the common prefix from the subdir name (base name without suffixes)
            prefix = subdir.split('_tiles_woverlap_predicted')[0]

            # Search for the original GeoTIFF file that matches this prefix
            matching_files = glob.glob(os.path.join(input_dir, f"{prefix}*.tif"))

            if not matching_files:
                print(f"Error: No matching original GeoTIFF found for {subdir}. Expected a file starting with {prefix}")
                continue

            original_geotiff = matching_files[0]  # Assuming the first match is the correct one

            output_file = os.path.join(input_dir, f"{prefix}_predicted_full.tif")  # Output reconstructed file

            print(f"Reconstructing full predicted mask for: {subdir}")
            ReMosaiker_overlap_v2.reconstruct_image(predicted_folder, tile_size, overlap_size, output_file, original_geotiff)

    print("All images reconstructed successfully.")

    # Step 5: Combine All Remosaicked Predicted Masks
    combined_output_file = os.path.join(output_dir, "combined_predicted_mask.tif")
    print("Combining all remosaicked predicted masks into a single binary mask...")
    AND_combiner.combine_predicted_masks(input_dir, combined_output_file)
    print("All remosaicked predicted masks have been combined successfully.")

    # Step 6: Clean the Combined Mask
    cleaned_output_file = combined_output_file.replace('.tif', '_cleaned.tif')
    print("Cleaning the combined binary mask to remove noise and small objects...")
    unet_output_cleaner.process_mask(combined_output_file, cleaned_output_file, apply_opening=True, min_size=9, area_threshold=60)
    print("Combined mask has been cleaned and saved.")

    # Step 7: Apply Watershed Algorithm Iteratively
    print("Applying Watershed segmentation to the cleaned binary mask...")

    # Read the cleaned mask and prepare for watershed segmentation
    binary_mask, geotransform, projection = ReMosaiker_overlap_v2.read_geotiff(cleaned_output_file)

    ref_image_mask = np.copy(binary_mask)
    watershed_labels = np.zeros_like(binary_mask, dtype=np.uint32)  # Changed to np.uint32 for more unique values

    min_distances = [100, 60, 40, 20, 15, 10]
    number = 0

    for round_idx, min_distance in enumerate(min_distances, start=1):
        # Perform watershed segmentation for the current iteration
        labels = iterativeWS_v4.process_watershed(min_distance, ref_image_mask)
        
        print(f"[INFO] Round {round_idx}: {len(np.unique(labels)) - 1} unique segments found")

        # Update masks for the next iteration
        ref_image_mask[labels > 0] = 0
        watershed_labels[labels > 0] = labels[labels > 0] + number

        number = np.max(watershed_labels)

        # Free memory by invoking garbage collector
        del labels
        gc.collect()
    
        # Save the final result after all iterations are complete
    base_dir = os.path.dirname(cleaned_output_file)
    input_filename = os.path.splitext(os.path.basename(cleaned_output_file))[0]
    output_path = os.path.join(base_dir, f"{input_filename}_WS_final.tif")
    iterativeWS_v4.write_geotiff(output_path, watershed_labels, geotransform, projection, dtype=gdal.GDT_UInt32)

    print("Final watershed segmentation image has been saved.")
    print("All watershed segmentation rounds completed.")
    
    # Step 8: Polygonize the final watershed mask
    print("Polygonizing the final watershed mask to create a shapefile...")

    # Define output shapefile path
    output_shapefile = output_path.replace('.tif', '.shp')

    # Perform polygonization using the imported function
    polygonize_watershed.polygonize_watershed_with_mask(output_path, output_shapefile)
    

if __name__ == "__main__":
    main()
