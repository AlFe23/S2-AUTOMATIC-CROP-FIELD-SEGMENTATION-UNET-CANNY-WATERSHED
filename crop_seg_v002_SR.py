"""
CropSeg Processing Script (v002_SR)

This script is an improved version of `crop_seg_v001_SR.py`,  
designed for inference using **pre-trained ResUNet crop field segmentation models**  
on **superresolved Sentinel-2 L2A images**.  
It incorporates additional functionalities for **small-segment removal, logging, and polygonization refinements**.

Key Features (Compared to v001_SR):
- **New Functionality:** Adds `remove_small_segments()`, which removes watershed segments smaller than a defined threshold.
- **Parameterization:** The `main()` function now supports configurable input/output paths, model paths, and an optional `min_segment_size` parameter.
- **Logging Enhancement:** Redirects all output to `processing_report.txt` for better tracking instead of printing to the console.
- **Improved Watershed Processing:** Applies small-segment removal after watershed segmentation to refine final results.
- **Updated Polygonization:** Uses `polygonize_watershed_v2.polygonize_watershed_with_mask()` instead of the previous version.

Quick User Guide:
1. Setup the Environment  
   Ensure all required dependencies (see below) are installed in your Python environment.

2. Modify Input Paths  
   Edit the `input_dir`, `output_dir`, and `model_path` variables in the script to match your dataset locations.  
   Optionally, adjust `min_segment_size` to control small-segment removal.

3. Run the Script  
   Execute the script from the command line:
       python crop_seg_v002_SR.py
   The output includes segmentation masks, watershed results, and final crop field polygons.

4. Output Files  
   - _predicted_full.tif: The fully reconstructed segmentation mask.
   - combined_predicted_mask.tif: The merged binary mask.
   - _WS_final.tif: The final watershed-segmented image.
   - .shp files: The vectorized crop field polygons.
   - processing_report.txt: The log file capturing all processing steps.

Dependencies:
Python packages: numpy, gdal (from osgeo), glob, gc, sys, os  
External modules: s2zip_reader, subtiler_wOverlap_auto_SR, inference_v3_solopred_auto, ReMosaiker_overlap_v2,  
AND_combiner, unet_output_cleaner, iterativeWS_v4, polygonize_watershed_v2  

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
import sys  # for redirecting stdout
from osgeo import gdal, osr

import s2zip_reader  
import subtiler_wOverlap_auto_SR
import inference_v3_solopred_auto
import ReMosaiker_overlap_v2
import AND_combiner
import unet_output_cleaner
import iterativeWS_v4  
import polygonize_watershed_v2

def remove_small_segments(label_array, min_size):
    """
    Removes segments smaller than `min_size` pixels by setting their label to 0.
    
    :param label_array: Numpy array of watershed-labeled segments.
    :param min_size: Minimum number of pixels for a segment to remain.
    :return: Numpy array with small segments removed.
    """
    unique_labels = np.unique(label_array)
    for label_val in unique_labels:
        if label_val == 0:
            # 0 is background/no label
            continue
        mask = (label_array == label_val)
        if np.count_nonzero(mask) < min_size:
            # Remove small segment
            label_array[mask] = 0
    return label_array

def main(
    input_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_set/Canada_geom2',
    output_dir = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/inference_set/Canada_geom2',
    model_path = '/mnt/h/CFS_superesolved_temp/DS_prelim_test/weights_unet/BFC-2024-12-25-013641/U-Net-Weights-BFCE.keras',
    min_segment_size = None
):
    """
    Main function to run the preprocessing workflow for Sentinel-2 images.

    This function sets up input and output directories, processes all zipped Sentinel-2 images,
    generates GeoTIFF files containing bands B2, NDVI, and NDWI, runs inference, reconstructs full predicted masks,
    combines them into a single binary mask, cleans the mask, and applies the Watershed algorithm iteratively.
    
    Optionally removes segments smaller than a specified number of pixels before polygonizing.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare a report file to capture all print statements
    report_path = os.path.join(output_dir, 'processing_report.txt')
    original_stdout = sys.stdout  # Save original stdout to restore later

    with open(report_path, 'w') as report_file:
        # Redirect prints to the report file
        sys.stdout = report_file

        # --- Write parameter info to report ---
        print("=== PROCESSING REPORT ===")
        print("Inputs:")
        print(f"  - input_dir:  {input_dir}")
        print(f"  - model_path: {model_path}")
        print("Outputs:")
        print(f"  - output_dir: {output_dir}")
        print()
        
        # ---------------------------
        # (OPTIONAL) Step 1: Process Sentinel-2 images and generate 3-channel GeoTIFFs
        # Uncomment these lines if you still want to unzip and prepare GeoTIFFs inside this script.
        #
        # print("Starting preprocessing of Sentinel-2 images...")
        # s2zip_reader.process_zip_directory(input_dir, output_dir)
        # print("Preprocessing complete. GeoTIFF files are saved in the output directory.")
        
        # ---------------------------
        # Step 2: Sub-tiling
        tile_size = 256
        overlap_size = 32

        print("Starting sub-tiling of GeoTIFF images...")
        subtiler_wOverlap_auto_SR.extract_subtiles_with_overlap(
            input_dir,
            output_dir,
            tile_size,
            overlap_size,
            channels_to_use=[1, 5, 6]
        )
        print("Sub-tiling complete. Tiles are saved in the output directory.")

        # ---------------------------
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

        # ---------------------------
        # Step 4: Reconstruct Full Predicted Masks
        for subdir in os.listdir(input_dir):
            predicted_folder = os.path.join(input_dir, subdir)
            if os.path.isdir(predicted_folder) and subdir.endswith('_tiles_woverlap_predicted'):
                # Extract the common prefix from the subdir name (base name without suffixes)
                prefix = subdir.split('_tiles_woverlap_predicted')[0]

                # Search for the original GeoTIFF file that matches this prefix
                matching_files = glob.glob(os.path.join(input_dir, f"{prefix}*.tif"))
                if not matching_files:
                    print(f"Error: No matching original GeoTIFF found for {subdir}. Expected a file starting with {prefix}")
                    continue

                original_geotiff = matching_files[0]  # Assuming the first match is correct
                output_file = os.path.join(input_dir, f"{prefix}_predicted_full.tif")  # Output reconstructed file

                print(f"Reconstructing full predicted mask for: {subdir}")
                ReMosaiker_overlap_v2.reconstruct_image(
                    predicted_folder,
                    tile_size,
                    overlap_size,
                    output_file,
                    original_geotiff
                )

        print("All images reconstructed successfully.")

        # ---------------------------
        # Step 5: Combine All Remosaicked Predicted Masks
        combined_output_file = os.path.join(output_dir, "combined_predicted_mask.tif")
        print("Combining all remosaicked predicted masks into a single binary mask...")
        AND_combiner.combine_predicted_masks(input_dir, combined_output_file)
        print("All remosaicked predicted masks have been combined successfully.")

        # ---------------------------
        # Step 6: Clean the Combined Mask
        cleaned_output_file = combined_output_file.replace('.tif', '_cleaned.tif')
        print("Cleaning the combined binary mask to remove noise and small objects...")
        unet_output_cleaner.process_mask(
            combined_output_file,
            cleaned_output_file,
            apply_opening=True,
            min_size=9,
            area_threshold=60
        )
        print("Combined mask has been cleaned and saved.")

        # ---------------------------
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
        
        # Save the final result after all iterations
        base_dir = os.path.dirname(cleaned_output_file)
        input_filename = os.path.splitext(os.path.basename(cleaned_output_file))[0]
        output_path = os.path.join(base_dir, f"{input_filename}_WS_final.tif")
        iterativeWS_v4.write_geotiff(
            output_path,
            watershed_labels,
            geotransform,
            projection,
            dtype=gdal.GDT_UInt32
        )

        print("All watershed segmentation rounds completed.")
        print(f"Final watershed segmentation image saved at: {output_path}")

        # ---------------------------
        # (NEW) Optionally Remove Small Segments
        if min_segment_size is not None and min_segment_size > 0:
            print(f"Removing segments smaller than {min_segment_size} pixels...")
            # Read the label file again
            ws_labels, geo, proj = ReMosaiker_overlap_v2.read_geotiff(output_path)
            ws_labels = remove_small_segments(ws_labels, min_segment_size)
            
            # Overwrite the final watershed file after removing small segments
            iterativeWS_v4.write_geotiff(
                output_path,
                ws_labels,
                geo,
                proj,
                dtype=gdal.GDT_UInt32
            )
            print("Small segments removed.")

        # ---------------------------
        # Step 8: Polygonize the final watershed mask
        print("Polygonizing the final watershed mask to create a shapefile...")
        output_shapefile = output_path.replace('.tif', '.shp')

        #polygonize_watershed.polygonize_watershed_with_mask(output_path, output_shapefile)
        polygonize_watershed.polygonize_watershed_with_mask(
            "path/to/input.tif",
            "path/to/output.shp",
            simplify_tolerance=5.0
        )
        print(f"Polygon shapefile saved at: {output_shapefile}")

        # Write final summary to the report
        print("\n=== PROCESSING COMPLETED SUCCESSFULLY ===")
        print("Please see above logs for the detailed workflow and outputs.")
    
    # Restore stdout so future prints go to console again
    sys.stdout = original_stdout

    print("Processing complete. A full report of this run is available at:")
    print(f"  {report_path}")

if __name__ == "__main__":
    # Example usage with min_segment_size set to 50 (remove segments < 50 pixels).
    # Adjust as needed, or pass None to skip removing small segments.
    main(min_segment_size=50)
