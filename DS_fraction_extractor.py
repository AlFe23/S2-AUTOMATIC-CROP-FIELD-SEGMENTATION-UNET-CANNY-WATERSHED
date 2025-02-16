#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:53:28 2024

@author: tesla
"""

import os
import random
import shutil
from collections import defaultdict

def extract_fraction_of_files(input_dir, labels_dir, output_input_dir, output_labels_dir, fraction):
    if not os.path.exists(output_input_dir):
        os.makedirs(output_input_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    # Dictionary to store groups of files
    file_groups = defaultdict(list)

    # Walk through the input directory and group files
    for file_name in os.listdir(input_dir):
        if '_' in file_name:
            prefix = '_'.join(file_name.split('_')[:2])
            file_groups[prefix].append(file_name)

    # Seed the random number generator for reproducibility
    random.seed(42)

    # Process each group
    for prefix, files in file_groups.items():
        # Calculate the number of files to extract
        num_files_to_extract = max(1, int(len(files) * fraction))
        # Randomly select files
        selected_files = random.sample(files, num_files_to_extract)
        # Copy selected files to the output directories
        for file_name in selected_files:
            input_file_path = os.path.join(input_dir, file_name)
            label_file_path = os.path.join(labels_dir, file_name)

            # Copy input file
            shutil.copy(input_file_path, os.path.join(output_input_dir, file_name))
            
            # Copy corresponding label file if it exists
            if os.path.exists(label_file_path):
                shutil.copy(label_file_path, os.path.join(output_labels_dir, file_name))
            else:
                print(f"Warning: Corresponding label file not found for {file_name}")



if __name__ == "__main__":
    input_directory = '/mnt/ssd3/unet/DS_A_input'
    labels_directory = '/mnt/ssd3/unet/DS_A_label'
    output_input_directory = '/mnt/ssd3/unet/DS_A_input_fraz'
    output_labels_directory = '/mnt/ssd3/unet/DS_A_label_fraz'
    fraction_to_extract = 0.25  # Example: 10% of the files

    extract_fraction_of_files(input_directory, labels_directory, output_input_directory, output_labels_directory, fraction_to_extract)
