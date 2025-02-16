#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:11:54 2024

@author: tesla
"""

import os
import shutil

def extract_files_to_main_dir(main_dir):
    # Walk through the directory
    for root, dirs, files in os.walk(main_dir, topdown=False):
        # Ignore the main directory itself
        if root == main_dir:
            continue
        for name in files:
            # Construct the full path of the file
            file_path = os.path.join(root, name)
            # Move the file to the main directory
            shutil.move(file_path, os.path.join(main_dir, name))
        for name in dirs:
            # Remove the now empty directory
            os.rmdir(os.path.join(root, name))

if __name__ == "__main__":
    main_directory = '/mnt/ssd3/unet/DS_A_label/'
    extract_files_to_main_dir(main_directory)
