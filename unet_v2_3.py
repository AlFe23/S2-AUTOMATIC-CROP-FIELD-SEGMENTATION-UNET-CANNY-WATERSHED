#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:30:30 2024

@author: tesla

ottimizzazione ulteriore rispetto alla v2.2

U-Net Training Script for Crop Field Segmentation (v2.3)

This version introduces **further optimizations** in dataset handling and training stability,  
improving both performance and scalability for large-scale datasets.

Key Features:
- **Refined dataset handling and memory optimizations**.
- **Larger batch sizes supported due to better VRAM management**.
- **Maintains TensorFlow's tf.data.Dataset approach for efficient training**.
- **Uses the same U-Net model architecture as previous versions**.

Differences from the Previous Version:
- **More aggressive memory optimization**, reducing RAM/VRAM consumption.
- **Better GPU utilization with larger batch sizes**.
- **More stable dataset transformations** to improve training robustness.

Quick User Guide:
1. Set `input_images_directory` and `canny_masks_directory` paths.
2. Run the script:
       python unet_v2_3.py
3. The trained model weights will be saved in the `weights_unet` folder.

Dependencies:
Python packages: numpy, gdal (from osgeo), tifffile, TensorFlow, keras, matplotlib

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

"""

import datetime
import os
import glob
import numpy as np
import tifffile as tiff
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.mixed_precision import set_global_policy
from matplotlib import pyplot as plt
from skimage.io import imshow

########################################################################
# Environment Variables: Based on the logs, you might want to experiment with the TensorFlow environment variable TF_GPU_ALLOCATOR=cuda_malloc_async as suggested by the warning. This setting can potentially improve memory management:
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

####################################################################

# Set mixed precision policy
set_global_policy('mixed_float16')

# Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set successfully")
    except RuntimeError as e:
        print("Error setting memory growth: ", e)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

####################################################################

base_directory = '/mnt/ssd3/unet/training_DS_A'
input_images_directory = os.path.join(base_directory, "DS_A_input/*.tif")
canny_masks_directory = os.path.join(base_directory, "DS_A_label/*.tif")

# Create directories for logs and weights if they don't exist
logs_directory = os.path.join(base_directory, "logs_unet")
weights_directory = os.path.join(base_directory, "weights_unet")
os.makedirs(logs_directory, exist_ok=True)
os.makedirs(weights_directory, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
current_run_directory = os.path.join(logs_directory, f"BFC-{timestamp}")
weights_run_directory = os.path.join(weights_directory, f"BFC-{timestamp}")
os.makedirs(current_run_directory, exist_ok=True)
os.makedirs(weights_run_directory, exist_ok=True)

model_checkpoint_path = os.path.join(weights_run_directory, 'U-Net-Weights-BFCE.h5')

# Function to load and preprocess a single image and label
def load_image(image_path, label_path):
    image_path = image_path.decode("utf-8")
    label_path = label_path.decode("utf-8")
    image = tiff.imread(image_path)
    label = tiff.imread(label_path)

    # Normalize the image
    image = image.astype(np.float32)
    image[..., 0] = image[..., 0] / 10000.0  # Normalize B2 channel
    image[..., 1] = image[..., 1] / 65535.0  # Normalize NDVI channel
    image[..., 2] = image[..., 2] / 65535.0  # Normalize NDWI channel

    # Normalize the label
    label = label.astype(np.float32) / 255.0  # Normalize binary image
    label = np.expand_dims(label, axis=-1)  # Add channel dimension

    return image, label

# Function to parse image paths into tensors
def parse_image(image_path, label_path):
    image, label = tf.numpy_function(load_image, [image_path, label_path], [tf.float32, tf.float32])
    image.set_shape([256, 256, 3])
    label.set_shape([256, 256, 1])
    return image, label

# Get file paths
image_paths = sorted(glob.glob(input_images_directory))
label_paths = sorted(glob.glob(canny_masks_directory))

# Split dataset indices
validation_size = int(len(image_paths) * 0.1)
indices = np.arange(len(image_paths))
np.random.shuffle(indices)
train_indices = indices[validation_size:]
val_indices = indices[:validation_size]

train_image_paths = [image_paths[i] for i in train_indices]
train_label_paths = [label_paths[i] for i in train_indices]
val_image_paths = [image_paths[i] for i in val_indices]
val_label_paths = [label_paths[i] for i in val_indices]

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_label_paths))
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_label_paths))

# Apply transformations
train_dataset = train_dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Apply caching, shuffling, batching, and prefetching
batch_size = 64  # Increase batch size to utilize more VRAM
train_dataset = train_dataset.shuffle(buffer_size=len(train_image_paths)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

###########################################################################

# Dice Loss function definition

K.set_image_data_format('channels_last')

smooth = 1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

###########################################################################

### Model Definition

image_row = 256
image_col = 256
image_ch = 3

inputs = Input((image_row, image_col, image_ch))

c1 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D(pool_size=(2, 2))(c1)

c2 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D(pool_size=(2, 2))(c2)

c3 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D(pool_size=(2, 2))(c3)

c4 = Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(1024, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(1024, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c5)

u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4], axis=3)
c6 = Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3], axis=3)
c7 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2], axis=3)
c8 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = Conv2D(1, 1, activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])

METRICS = [
    tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5), 
    tf.keras.metrics.BinaryAccuracy(threshold=0.5),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    dice_coef]

model.compile(optimizer= 'adam', loss='BinaryFocalCrossentropy', metrics=METRICS) # For the Adam optimizer, the default learning rate is 0.001

model.summary()

### Model training

# Callbacks configuration
callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss', min_delta=0.0001),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, verbose=1, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=current_run_directory)
        ]

results = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,  # Only if you have a validation dataset prepared similarly
    callbacks=callbacks
)

# PLOT VALIDATION ACCURACY AND LOSS
print(results.history.keys())

# "Accuracy"
plt.plot(results.history['binary_accuracy'])
plt.plot(results.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.4)
plt.show()

# "Loss"
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()
 
plt.plot(results.history['binary_io_u'])
plt.plot(results.history['val_binary_io_u'])
plt.title('IoU (Jaccard''s index)')
plt.ylabel('IoU')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()

plt.plot(results.history['dice_coef'])
plt.plot(results.history['val_dice_coef'])
plt.title('model accuracy (Dice coefficient)')
plt.ylabel('Dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()

plt.plot(results.history['precision'])
plt.plot(results.history['val_precision'])
plt.title('Precision')
plt.ylabel('Precision')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()

plt.plot(results.history['recall'])
plt.plot(results.history['val_recall'])
plt.title('Recall')
plt.ylabel('Recall')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()
