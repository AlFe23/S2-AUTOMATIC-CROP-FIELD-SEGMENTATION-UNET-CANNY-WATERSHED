#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:30:30 2024

@author: tesla

ResUNet al posto della UNet
"""

import datetime
import os
import glob
import numpy as np
import tifffile as tiff
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input, Add, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import Precision, Recall, BinaryIoU, BinaryAccuracy
from tensorflow.keras.mixed_precision import set_global_policy
from matplotlib import pyplot as plt
from skimage.io import imshow
from tensorflow.keras import backend as K

###################################################################
# Environment Variables: Based on the logs, you might want to experiment with the TensorFlow environment variable TF_GPU_ALLOCATOR=cuda_malloc_async as suggested by the warning. This setting can potentially improve memory management:
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Set mixed precision policy
# set_global_policy('mixed_float16') # N.B. Forse è questo a causare l'esplosione del cradiente e valori di loss infiniti !
set_global_policy('float32')  # Disable mixed precision

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

###################################################################

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
batch_size = 32  # Increase batch size to utilize more VRAM
train_dataset = train_dataset.shuffle(buffer_size=len(train_image_paths)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

###########################################################################

# Dice Loss function definition

K.set_image_data_format('channels_last')

# Function for Dice Coefficient
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

###################################################################
# Model Definition

def residual_block(x, filters):
    shortcut = Conv2D(filters, (1, 1), padding='same')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def res_unet(input_shape):
    inputs = Input(input_shape)
    
    c1 = residual_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = residual_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = residual_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = residual_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)
    
    c5 = residual_block(p4, 1024)
    
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = residual_block(u6, 512)
    
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = residual_block(u7, 256)
    
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = residual_block(u8, 128)
    
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = residual_block(u9, 64)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


#####################################################################

# input_shape = (256, 256, 3)
# model = res_unet(input_shape)

# METRICS = [
#     BinaryIoU(target_class_ids=[0, 1], threshold=0.5),
#     BinaryAccuracy(threshold=0.5),
#     Precision(name="precision"),
#     Recall(name="recall"),
#     dice_coef
# ]

# model.compile(optimizer='adam', loss='BinaryFocalCrossentropy', metrics=METRICS)
# model.summary()

# ### Model training

# # Callbacks configuration
# callbacks = [
#     EarlyStopping(patience=20, monitor='val_loss', min_delta=0.0001),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),
#     ModelCheckpoint(model_checkpoint_path, verbose=1, save_best_only=True),
#     TensorBoard(log_dir=current_run_directory)
# ]


#####################################################################
# Per incrementare la stabilità ed evitare l'esplosione del gradiente si è aggiunto il parametro epsilon all'ottimizzatore del gradiente:
#    The epsilon parameter in the Adam optimizer helps in stabilizing the training process by preventing division by very small numbers, which can result in numerical instability or NaN values. 

model = res_unet((256, 256, 3))
model.summary()

# Compile model with a reduced learning rate and added NaN prevention measures
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8)  # added epsilon for numerical stability

model.compile(optimizer=optimizer, 
              loss='binary_crossentropy',  # Use a simpler loss function for stability
              metrics=[BinaryIoU(threshold=0.5), BinaryAccuracy(), Precision(), Recall(), dice_coef])

# Callbacks
callbacks = [
    ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
    TensorBoard(log_dir=current_run_directory)
]

# Assuming train_dataset and val_dataset are already defined
results = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=callbacks
)

# PLOT VALIDATION ACCURACY AND LOSS
print(results.history.keys())

# "Accuracy"
plt.plot(results.history['binary_accuracy'])
plt.plot(results.history['val_binary_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color='green', linestyle='--', linewidth=0.4)
plt.show()

# "Loss"
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()

plt.plot(results.history['binary_io_u'])
plt.plot(results.history['val_binary_io_u'])
plt.title("IoU (Jaccard's Index)")
plt.ylabel('IoU')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()

plt.plot(results.history['dice_coef'])
plt.plot(results.history['val_dice_coef'])
plt.title('Model Accuracy (Dice Coefficient)')
plt.ylabel('Dice Coefficient')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()

plt.plot(results.history['precision'])
plt.plot(results.history['val_precision'])
plt.title('Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()

plt.plot(results.history['recall'])
plt.plot(results.history['val_recall'])
plt.title('Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()
