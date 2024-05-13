# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:55:13 2024

@author: Alvise Ferrari

Differences between ver2.0 and ver2.1

### 1. **Dataset Preparation and Management**

**Old Version:**
- Directly loads images and masks from files using loops and file path handling, then processes these into numpy arrays for training.
- Does not utilize TensorFlow’s data handling efficiencies such as `tf.data.Dataset`.
- Manages training data purely with numpy arrays, potentially less efficient in memory management and might not be as optimized for large datasets.

**New Version:**
- Introduces TensorFlow’s `tf.data.Dataset` for dataset management.
- Manually splits the dataset into training and validation sets using indices, which is a change from potentially using `validation_split` directly in `model.fit()` (common in simpler Keras workflows).
- Applies transformations like caching, shuffling, batching, and prefetching directly within the TensorFlow pipeline. This method is highly efficient for large datasets and leverages TensorFlow’s built-in optimizations for performance improvement.
- The utilization of `tf.data.Dataset` is a significant enhancement, particularly for handling large volumes of data and for optimizing I/O operations, which can be critical in deep learning workflows involving extensive data like images.

### 2. **Validation Data Handling**

**Old Version:**
- Likely uses a straightforward split for validation, or it might not explicitly separate training and validation data during the setup.

**New Version:**
- Explicitly creates a validation dataset using a portion of the data (10% as indicated).
- Ensures that validation data is treated consistently with training data in terms of transformations and batching, making the model evaluation during training more reliable and structured.

### 3. **Performance Optimizations**

**New Version:**
- Caching is applied to store the preprocessed data in memory after the first epoch, reducing the time spent in data loading for subsequent epochs.
- Prefetching is used to prepare data while the model is training, reducing GPU idle time and improving overall training speed.
- Shuffling helps in reducing variance and making sure that models remain general and overfit less.

### 4. **TensorFlow’s Advanced Features**
- The new version likely takes advantage of TensorFlow’s capabilities more fully, including potential integration with other TensorFlow functionalities like mixed precision training (if used) that can significantly speed up training and reduce memory usage.

### 5. **Data Loading and Augmentation**
- If any, changes in how data augmentation is applied (not explicitly detailed but can be inferred if using TensorFlow's pipeline more extensively).

### Conclusion:
These changes signify a shift towards more robust, scalable, and efficient data handling practices using TensorFlow’s best practices. This shift is critical for improving model training efficiency, especially when dealing with large datasets typical in image processing tasks. This adaptation not only makes the code more maintainable but potentially enhances performance on larger datasets or more complex training scenarios.
"""
import datetime
import os
import cv2
import glob
import numpy as np
import tifffile as tiff

########################################################################
#Environment Variables: Based on the logs, you might want to experiment with the TensorFlow environment variable TF_GPU_ALLOCATOR=cuda_malloc_async as suggested by the warning. This setting can potentially improve memory management:
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

####################################################################

from tifffile import imread 
from albumentations import Transpose,RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip, ElasticTransform
from skimage.io import imshow
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input, Dropout, Add, Activation, UpSampling2D,  Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy
from keras import backend as K
from tensorflow.keras.utils import array_to_img
from sklearn.model_selection import train_test_split

###################################################################
# Mixed Precision Training: This approach utilizes both 16-bit and 32-bit floating-point types during training, which can reduce memory usage and possibly even speed up training, with minimal impact on the accuracy of your model.

#Set the Global Mixed Precision Policy
from tensorflow.keras.mixed_precision import set_global_policy

# Set mixed precision policy
set_global_policy('mixed_float16')
###################################################################
#Enable GPU Memory Growth: Instead of allocating all GPU memory at once, you can configure TensorFlow to allocate memory as needed. This can prevent TensorFlow from using all the GPU memory upfront. Here’s how you can set this configuration:
    
#Setting up TensorFlow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set successfully")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print("Error setting memory growth: ", e)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

####################################################################

# #HIDE GPU
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




# # INPUT DATA
# train_img_dir = os.path.join(r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\input_imgs\IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_tiles_woverlap\*.tif")
# train_label_dir = os.path.join(r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\canny_mask\IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_tiles\*.tif")
# X = np.array([imread(file) for file in glob.glob(train_img_dir)])
# Y = np.array([cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob(train_label_dir)])

#######################################################


base_directory = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020"
input_images_directory = os.path.join(base_directory, "input_2020_simple\*.tif")
canny_masks_directory = os.path.join(base_directory, "labels_2020_simple\*.tif")

# Create directories for logs and weights if they don't exist
logs_directory = os.path.join(base_directory, "logs_unet")
weights_directory = os.path.join(base_directory, "weights_unet")
os.makedirs(logs_directory, exist_ok=True)
os.makedirs(weights_directory, exist_ok=True)
# Use datetime to create a unique subdirectory name based on the current date and time
# Paths for logs and weights
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
current_run_directory = os.path.join(logs_directory, f"BFC-{timestamp}")
weights_run_directory = os.path.join(weights_directory, f"BFC-{timestamp}")
os.makedirs(current_run_directory, exist_ok=True)
os.makedirs(weights_run_directory, exist_ok=True)

model_checkpoint_path = os.path.join(weights_run_directory, 'U-Net-Weights-BFCE.h5')

# Function to read all TIFF files matching a pattern using tifffile
def load_tiff_directory_tifffile(pattern):
    filepaths = glob.glob(pattern)
    return np.array([tiff.imread(fp) for fp in filepaths])

# # Paths
# train_img_path = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IMG_IOWA_15TWG_20200710_tiles_woverlap\*.tif"
# train_label_path = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2020\IOWA_15TWG_canny_2020_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh_tiles_woverlap\*.tif"

# Load input images
X = load_tiff_directory_tifffile(input_images_directory)

# Load label images
Y = load_tiff_directory_tifffile(canny_masks_directory)


#######################################################



#INPUT SCALING between 0 and 1 (mantenere i valori nel range 0-255 porta a valori di Loss molto grandi rendendo la convergenza del modello molto più difficile)

# Make a copy of X for normalization
X_train = np.copy(X)

# Normalize B2 channel: 0-10000 to 0-1
X_train[:,:,:,0] = X_train[:,:,:,0] / 10000.0

# Normalize NDVI channel: 0-65535 to 0-1
X_train[:,:,:,1] = X_train[:,:,:,1] / 65535.0

# Normalize NDWI channel: 0-65535 to 0-1
X_train[:,:,:,2] = X_train[:,:,:,2] / 65535.0

# Normalize binary image (0,255) to (0,1)
Y_train = Y/255

# # TRAIN-TEST SPLIT with sklearn
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=2)

#######################################################
# #Tentativo di usare tf.data API per migliorare la gestione memoria:

# Define the size of the validation set
validation_size = int(len(X_train) * 0.1)  # 10% for validation

# Shuffle the dataset indices
indices = np.arange(len(X_train))
np.random.shuffle(indices)

# Split indices for training and validation sets
train_indices = indices[validation_size:]
val_indices = indices[:validation_size]

# Create training and validation datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train[train_indices], Y_train[train_indices]))
val_dataset = tf.data.Dataset.from_tensor_slices((X_train[val_indices], Y_train[val_indices]))

# Apply the necessary transformations like caching, shuffling, batching, and prefetching
train_dataset = train_dataset.cache().shuffle(buffer_size=len(train_indices)).batch(4).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(4).prefetch(tf.data.experimental.AUTOTUNE)

'''
Important Points:
Caching: You should use .cache() before .shuffle() to cache the initial loading and any expensive preprocessing operations. Since your data is preprocessed already, caching will just store these numpy arrays in memory, reducing disk I/O during training.
Shuffling: Shuffling is crucial for training neural networks to prevent the model from learning any order in the training data. The buffer_size should ideally be the size of the dataset for the best shuffle.
Batching: Adjust the batch size if you encounter memory issues, though a batch size of 4 is usually small enough for most GPU memory limits.
Prefetching: This is a performance optimization that allows later batches to be prepared while the current batch is being used. This helps reduce the time the GPU spends waiting for data.
'''


#######################################################

#shape check
print("The shape of train set is {s1}.\nThe shape of train labels is {s2}.".format(s1=X_train.shape,s2=Y_train.shape))

#random print to check correct mapping of data to labels
rand_num = np.random.randint(0,len(X_train))
print('At Index : {index}'.format(index = rand_num))

from tensorflow.keras.utils import array_to_img
rand_img = X_train[rand_num]
fig = plt.subplots(dpi=300)
image=array_to_img(rand_img)
plt.subplot(121), plt.imshow(image)
plt.title('Train image') 
plt.subplot(122), imshow(Y_train[rand_num])
plt.title('Train label') 

plt.show()



'''
### Focal Loss for training model

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

loss = tf.keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=True, gamma=2)
'''

###########################################################################

#Dice Loss function definition

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
    return (1 -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
###########################################################################  



### Model Definition

image_row = 256
image_col = 256
image_ch = 3

inputs = Input((image_row, image_col, image_ch))

c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(c1)

c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(512,3, activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c5)

u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4],axis = 3)
c6 = tf.keras.layers.Conv2D(512,3, activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3],axis = 3)
c7 = tf.keras.layers.Conv2D(256,3, activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(256,3, activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2],axis = 3)
c8 = tf.keras.layers.Conv2D(128,3, activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(128,3, activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(64,3, activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(64,3, activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])



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
        tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss',min_delta=0.0001),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, verbose=1, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=current_run_directory)
        ]

#results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=4, epochs=100, callbacks=callbacks)

results = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,  # Only if you have a validation dataset prepared similarly
    callbacks=callbacks
)


# PLOT VALIDATION ACCURACY AND LOSS
print(results.history.keys())

#  "Accuracy"
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
plt.title('model accuracy (Dice coefficient')
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

# #ANOTHER WAY TO PLOT TEST AND VALIDATION ACCURACY
# import pandas as pd
# pd.DataFrame(results.history).plot(figsize=(8,5))
# plt.show()





# #LOAD A PRE-TRAINED MODEL
# pretrained_model = 'C:/Users/ferra/Desktop/igarss2023/rgb19feb20-U-Net-Weights-BFCE.h5'
# model = tf.keras.models.load_model(pretrained_model, custom_objects={'dice_coef': dice_coef}) #remember to re-run the dice_coef function before to load the model since it is a custom object


# ### Testing the model

# '''
# The Keras.evaluate() method is for testing or evaluating the trained model. It’s output is accuracy or loss of the model.

# The Keras.Predict() method is for predicting the output. It’s output is predicted value or output from the input data.

# '''

# # # Model evaluation using the test dataset
# # model.evaluate(X_test, Y_test, return_dict=True)


# # Model prediction using test images
# preds_test = model.predict(X_test[0:10], verbose=1)
# new_pred = preds_test.reshape((10,512,512))


# # test label vs predicted label
# test_rand_num = np.random.randint(0,len(new_pred))
# print('At Index : {index}'.format(index = test_rand_num))
# test_label = new_pred[test_rand_num]
# fig = plt.subplots(dpi=300)
# plt.subplot(121), plt.imshow(test_label>0.65, cmap='gray')
# plt.title('Predicted label')
# plt.subplot(122), imshow(Y_test[test_rand_num])
# plt.title('Test label') 
# plt.show()


# # test img and predicted labels
# print('At Index : {index}'.format(index = test_rand_num))
# test_img = X_test[test_rand_num]
# test_image=array_to_img(test_img)
# fig = plt.subplots(dpi=300)
# plt.subplot(121), plt.imshow(test_image)
# plt.title('Test image') 
# plt.subplot(122), plt.imshow(test_label>0.65, cmap='gray')
# plt.title('Predicted label') 


# '''
# plt.show()
# imlist = []
# for m in new_pred:
#     imlist.append(Image.fromarray(m>0.8))

# imlist[9].save("test-labels-rgb-2.tif", compression="tiff_deflate", save_all=True,append_images=imlist[1:])
# '''


