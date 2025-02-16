# AUTOMATIC-CROP-FIELD-SEGMENTATION-USING-UNET-CANNY-WATERSHED

Automatic software for crop field segmentation using Sentinel-2 L2A images. This tool leverages a UNet architecture, trained on multitemporal canny edges masks, and the watershed algorithm, to deliver high-precision crop field segmentation. Ideal for agricultural researchers and GIS specialists seeking efficient and scalable solutions.

**Citation:**


A. Ferrari, S. Saquella, G. Laneve and V. Pampanoni,  
"Automating Crop-Field Segmentation in High-Resolution Satellite Images:  
A U-Net Approach with Optimized Multitemporal Canny Edge Detection,"  
IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium,  
Athens, Greece, 2024, pp. 4094-4098, doi: 10.1109/IGARSS53475.2024.10641103

**Links:**

Download pre-trained models at:

- [pre-trained-model-v1](https://drive.google.com/drive/folders/1yAT4z0pPhm3MQbG0jzgqqqkhjf_C_PSS)

### Index
1. **Dataset Preparation**
   - 1.1 Dataset Preparation with GEE
   - 1.2 Binarization of Multitemporal Canny Mask
   - 1.3 Splitting Training Dataset into Sub-Tiles

2. **Segmentation with UNet**
   - 2.1 U-Net Training
     - 2.1.1 Features
     - 2.1.2 Prerequisites and dependencies
     - 2.1.3 Dataset Preparation
     - 2.1.4 Network Architecture
     - 2.1.5 Optimizer Configuration
     - 2.1.6 Training
     - 2.1.7 Input Management
     - 2.1.8 TensorBoard Log and Model Weights Saving
     - 2.1.9 Evaluation and Metrics
     - 2.1.10 Results Visualization
   - 2.2 Fine-Tuning Pre-Trained U-Net
     - 2.2.1 Description
     - 2.2.2 Key Differences from Training from Scratch
     - 2.2.3 Prerequisites
     - 2.2.4 Fine-Tuning Configuration
     - 2.2.5 Input Management
     - 2.2.6 Log and Weights Saving
     - 2.2.7 Training Output
   - 2.3 Inference with U-Net on New Data
     - 2.3.1 Prerequisites and Dependencies
     - 2.3.2 Configuration
     - 2.3.3 Dataset Preparation
     - 2.3.4 Inference Execution
     - 2.3.5 Saving Predicted Masks
   - 2.4 Reconstructing Complete Images from Predicted Subtiles
   - 2.5 (Optional) Overlaying Multiple Predictions
   - 2.6 Cleaning the Predicted Mask with Morphological Transformations

3. **Watershed Segmentation and Polygonization**
   - 3.1 Iterative Growing-Regions Segmentation with Watershed
     - 3.1.1 Description
     - 3.1.2 Input
     - 3.1.3 Output
     - 3.1.4 Iterative Watershed
   - 3.2 Polygonization of Watershed Segments
     - 3.2.1 Description
     - 3.2.2 Technical Details
4. **Automatic segmentation with pre-trained models**
   - 4.1 Pre-trained Model v1 with Generalised Training Dataset
   - 4.2 Example of segmentation obtained on a new AOI, Modesto (CA), USA.
   - 4.3 Automated Inference Scripts



## 1. **Dataset Preparation**

## 1.1 Dataset Preparation with GEE

The GEE script `Canny_Multitemporale_standard` generates a dataset with the following characteristics:

**Geotiff uint16 composed of 3 channels (B2, NDVI, NDWI)**, generated from completely cloud-free multitemporal Sentinel-2 L2A images, LZW compressed, projected in EPSG:4326 - WGS 84. The number of images generated for download depends on the number of cloud-free images (cl.cov.<0.5%) available within the specified time frame.

The three channels are scaled as follows:
- **B2, blue reflectance**, with dynamics 0,1 scaled linearly in the range 0,10000 already from the L2A product.
- **NDVI** with dynamics -1,1 scaled linearly in the range 0,65535 with the formula: ndvi_uint16 = (ndvi * 32767.5) + 32767.5
- **NDWI** with dynamics -1,1 scaled linearly in the range 0,65535 with the formula: ndwi_uint16 = (ndwi * 32767.5) + 32767.5

When preparing the dataset for UNet training, it is important to rescale the dataset values to the range [0-1]; thus, the first channel will be divided by 10000 and the second and third channels by 65535.

**Geotiff uint8 composed of 1 channel** containing the mask generated by overlaying the output of the Canny filter applied to a multitemporal sequence of Sentinel-2 L2A images. LZW compressed, projected in EPSG:4326 - WGS 84.
- This mask is generated by summing all Canny outputs, and the pixels have discrete values between 0 and 255. It must be binarized by applying a threshold before proceeding with the sub-tiling operation.

## 1.2 Binarization of Multitemporal Canny Mask

Once the Multitemporal Canny Masks are obtained, they need to be binarized using the function in`canny_binarizer.py` before they can be used for U-Net training.
(reminder: This script is an extract of the `canny_cleaner_v3.py` script, retaining only the first part where thresholding is performed).

**Input:**
   - Grayscale Multitemporal Canny Mask uint8 (obtained from GEE)

**Output:**
   - Binarized Canny Mask (0 == edge; 255 == non-edge)

## 1.3 Training I/O Splitting into Sub-Tiles

The script `new_subtiler_wOverlap.py` is designed to prepare 256x256 pixel tiles for U-Net training, starting from images binarized with `canny_binarizer.py`. The script supports both 1-channel and 3-channel images and saves the subtiles in a newly generated folder. This operation facilitates creating a dataset composed of various input images.

**Features:**
- **Multichannel Support**: Handles 1-channel or 3-channel images.
- **Automatic Folder Generation**: Automatically creates a folder in the same directory as the input image.
- **Intelligent Naming**: Subtiles are systematically named based on their position in the tile grid, with formats like `prefixes_subtile_0_0`, `prefixes_subtile_0_N`, up to `prefixes_subtile_M_N`; where `N` is the number of image columns divided by 256, and `M` is the number of rows of the input image divided by 256.
- **Overlap Management**: Allows extraction of subtiles considering an overlap between them.

To use the script, specify the input file, tile size, and overlap size.

Compared to the previous version, this version automatically adds a prefix to the tile name, provided in a list of prefixes corresponding to the list of input images.

**Example:**
```python
input_files = [
    input_img_path_1,
    input_img_path_2, 
    input_img_path_3, 
    input_img_path_4, ...
]

prefix_name_list = ['tilename1_data1', 'tilename1_data2', 'tilename1_data3', 'tilename1_data4', ...]
```

Note: the date in the prefix is also added within the name of the folder containing the subtiles of each image, ensuring that a different folder is created for each Canny input, repeated as many times as the number of 3ch input images. Remind that the training strategz proposed by the authors associates different input images to the same multitemporal Canny output mask!




> **Additionally**, besides `new_subtiler_wOverlap.py`, we have:
> - **`subtiler_wOverlap_auto.py`**: Basic tiling for original 10m Sentinel-2 images using user-defined overlap.
> - **`subtiler_wOverlap_v2.py`**: Adds support for **prefix-based** naming of output tiles, allowing multiple images in one run.
> - **`subtiler_wOverlap_auto_SR.py`**: Specifically designed for **superresolved Sentinel-2 L2A images**, and supports **channel selection** (e.g., `[1, 5, 6]` for B2, NDVI, NDWI).

Each script still produces 256×256 (default) tiles, with optional overlap (e.g. 32 px). Check the docstrings in each script for usage examples.

## 2. **Segmentation with UNet**
 
## 2.1 U-Net Training 

The Python script `Resunet_v2_3.py` uses TensorFlow to train a U-Net neural network. The network is optimized to run on GPU hardware, utilizing mixed precision and dynamic memory management to improve efficiency and performance.

> We provide multiple **training scripts**, each adding further optimizations:
> - **`unet_v2_1.py`**: Introduces `tf.data.Dataset` for efficient data loading (shuffling, prefetching).
> - **`unet_v2_2.py`**: Improves memory usage while loading large datasets.
> - **`unet_v2_3.py`**: Refines dataset handling and VRAM usage, enabling larger batch sizes.
> - **`Resunet_v2_3.py`**: Switches from plain U-Net to **ResUNet** (with residual connections).
> - **`Resunet_v3.py`**: Further refines ResUNet training (e.g. numerical stability, disabling mixed precision if unstable).

Use whichever best suits your system constraints. All share the same fundamental training flow described here.


### 2.1.1 Features
- **Mixed Precision Training**: Utilizes 16-bit and 32-bit data types during training, reducing memory consumption and speeding up the process.
- **Dynamic GPU Memory Management**: Configures TensorFlow to allocate GPU memory dynamically, preventing initial allocation of all memory.
- **Advanced Logging and Checkpointing**: Implements automatic checkpointing and detailed logging to monitor training and facilitate resumption in case of interruption.


### 2.1.2 Prerequisites
- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- tifffile
- scikit-image
- PIL


### 2.1.3 Dataset Preparation

First, all the I/O pairs of subtiles produced in the previous steps should be moved to unique folders for inputs and outputs (labels). The UNet code specifies two paths for inputs and labels.

The dataset should be composed of 3-channel images and their corresponding segmentation masks. Images are loaded and pre-processed as follows:

- **Loading**: Images and masks are loaded using `tifffile` to support multi-channel TIFF formats.
- **Normalization**: Images are normalized by dividing by the channel-specific maximum value to bring them into the range [0, 1].
- **Manual Splitting**

: The dataset is divided into training and validation sets using a predefined portion of the original dataset for validation. This split is based on a random index separating 10% of the data for validation.

### 2.1.4 Network Architecture

The architecture of choice, ResUnet performed slightly better with non-multitemporal inputs, showing versatility for single acquisition data. Given the practical need for models that perform well with single-date inputs, ResUnet was chosen as the optimal model.


**Detailed Architecture:**

- Residual Blocks: Each block comprises two convolutional layers, equipped with Batch Normalization and ReLU activation function, integrated with a shortcut connection that feeds the input directly to the output of the block. This setup is crucial for mitigating training degradation, allowing the construction of deeper networks through effective skip connections.
- training degradation, allowing the construction of deeper networks through effective skip connections.
- Encoding Path: Consists of an increasing number of filters across four residual blocks (64, 128, 256, 512), each followed by MaxPooling (2x2) to reduce dimensionality progressively.
- Bottleneck: Central to the network, this single residual block uses 1024 filters, functioning as a pivotal feature compressor and transformer within the network.
- Decoding Path: Features four transpose convolutional layers (512, 256, 128, 64 filters) paired with residual blocks to refine features progressively as they are upsampled.
- Output Layer: Utilizes a 1x1 Conv2D layer with Sigmoid activation to produce the final binary segmentation map.

**Training Details:**

- Optimizer: Adam, with a learning rate of 0.001 and epsilon set to 1e-8, for robust and adaptive parameter updates.
- Loss Function: Binary Cross-Entropy (BFCE), chosen specifically to address class imbalance issues within the datasets.
- Metrics: Includes Binary Intersection over Union (IoU), Binary Accuracy, Precision, Recall, and Dice Coefficient to provide a comprehensive assessment of model performance.
- Epochs: The network is trained for a total of 100 epochs, ensuring adequate learning without overfitting.

### 2.1.5 Optimizer Configuration
The model uses the Adam optimizer with the following parameters:

- **Learning Rate**: Initial 0.001, with dynamic reduction based on validation loss plateau, down to 10e-5
- **Loss Function**: Binary Focal Crossentropy, to handle class imbalance.

### 2.1.6 Training
The model is trained with the following specifications:

- **Batch Size**: 32 (variable based on available hardware)
- **Epochs**: 100 (70 is usually sufficient)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint to save the best weights, and TensorBoard for monitoring.


### 2.1.7 Input Management

For U-Net training, inputs must be organized as:

Directories are specified for input images and masks (labels) images, respectively through the variables `input_images_directory` and `canny_masks_directory`. These directories must contain TIFF image files preprocessed as described in previous sections.

Here's how they are specified in the code:

```python
base_directory = "path/to/dataset/base_directory"
input_images_directory = os.path.join(base_directory, "input_images/*.tif")
canny_masks_directory = os.path.join(base_directory, "labels/*.tif")
```

These variables point to the directories where input tiles and mask tiles are located.

### 2.1.8 TensorBoard Log and Model Weights Saving

Saving logs and model weights is essential to monitor training progress and resume training from a certain point in case of interruption. In your script, these operations are managed as follows:

- **Log Directory**: TensorBoard logs are saved in a subdirectory within the `logs_unet` directory, which is in turn located in the `base_directory`. The subdirectory for each training session is named with a timestamp to ensure uniqueness and facilitate identification of the corresponding training session.

```python
logs_directory = os.path.join(base_directory, "logs_unet")
current_run_directory = os.path.join(logs_directory, f"BFC-{timestamp}")
os.makedirs(current_run_directory, exist_ok=True)
```

- **Weights Directory**: Similarly to logs, model weights are saved in a directory called `weights_unet`, also located in the `base_directory`. Here, a subdirectory specific to each training session is created using the same timestamp used for the logs.

```python
weights_directory = os.path.join(base_directory, "weights_unet")
weights_run_directory = os.path.join(weights_directory, f"BFC-{timestamp}")
model_checkpoint_path = os.path.join(weights_run_directory, 'U-Net-Weights-BFCE.h5')
os.makedirs(weights_run_directory, exist_ok=True)
```

Each model is saved with the name `U-Net-Weights-BFCE.h5` within its specific timestamp-identified subdirectory, facilitating retrieval of weights for future sessions or post-training analysis.

This organizational structure not only keeps files orderly and easily accessible but also allows multiple training sessions to run in parallel without the risk of overwriting important data.

### 2.1.9 Evaluation and Metrics
The model is evaluated using standard metrics like IoU (Jaccard index), Precision, Recall, and Dice Coefficient. These metrics are calculated for each batch and displayed at the end of each epoch for the training and validation sets.

### 2.1.10 Results Visualization
Accuracy, loss, IoU, and other metric graphs are generated using Matplotlib to visualize the model's performance over the course of training.

## 2.2 Fine-Tuning Pre-Trained U-Net

### 2.2.1 Description
The Python script `Fine_tuning.py` is intended for fine-tuning a pre-trained U-Net network to improve its accuracy on a new dataset. Unlike training from scratch, fine-tuning adapts a pre-trained model to refine its predictive capabilities further, leveraging the knowledge already acquired.

### 2.2.2 Key Differences from Training from Scratch
- **Starting Point**: Begins with a model that has already learned significant patterns from a similar or related dataset, instead of starting with random weights.
- **Learning Rate**: Uses a much lower learning rate (0.0001 in fine-tuning vs. typically 0.001 in training from scratch) to make more subtle adjustments to the weights and prevent loss of previously learned information.
- **Epochs**: Generally, fewer epochs are needed in fine-tuning because the model does not have to learn from scratch.
- **Callbacks**: Similar configurations for callbacks, but with a lower threshold for early stopping and reducing the learning rate, reflecting the expectation of more marginal and refined progress.

### 2.2.3 Prerequisites
The list of software dependencies such as Python, TensorFlow, OpenCV, etc., remains unchanged from the training script from scratch.

### 2.2.4 Fine-Tuning Configuration
#### Model Loading
Load a pre-trained model by specifying the path to the saved weights. This step is crucial to start fine-tuning on a solid foundation.

#### Data Preparation
- **Input and Normalization**: Images are loaded and normalized in the same way as in training from scratch, but consistency with the original model's preprocessing is vital.

#### Network Configuration
- **Optimizer**: Uses `Adam` with a `learning_rate` of 0.0001 to minimize the risk of disrupting previous learning.
- **Loss and Metrics**: Configures loss functions and metrics as in the original training to maintain consistency in evaluations.

### 2.2.5 Input Management
Specifies paths and loading and normalization procedures, as described in the data preparation section of the training script from scratch.

### 2.2.6 Log and Weight Saving
- **Logs**: Saves TensorBoard logs to monitor fine-tuning according to the same directory logic described in the training script from scratch.
- **Weight Saving**: Weights are saved in a unique timestamp-identified directory to distinguish fine-tuned weight sets from those of the original training.

### 2.2.7 Training Output
Generates and visualizes graphs of accuracy, loss, IoU, Dice coefficient, precision, and recall, allowing direct comparison with pre-fine-tuning results.

## 2.3 Inference with U-Net on New Data

The script `inference.py` performs inference on a new dataset using a previously trained U-Net model. The objective is to generate segmentation masks for individual Sentinel-2 input images, applying the same preprocessing used in training to ensure consistency of results.

### 2.3.1 Prerequisites
- Python 3.x
- TensorFlow 2.x
- tifffile
- NumPy

### 2.3.2 Configuration
- **Disable GPU**: By default, the script runs inference using the CPU. This can be modified by removing or commenting out the line `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`.
- **Pre-trained Model**: Ensure that the path to the saved model is correct and accessible by the script.

> In addition to `inference.py`, we include:
> - **`inference_v2.py`**: Computes IoU, Precision, Recall at inference time.
> - **`inference_v3.py`**: Supports **both `.keras` and `.h5`** model formats, plus custom layers.
> - **`inference_v3_solopred_auto.py`**: Automates **batch inference** on multiple subfolders, letting you run inference on all tile directories in one go.

These scripts share the same structure: load, normalize images, run predictions, save masks.


### 2.3.3 Dataset Preparation
The dataset for inference must be prepared with the same preprocessing used for training:
- **Loading Images**: Images must be loaded from the specified directory.
- **Normalization**: Images are normalized in the same way they were during training.

```python
def load_and_preprocess_images(directory):
    image_paths = glob.glob(os.path.join(directory, "*.tif"))
    images = [tiff.imread(fp) for fp in image_paths]
    images = np.array(images)
    # Apply the same normalization as during training
    images[:,:,:,0] = images[:,:,:,0] / 10000.0
    images[:,:,:,1] = images[:,:,:,1] / 65535.0
    images[:,:,:,2] = images[:,:,:,2] / 65535.0
    return images, image_paths
```

### 2.3.4 Inference Execution
- **Load and Preprocess New Images**: Use the `load_and_preprocess_images` function to prepare the data.
- **Model Execution**: Apply the model to the preprocessed images to generate predictions.

### 2.3.5 Saving Predicted Masks
Each predicted mask is saved in a specified output directory using a format that facilitates identification:

```python
for i, predicted_mask in enumerate(predicted_masks):
    filename = os.path.basename(image_paths[i])
    output_path = os.path.join(output_directory, f"predicted_{filename}")
    tiff.imwrite(output_path, predicted_mask)
```

The predicted subtiles will thus have the same naming logic as the input ones. This is important because it is through the naming of the subtiles that the complete output mask can be easily reconstructed with the script `ReMosaicker_overlap.py` described later.
At the end of the inference, the predicted masks will be saved in the specified directory. The script provides feedback by printing the output directory path.

## 2.4 Reconstructing Complete Images from Predicted Subtiles with U-Net

### Description
The Python script `ReMosaiker_overlap_v2.py` is designed to recompose a complete image from subtiles generated during inference using the U-Net network. `ReMosaiker_overlap_v2.py` manages recomposition considering the overlaps between the subtiles based on their naming. Specifically, the naming of the subtiles contained in the input folder must be as follows:

- predicted_20211018_subtile_m_n.tif

Where 'm' and 'n' respectively represent the row and column position of the 256x256 pixel tile mosaic.

Note: the script reads the positions starting from the third position, where the positions are separated by '_'.

Remember to set the same overlap size as the one chosen within input pre-processing step.

> We now recommend using `ReMosaiker_overlap_v2.py` instead of the older script, as it **preserves georeferencing** from an original GeoTIFF. It aligns the reassembled mask precisely with the source image’s projection and extent.

### Prerequisites
- Python
- GDAL
- NumPy
- glob

### Configuration
To use this script, specify:
- **Subtiles folder**: The path to the directory containing the subtiles.
- **Tile size**: The size of the subtiles (e.g., 256x256 pixels).
- **Overlap size**: The size of the overlap between the subtiles (e.g., 32 pixels).
- **Output file**: The path and name of the output TIFF file.

For example:

```python
subtiles_folder = "path_to_your_subtiles_folder"
original_geotiff = "path_to_original_geotiff"
output_file = "path_to_output_file.tif"
tile_size = 256
overlap_size = 32
reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file, original_geotiff)
```

## 2.5 (Optional) Overlaying Multiple Predictions

Using the script `AND_combiner.py`, it is possible to overlay masks predicted from multiple input images. This can be useful for capturing boundaries that are not present in one or another subsequent image, or vice versa.

Assuming the information to be retained and overlapped are the edges, with the pixel value 0, the script applies an AND operation between all predicted masks since the edge equals False and the field equals True.

## 2.6 Cleaning the Predicted Mask with Morphological Transformations

The script `unet_output_cleaner.py` is intended for cleaning the binary edge mask obtained from the UNet model after reconstructing the integral image. Before applying further segmentation algorithms such as watershed, it is essential to clean the binary mask of noise elements and non-actual edges using morphological transformations.

### Input

The input file is a binary mask with values 0 and 1 where:
- **0**: indicates the presence of an edge
- **1**: indicates the absence of an edge

### Output

The output file is a binary mask with values 0 and 255 where:
- **0**: indicates the presence of an edge
- **255**: indicates the absence of an edge

### Morphological Transformations Applied

The following morphological transformations are performed on the image to improve the mask quality for subsequent processing:

1. **Opening**: A morphological opening (erosion followed by dilation) using a circular structuring element with a radius of 2.5 pixels. This step helps remove small bright spots and connect small dark cracks.
2. **Small Object Removal**: Removal of small objects smaller than 200 pixels with connectivity of type 2. This step is crucial for eliminating isolated components that can be interpreted as noise.
3. **Small Object Removal on Negative**: Inversion of the mask to work on non-edges as objects, followed by small object removal from the inverted mask, setting a minimum size threshold of 80 pixels and connectivity of type 2.

> Our **`unet_output_cleaner.py`** includes:
> 1. **Morphological Opening**  
> 2. **Remove Small Objects**  
> 3. **Optional Hole-Filling**  
> 
> It reads/writes GeoTIFF and **preserves georeferencing**. Values remain **0 == edge** or **255 == non-edge** after cleaning.

### Usage

To use this script, specify the input file path as an argument to the command and run the script. The result will be a clean GeoTIFF file saved at the specified path.

## 3. **Watershed Segmentation and Polygonization**
   
## 3.1 Iterative Growing-Regions Segmentation with Watershed

### 3.1.1 Description

The script `iterativeWS_v2.py` implements an iterative approach to the watershed segmentation algorithm for identifying closed segments, ideal for subsequent polygonization of agricultural fields. Starting with identifying the largest fields and then moving to smaller ones, the iterative application of this method aims to minimize the risk of over-segmentation, improving the quality and accuracy of segmentation in complex agricultural image scenarios.

After generating a binary edge mask through a UNet model and subsequent morphological cleaning, the watershed algorithm is applied to identify closed areas representing agricultural fields. Using an iterative approach, the process starts with the search for local maxima at greater distances to identify the largest fields, then progressively decreases the minimum search distance, allowing for the identification of smaller fields without excessively fragmenting the larger ones.

> We have multiple watershed scripts:
> - **`iterativeWS_v2.py`**: Basic iterative approach.  
> - **`iterativeWS_v3.py`**: Fix for local maxima detection (shape mismatch).  
> - **`iterativeWS_v4.py`**: Introduces **parallelization** (`multiprocessing`) and uses `uint32` to handle >65535 segments.


### 3.1.2 Input

- **GeoTIFF File**: A cleaned binary edge mask, obtained and prepared through previous processes. Edges are represented with the value 0, while non-edge areas are represented with the value 255.

### 3.1.3 Output

- **GeoTIFF File**: A series of GeoTIFF files for each round of segmentation, containing unique labels for each identified field. Each label corresponds to a unique segment identified during that specific round of segmentation.

### 3.1.4 Iterative Watershed

1. **distance_transform_edt**: The distance transform from the binary image is calculated, serving to identify the central points of potential fields, finding the points farthest from the edges and respecting a minimum mutual distance.
2. **Local Maxima Search**: Using `peak_local_max` from `skimage.feature`, local maxima are identified, serving as markers for the watershed.
3. **Connected Component Analysis**: The local peaks are analyzed to define connected components, serving as initial markers for the watershed.
4. **Watershed Segmentation**: Using the binary mask as a mask and the identified markers as initializers, the watershed is applied to segment the image.
5. **Iteration with Decreasing Distances**: The process is repeated with decreasing minimum distances between local peaks to minimize over-segmentation.

## 3.2 Polygonization of Watershed Segments

This script converts closed and unique segments, identified through the watershed segmentation algorithm, into polygon vectors. The goal is to facilitate subsequent analysis and GIS operations by converting raster masks into more usable vector formats for mapping and agricultural monitoring applications.

### 3.2.1 Description

After completing the watershed segmentation of agricultural fields, the next step is to polygonize these segments. This script polygonizes the raster of unique segments, i.e., the unique labels generated by the watershed algorithm, into a vector format (Shapefile), which is more suitable for subsequent analysis.

The process uses the GDAL library to read an input raster representing the watershed segmentation mask and produces a Shapefile containing polygons corresponding to each unique segment identified.

### 3.2.2 Technical Details

- **Input Mask**: A raster where each unique value represents a distinct segment.
- **Shapefile**: A Shapefile containing polygons for each identified segment. Each polygon has a 'Label' attribute corresponding to the segment's label in the input raster.


> If you need **geometry simplification**, use **`polygonize_watershed_v2.py`**, which adds a `simplify_tolerance` parameter. That can reduce polygon complexity while preserving topology (via `SimplifyPreserveTopology()`).

## 4. **Automatic segmentation with pre-trained models**

### 4.1 Pre-trained Model v1 with Generalised Training Dataset:

To enhance generalizability, ResUnet is trained on a varied global dataset capturing a wide range of agricultural conditions and seasonal variations. This generalized model is tested across different regions and seasons. The training dataset is built by associating the same multitemporal Canny-generated segmentation mask with multiple 3-channel input images from different times of the season (NDVI, NDWI, B2). 

The training dataset v1 consists of about 90,000 images, each 256x256 pixels, sub-mosaicked from Sentinel-2 tiles specified in the table below.

![image](https://github.com/AlFe23/S2-AUTOMATIC-CROP-FIELD-SEGMENTATION-UNET-CANNY-WATERSHED/assets/105355911/0ee9f30e-076b-4fbc-93a5-ba3f3b940a1b)

![image](https://github.com/AlFe23/S2-AUTOMATIC-CROP-FIELD-SEGMENTATION-UNET-CANNY-WATERSHED/assets/105355911/e79a3d76-1604-4931-8d00-4d94c9ac1b2c)

### 4.2 Example of segmentation obtained on a new AOI, Modesto (CA), USA.

Below is reported an example of automatic processing executed on a test area of Modesto (CA), USA.

![image](https://github.com/AlFe23/S2-AUTOMATIC-CROP-FIELD-SEGMENTATION-UNET-CANNY-WATERSHED/assets/105355911/99d5221a-88b6-4a41-ab1e-63315f118dac)


### 4.3 Automated Inference Scripts

For **fully automated** U-Net or ResUNet predictions, we provide:

- **`inference_v3_solopred_auto.py`**  
  - Scans a **base directory** with multiple subfolders of **tiled input images**.  
  - Loads a specified model (either `.keras` or `.h5`) and runs inference **folder by folder**.  
  - Automatically saves predicted masks in appropriately named output directories (e.g., `_predicted` suffix).  
  - Especially convenient when handling large numbers of input directories without manually enumerating them.

**Usage Overview**  
1. **Set the `base_directory`** (contains multiple subfolders of tiles).  
2. **Provide your trained model path** (e.g., `U-Net-Weights-BFCE.keras`).  
3. **Run the script**:
   ```bash
   python inference_v3_solopred_auto.py
