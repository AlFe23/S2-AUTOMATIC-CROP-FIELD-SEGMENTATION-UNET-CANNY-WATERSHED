# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:26:07 2024

Inference and Evaluation Script for Sentinel-2 Crop Segmentation (v2)

This version extends the original inference script by **adding evaluation metrics**  
to assess model performance. It compares the predicted segmentation masks with **ground truth labels**  
and calculates various accuracy metrics.

Key Features:
- **Performs inference** using a pre-trained U-Net model.
- **Evaluates predictions** against ground truth labels.
- Computes **IoU, Precision, Recall, Dice Coefficient, and Accuracy**.
- Saves **detailed evaluation results** in a text file.
- Includes visualization of sample results.

Differences from the Previous Version:
- **Ground Truth Support:** Loads corresponding label images for evaluation.
- **New Metrics:** Computes accuracy, IoU, precision, recall, and Dice coefficient.
- **Evaluation Report:** Saves detailed scores for each image to `model_evaluation.txt`.
- **Visualization:** Displays test images, ground truth labels, and predictions.

Quick User Guide:
1. Modify `input_directory`, `labels_directory`, and `output_directory` in the script.
2. Ensure the **U-Net model** is available at `model_path`.
3. Run the script:
       python inference_v2.py
4. The output masks will be saved in `<output_directory>/predicted_masks/`.
5. Evaluation results are saved in `model_evaluation.txt`.

Dependencies:
Python packages: numpy, gdal (from osgeo), glob, os, tifffile, tensorflow (keras),  
scikit-learn (for IoU, precision, recall), matplotlib (for visualization)

License:
This code is released under the MIT License.

Author: Alvise Ferrari  

"""

"""




import os
import glob
import numpy as np
import tifffile as tiff
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

# # Let's use CPU for inference
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disables GPU visibility

# Paths to the directories containing new dataset images and labels
input_directory = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180420T094031_20180420T094644_T33TXF_B2_NDVI_NDWI_33TXF_20180420_tiles_woverlap'
labels_directory = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/T33TXF_canny_2018_NDVIth020_sigma1dot7_NDWIth020_sigmaNDWI2dot5_optimized_33TXF_20180420_tiles_woverlap'


output_directory = os.path.join(input_directory, "predicted_masks")

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define Dice coefficient and Dice loss
smooth = 1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_float32(y_true, y_pred):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Define metrics for evaluation
METRICS = [
    tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5), 
    tf.keras.metrics.BinaryAccuracy(threshold=0.5),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    dice_coef
]

###############################################################################################################################################
# Load the trained model
model = load_model('/mnt/h/Alvise/training_DS_A/weights_unet/BFC-2024-05-16-204727/U-Net-Weights-BFCE.h5', compile=False)
model.compile(optimizer='adam', metrics=METRICS)  # No need to specify loss during inference




###############################################################################################################################################
# Function to load and preprocess images and labels
def load_and_preprocess_images_and_labels(image_directory, label_directory):
    image_paths = glob.glob(os.path.join(image_directory, "*.tif"))
    images = []
    labels = []
    
    for image_path in image_paths:
        images.append(tiff.imread(image_path))
        label_path = os.path.join(label_directory, os.path.basename(image_path))
        labels.append(tiff.imread(label_path))

    images = np.array(images)
    labels = np.array(labels)

    # Normalize the images (ensure this matches training preprocessing)
    images[:,:,:,0] = images[:,:,:,0] / 10000.0  # Normalize B2 channel
    images[:,:,:,1] = images[:,:,:,1] / 65535.0  # Normalize NDVI channel
    images[:,:,:,2] = images[:,:,:,2] / 65535.0  # Normalize NDWI channel
    
    # Normalize the labels (consistent with training)
    labels = labels.astype(np.float32) / 255.0
    labels = np.expand_dims(labels, axis=-1)  # Ensure labels have the same shape as during training

    return images, labels, image_paths

def write_scores(image_directory, output_directory, average_accuracy, average_dc, average_iou, average_precision, average_recall, evaluation, iou_scores, prec_scores, recall_scores):
    image_paths = glob.glob(os.path.join(image_directory, "*.tif"))
    output_path_txt = os.path.join(output_directory, "model_evaluation.txt")
    
    with open(output_path_txt, "w") as file:
        file.write("\nAverages:\n")
        file.write(f"Average Binary Accuracy: {average_accuracy:.4f}\n")
        file.write(f"Average IoU: {average_iou:.4f}\n")
        file.write(f"Average Precision: {average_precision:.4f}\n")
        file.write(f"Average Recall: {average_recall:.4f}\n")
        file.write(f"Average Dice Coefficient: {average_dc:.4f}\n")
        
        file.write("Model Evaluation Results:\n")
        file.write("============================================\n")
        for metric, value in evaluation.items():
            file.write(f"{metric}: {value:.4f}\n")
        
        file.write("\nDetailed Scores:\n")
        file.write("============================================\n")
        file.write("Image_Path, IoU_Score, Precision_Score, Recall_Score\n")
        for i in range(len(image_paths)):
            image_path = image_paths[i]
            iou_score = iou_scores[i]
            prec_score = prec_scores[i]
            recall_score = recall_scores[i]
            line = f"{image_path}, {iou_score:.4f}, {prec_score:.4f}, {recall_score:.4f}\n"
            file.write(line)
        
        file.write("\nAverages:\n")
        file.write(f"Average IoU: {average_iou:.4f}\n")
        file.write(f"Average Precision: {average_precision:.4f}\n")
        file.write(f"Average Recall: {average_recall:.4f}\n")
    
    print(f"Model evaluation results have been written to {output_path_txt}")


        
        
    
# Load and preprocess new images and labels
new_images, true_labels, image_paths = load_and_preprocess_images_and_labels(input_directory, labels_directory)

# Evaluate the model
evaluation = model.evaluate(new_images, true_labels, return_dict=True)
print(f"Model evaluation: {evaluation}")

# Perform inference
predictions = model.predict(new_images)

# Convert predictions to binary masks
predicted_masks = (predictions > 0.5).astype(np.uint8)

# Save each predicted mask and compare with true label
iou_scores = []
prec_scores = []
recall_scores = []
dc_scores = []
acc_scores = []
for i, predicted_mask in enumerate(predicted_masks):
    filename = os.path.basename(image_paths[i])
    output_path = os.path.join(output_directory, f"predicted_{filename}")
    tiff.imwrite(output_path, predicted_mask)

    # Calculate IoU, Precision, and Recall for the predicted mask and the true label
    iou_score = jaccard_score(true_labels[i].flatten(), predicted_mask.flatten())
    prec_score = precision_score(true_labels[i].flatten(), predicted_mask.flatten())
    recall_score_ = recall_score(true_labels[i].flatten(), predicted_mask.flatten())
    dc_score = dice_coef_float32(true_labels[i], predicted_mask)
    acc_score = accuracy_score(true_labels[i].flatten(), predicted_mask.flatten())
    iou_scores.append(iou_score)
    prec_scores.append(prec_score)
    recall_scores.append(recall_score_)
    acc_scores.append(acc_score)
    dc_scores.append(dc_score)
    
# Calculate average IoU, Precision, and Recall
average_iou = np.mean(iou_scores)
average_precision = np.mean(prec_scores)
average_recall = np.mean(recall_scores)
average_accuracy = np.mean(acc_scores)
average_dc = np.mean(dc_scores)
print(f"Average Accuracy: {average_accuracy}")
print(f"Average IoU: {average_iou}")
print(f"Average Precision: {average_precision}")
print(f"Average Recall: {average_recall}")
print(f"Average dice coeff.: {average_dc}")
print(f"Inference done. Predicted masks saved to: {output_directory}")

write_scores(input_directory, output_directory, average_accuracy, average_dc, average_iou, average_precision, average_recall, evaluation, iou_scores, prec_scores, recall_scores)

# Visualize a few test images, their true labels, and predicted masks
num_samples = 30  # Number of samples to visualize

# Find indices of elements sorted in ascending order
sorted_indices = np.argsort(prec_scores)
# Select the last n indices to get the highest values
top_indices = sorted_indices[-num_samples:]
# Since the indices are in ascending order, reverse them to have descending order
top_indices = top_indices[::-1]

for index in top_indices:
    print(prec_scores[index])
    test_img = new_images[index]
    true_label = true_labels[index]
    pred_mask = predicted_masks[index]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(test_img)
    plt.title('Test Image')
    plt.subplot(1, 3, 2)
    plt.imshow(true_label, cmap='gray')
    plt.title('True Label')
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask')
    iou_score= iou_scores[index]
    plt.text(-600,320 , f'IoU: {iou_score:.3f}', ha='left', va='bottom', fontsize=20)
    prec_score= prec_scores[index]
    plt.text(-470,320 , f'Precision: {prec_score:.3f}', ha='left', va='bottom', fontsize=20)
    rec_score= recall_scores[index]
    plt.text(-280,320 , f'Recall: {rec_score:.3f}', ha='left', va='bottom', fontsize=20)
    plt.show()
