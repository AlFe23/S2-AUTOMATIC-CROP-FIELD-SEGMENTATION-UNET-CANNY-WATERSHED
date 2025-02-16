# -*- coding: utf-8 -*-
"""
Created on 8/30/2024

@author: Alvise Ferrari

Versione v3: Aggiornamenti rispetto alla versione v2
- Supporto per il caricamento di modelli in entrambi i formati: 
  1. Formato `.keras` (nuovo formato di salvataggio dei modelli di Keras).
  2. Formato `.h5` (vecchio formato di salvataggio dei modelli di Keras).
- Implementazione di una funzione `load_trained_model` per caricare e compilare modelli in modo flessibile:
  - Riconoscimento automatico del formato del modello (`.keras` o `.h5`) e applicazione della logica di caricamento appropriata.
  - Gestione dei custom layers, come `Conv2DTranspose`, per garantire la compatibilità con modelli `.h5` salvati in versioni precedenti di Keras.
- Aggiunta di `custom_objects` per il caricamento di modelli `.h5` con layers non standard o parametri non supportati nelle nuove versioni di Keras.
- Compilazione del modello dopo il caricamento con un ottimizzatore (Adam), una funzione di perdita (`binary_crossentropy`) e le metriche definite (`METRICS`).
- Modifica delle funzioni di perdita e metriche per utilizzare `tf.keras.backend` direttamente per evitare errori di attributo e garantire la compatibilità con diverse versioni di TensorFlow/Keras.
- Mantenimento della funzione `dice_coef_float32` per garantire la compatibilità con i dati in formato `float32` e prevenire errori numerici durante l'inferenza.
- Miglioramento della robustezza dello script di inferenza per garantire la compatibilità con diverse versioni di Keras e TensorFlow.
- Miglioramenti nella documentazione e struttura del codice per una maggiore leggibilità e manutenzione.
"""

"""
Inference and Evaluation Script for Sentinel-2 Crop Segmentation (v3)

This version improves inference by **supporting both `.keras` and `.h5` model formats**,  
ensuring compatibility with different versions of TensorFlow/Keras.  
It also includes optimizations for **loading custom layers**.

Key Features:
- **Supports model formats:** Loads models in `.keras` (new format) and `.h5` (legacy format).
- **Handles custom layers**: Ensures compatibility with older `.h5` U-Net models.
- **Performs inference and evaluation** on Sentinel-2 image tiles.
- Computes **IoU, Precision, Recall, Dice Coefficient, and Accuracy**.
- Saves **detailed evaluation results** in a text file.
- **Optimized model loading** to prevent compatibility issues.

Differences from the Previous Version:
- **Dual Model Format Support:** Loads both `.keras` and `.h5` models automatically.
- **Custom Layer Handling:** Fixes compatibility issues with `Conv2DTranspose` layers in older models.
- **Robust Preprocessing:** Improved normalization and label handling.
- **Enhanced Logging:** Provides detailed feedback on model loading and inference.

Quick User Guide:
1. Modify `input_directory`, `labels_directory`, and `output_directory` in the script.
2. Ensure the **U-Net model** (either `.keras` or `.h5`) is available at `model_path`.
3. Run the script:
       python inference_v3.py
4. The output masks will be saved in `<output_directory>/predicted_masks/`.
5. Evaluation results are saved in `model_evaluation.txt`.

Dependencies:
Python packages: numpy, gdal (from osgeo), glob, os, tifffile, tensorflow (keras),  
scikit-learn (for IoU, precision, recall), matplotlib (for visualization)

License:
This code is released under the MIT License.

Author: Alvise Ferrari  
 
"""


import os
import glob
import numpy as np
import tifffile as tiff
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

# Disabilita la visibilità della GPU per eseguire l'inferenza su CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Percorsi ai dataset di immagini e label
input_directory = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/20180719T094031_20180719T094430_T33TXF_B2_NDVI_NDWI_33TXF_20180719_tiles_woverlap'
labels_directory = '/mnt/h/Alvise/CIMA_cooperation/UNET_DATASET_CIMA/2018_33TXF/T33TXF_canny_2018_NDVIth020_sigma1dot7_NDWIth020_sigmaNDWI2dot5_optimized_thresh_33TXF_20180719_tiles_woverlap'

output_directory = os.path.join(input_directory, "predicted_masks")

# Assicura l'esistenza della cartella di output
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

import os
import glob
import numpy as np
import tifffile as tiff
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K  # Usato solo per le funzioni che sono sicure
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

# Funzione del coefficiente di Dice utilizzando il backend di TensorFlow direttamente
smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_float32(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, 'float32'))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Definisci le metriche per la valutazione
METRICS = [
    keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5), 
    keras.metrics.BinaryAccuracy(threshold=0.5),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    dice_coef
]

###############################################################################################################################################
# Funzione per caricare il modello addestrato
def load_trained_model(model_path):
    if model_path.endswith('.keras'):
        # Caricamento diretto del modello nel nuovo formato .keras
        model = load_model(model_path, compile=False)
    elif model_path.endswith('.h5'):
        # Gestione del caricamento per modelli salvati in formato .h5 che potrebbero non essere compatibili direttamente
        from keras.layers import Conv2DTranspose
        
        # Definisci un layer personalizzato per gestire l'argomento 'groups' non riconosciuto in alcuni modelli salvati in formato .h5
        class CustomConv2DTranspose(Conv2DTranspose):
            def __init__(self, *args, **kwargs):
                if 'groups' in kwargs:
                    kwargs.pop('groups')  # Rimuovi l'argomento 'groups' se presente
                super(CustomConv2DTranspose, self).__init__(*args, **kwargs)
        
        # Carica il modello con la gestione personalizzata del layer
        model = load_model(model_path, custom_objects={'Conv2DTranspose': CustomConv2DTranspose}, compile=False)
    else:
        raise ValueError("Unsupported model format. Only '.keras' and '.h5' formats are supported.")
    
    # Compila il modello con ottimizzatore, funzione di perdita e metriche appropriate
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='binary_crossentropy',  # Funzione di perdita utilizzata durante il training
        metrics=METRICS
    )
    
    return model

# Specifica il percorso del modello addestrato
model_path = '/mnt/h/Alvise/training_DS_A/weights_unet/BFC-2024-05-16-204727/U-Net-Weights-BFCE.h5'  # o .keras

# Carica il modello addestrato
model = load_trained_model(model_path)


###############################################################################################################################################
# Funzione per caricare e preprocessare immagini e label
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

    # Normalizza le immagini (assicurati che corrisponda al preprocessing del training)
    images[:,:,:,0] = images[:,:,:,0] / 10000.0  # Normalizza il canale B2
    images[:,:,:,1] = images[:,:,:,1] / 65535.0  # Normalizza il canale NDVI
    images[:,:,:,2] = images[:,:,:,2] / 65535.0  # Normalizza il canale NDWI
    
    # Normalizza le label (coerente con il training)
    labels = labels.astype(np.float32) / 255.0
    labels = np.expand_dims(labels, axis=-1)  # Assicura che le label abbiano la stessa forma del training

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

# Carica e preprocessa nuove immagini e label
new_images, true_labels, image_paths = load_and_preprocess_images_and_labels(input_directory, labels_directory)

# Valuta il modello
evaluation = model.evaluate(new_images, true_labels, return_dict=True)
print(f"Model evaluation: {evaluation}")

# Esegui inferenza
predictions = model.predict(new_images)

# Converti le predizioni in maschere binarie
predicted_masks = (predictions > 0.5).astype(np.uint8)

# Salva ogni maschera predetta e confronta con la label vera
iou_scores = []
prec_scores = []
recall_scores = []
dc_scores = []
acc_scores = []

for i, predicted_mask in enumerate(predicted_masks):
    filename = os.path.basename(image_paths[i])
    output_path = os.path.join(output_directory, f"predicted_{filename}")
    tiff.imwrite(output_path, predicted_mask)

    # Calcola IoU, Precision, e Recall per la maschera predetta e la label vera
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
    
# Calcola valori medi di IoU, Precision, e Recall
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

# Visualizza alcune immagini di test, le loro label vere, e le maschere predette
num_samples = 30  # Numero di campioni da visualizzare

# Trova gli indici degli elementi ordinati in ordine crescente
sorted_indices = np.argsort(prec_scores)
# Seleziona gli ultimi n indici per ottenere i valori più alti
top_indices = sorted_indices[-num_samples:]
# Poiché gli indici sono in ordine crescente, inverti per avere ordine decrescente
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
