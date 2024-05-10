# AUTOMATIC-CROP-FIELD-SEGMENTATION-USING-UNET-CANNY-WATERSHED
Completely automatic software for crop field segmentation using Sentinel-2 satellite images. This tool leverages a UNet architecture, trained on  multitemporal Canny filter images, and watershed algorithm, to deliver high-precision segmentation. Ideal for agricultural researchers and GIS specialists seeking efficient and scalable solutions.


## 1. Preparazione del Dataset con GEE

Tramite lo script per GEE ('Canny_Multitemporale_standard') si genera un dataset aventi le seguenti caratteristiche:

**Geotiff uint16 composto da 3 canali (B2, NDVI, NDWI)**, generato da immagini Sentinel-2 L2A multi-temporali completamente cloud-free, compressi LZW, proiettati in EPSG:4326 - WGS 84. Il numero di immagini per cui viene generato il download dipende dal numero di immagini cloud-free (cl.cov.<0.5%) disponibili nell’arco temporale specificato.

I tre canali sono scalati come segue:
- **B2, riflettanza blu**, con dinamica 0,1 scalata linearmente nell’intervallo 0,10000 già dal prodotto L2A.
- **NDVI** con dinamica -1,1 scalata linearmente nell’intervallo 0,65535 con la formula: ndvi_uint16 = (ndvi * 32767.5) + 32767.5
- **NDWI** con dinamica -1,1 scalata linearmente nell’intervallo 0,65535 con la formula: ndwi_uint16 = (ndwi * 32767.5) + 32767.5

Nella preparazione del dataset per il training UNet è importante riscalare i valori del dataset nell’intervallo 0,1; quindi si dividerà per 10000 il primo canale e per 65535 il secondo e terzo canale.

**Geotiff uint8 composto da 1 canale** che contiene la maschera generata sovrapponendo output di filtro Canny applicato a sequenza multitemporale di immagine Sentinel-2 L2A. Compresso LZW, proiettato in EPSG:4326 - WGS 84.
- Questa maschera è generata sommando tutti gli output di Canny ed i pixel hanno valori discreti compresi tra 0 e 255. Dovrà essere binarizzata applicando una threshold prima di proseguire con l’operazione di sub-tiling.

