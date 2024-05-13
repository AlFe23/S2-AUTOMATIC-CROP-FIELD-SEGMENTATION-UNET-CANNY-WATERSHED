# AUTOMATIC-CROP-FIELD-SEGMENTATION-USING-UNET-CANNY-WATERSHED
Automatic software for crop field segmentation using Sentinel-2 satellite images. This tool leverages a UNet architecture, trained on  multitemporal Canny filter images, and watershed algorithm, to deliver high-precision segmentation. Ideal for agricultural researchers and GIS specialists seeking efficient and scalable solutions.


## 1.1 Preparazione del Dataset con GEE

Tramite lo script per GEE `Canny_Multitemporale_standard` si genera un dataset aventi le seguenti caratteristiche:

**Geotiff uint16 composto da 3 canali (B2, NDVI, NDWI)**, generato da immagini Sentinel-2 L2A multi-temporali completamente cloud-free, compressi LZW, proiettati in EPSG:4326 - WGS 84. Il numero di immagini per cui viene generato il download dipende dal numero di immagini cloud-free (cl.cov.<0.5%) disponibili nell’arco temporale specificato.

I tre canali sono scalati come segue:
- **B2, riflettanza blu**, con dinamica 0,1 scalata linearmente nell’intervallo 0,10000 già dal prodotto L2A.
- **NDVI** con dinamica -1,1 scalata linearmente nell’intervallo 0,65535 con la formula: ndvi_uint16 = (ndvi * 32767.5) + 32767.5
- **NDWI** con dinamica -1,1 scalata linearmente nell’intervallo 0,65535 con la formula: ndwi_uint16 = (ndwi * 32767.5) + 32767.5

Nella preparazione del dataset per il training UNet è importante riscalare i valori del dataset nell’intervallo 0,1; quindi si dividerà per 10000 il primo canale e per 65535 il secondo e terzo canale.

**Geotiff uint8 composto da 1 canale** che contiene la maschera generata sovrapponendo output di filtro Canny applicato a sequenza multitemporale di immagine Sentinel-2 L2A. Compresso LZW, proiettato in EPSG:4326 - WGS 84.
- Questa maschera è generata sommando tutti gli output di Canny ed i pixel hanno valori discreti compresi tra 0 e 255. Dovrà essere binarizzata applicando una threshold prima di proseguire con l’operazione di sub-tiling.

## 1.2 Binarizzazione Maschera di Canny Multitemporale
Una volta ottenute le Maschere multitemporali di Canny, è necessario binarizzarle tramite lo script 'canny_binarizer.py' prima di poterle utilizzare per l'addestramento della U-Net'
Questo script è un estratto dello script `canny_cleaner_v3.py`, solo la prima parter dove viene effettuato il thresholding è mantenuta in questa parte.

**Input:**
   - Maschera Canny Multitemporale uint8 in scala di grigi (ottenuta da GEE)

**Output:**
   - Maschera di Canny binarizzata (0 == bordo ; 255 == non bordo)

## 1.3 Suddivisione training i/o in Sub-Tiles 

Lo script `new_subtiler_wOverlap.py` è progettato per preparare le immagini in tile da 256x256 pixel per il training della rete U-Net, partendo da immagini binarizzate con `canny_binarizer.py`. Lo script supporta immagini sia a 1 che a 3 canali e salva le subtiles in una nuova cartella generata automaticamente. Questa operazione facilita la creazione di un dataset composto da diverse immagini di input.

**Funzionalità:**
- **Supporto Multicanale**: Gestisce immagini a 1 o 3 canali.
- **Generazione Automatica di Cartelle**: Crea automaticamente una cartella nella stessa directory dell'immagine di input.
- **Naming Intelligente**: Le subtiles vengono nominate in modo sistematico basato sulla loro posizione nella griglia di tile, con formati come `subtile_0_0`, `subtile_0_N`, fino a `subtile_M_N`; dove N è pari al numero di colonne dell'immagine diviso per 256, mentre M è pari al numero di righe dell'immagine di input diviso per 256.
- **Gestione dell'Overlap**: Permette l'estrazione di subtiles considerando un overlap tra di esse.

Per utilizzare lo script, è necessario specificare il file di input, la dimensione dei tile e la dimensione dell'overlap. 

## 1.3 Rinominazione finale per input multipli 

Al fine di poter costituire un dataset con input multipli, a cui però corrisponde sempre lo stesso output 
(ricordiamo che la maschera di Canny è costituita sovrapponendo maschere ottenute applicando l'omonimo filtro a immagini multiple),
si è definita la semplice funzione `add_prefix_to_files` che aggiunge un determinato prefisso ai nomi di tutti i file contenuti 
all'interno di una cartella specificata. In questa fase sperimentale sarà necessario aggiungere tale prefisso
a tutte le cartelle contenenti le subtile di input; inoltre sarà necessario moltiplicare la cartella contenente 
le subtile di output, per un numero di volte pari al numero delle immagini di input, ed applicare fittizziamente
il prefisso corrispondente a ciascun input, a ognuna delle cartelle di output. Infine tutte le coppie i/o possono 
essere organizzate in una unica directory che verrà utilizzata per training o fine tuning.

- **Funzionamento:** Basta specificare il percorso della directory e il prefisso desiderato per processare tutti i file contenuti.


