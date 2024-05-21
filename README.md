# AUTOMATIC-CROP-FIELD-SEGMENTATION-USING-UNET-CANNY-WATERSHED
Automatic software for crop field segmentation using Sentinel-2 L2A images. This tool leverages a UNet architecture, trained on  multitemporal canny edges masks, and watershed algorithm, to deliver high-precision crop field segmentation. Ideal for agricultural researchers and GIS specialists seeking efficient and scalable solutions.


### Indice
1. **Preparazione del Dataset**
   - 1.1 Preparazione del Dataset con GEE
   - 1.2 Binarizzazione della Maschera di Canny Multitemporale
   - 1.3 Suddivisione Training I/O in Sub-Tiles

2. **Segmentazione con UNet**
   - 2.1 U-Net Training
     - 2.1.1 Funzionalità
     - 2.1.2 Prerequisiti
     - 2.1.3 Preparazione del Dataset
     - 2.1.4 Architettura della Rete
     - 2.1.5 Configurazione dell'Ottimizzatore
     - 2.1.6 Training
     - 2.1.7 Gestione degli Input
     - 2.1.8 Salvataggio dei Log di TensorBoard e dei Pesi del Modello
     - 2.1.9 Valutazione e Metriche
     - 2.1.10 Visualizzazione dei Risultati
   - 2.2 Fine-Tuning di U-Net Pre-addestrata
     - 2.2.1 Descrizione
     - 2.2.2 Differenze Chiave dal Training da Zero
     - 2.2.3 Prerequisiti
     - 2.2.4 Configurazione del Fine-Tuning
     - 2.2.5 Gestione degli Input
     - 2.2.6 Log e Salvataggio dei Pesi
     - 2.2.7 Output del Training
   - 2.3 Inferenza con U-Net su Nuovi Dati
     - 2.3.1 Prerequisiti
     - 2.3.2 Configurazione
     - 2.3.3 Preparazione del Dataset
     - 2.3.4 Esecuzione dell'Inferenza
     - 2.3.5 Salvataggio delle Maschere Predette
   - 2.4 Ricostruzione di Immagini Integrali da Subtiles Predette con U-Net
   - 2.5 Pulizia della Maschera Predetta con Trasformazioni Morfologiche

3. **Segmentazione Watershed e Poligonizzazione**
   - 3.1 Segmentazione Growing-Regions con Watershed Iterativo
     - 3.1.1 Descrizione
     - 3.1.2 Input
     - 3.1.3 Output
     - 3.1.4 Watershed Iterativo
   - 3.2 Poligonizzazione dei Segmenti Watershed
     - 3.2.1 Descrizione
     - 3.2.2 Dettagli Tecnici




## 1. **Preparazione del Dataset**

## 1.1 Preparazione del Dataset con GEE

Tramite lo script per GEE `Canny_Multitemporale_standard` si genera un dataset aventi le seguenti caratteristiche:

**Geotiff uint16 composto da 3 canali (B2, NDVI, NDWI)**, generato da immagini Sentinel-2 L2A multi-temporali completamente cloud-free, compressi LZW, proiettati in EPSG:4326 - WGS 84. Il numero di immagini per cui viene generato il download dipende dal numero di immagini cloud-free (cl.cov.<0.5%) disponibili nell’arco temporale specificato.

I tre canali sono scalati come segue:
- **B2, riflettanza blu**, con dinamica 0,1 scalata linearmente nell’intervallo 0,10000 già dal prodotto L2A.
- **NDVI** con dinamica -1,1 scalata linearmente nell’intervallo 0,65535 con la formula: ndvi_uint16 = (ndvi * 32767.5) + 32767.5
- **NDWI** con dinamica -1,1 scalata linearmente nell’intervallo 0,65535 con la formula: ndwi_uint16 = (ndwi * 32767.5) + 32767.5

Nella preparazione del dataset per il training UNet è importante riscalare i valori del dataset nell’intervallo [0-1]; quindi si dividerà per 10000 il primo canale e per 65535 il secondo e terzo canale.

**Geotiff uint8 composto da 1 canale** che contiene la maschera generata sovrapponendo output di filtro Canny applicato a sequenza multitemporale di immagine Sentinel-2 L2A. Compresso LZW, proiettato in EPSG:4326 - WGS 84.
- Questa maschera è generata sommando tutti gli output di Canny ed i pixel hanno valori discreti compresi tra 0 e 255. Dovrà essere binarizzata applicando un threshold prima di proseguire con l’operazione di sub-tiling.

## 1.2 Binarizzazione Maschera di Canny Multitemporale
Una volta ottenute le Maschere multitemporali di Canny, è necessario binarizzarle tramite lo script 'canny_binarizer.py' prima di poterle utilizzare per l'addestramento della U-Net'
Questo script è un estratto dello script `canny_cleaner_v3.py`, solo la prima parte dove viene effettuato il thresholding è mantenuta in questa parte.

**Input:**
   - Maschera Canny Multitemporale uint8 in scala di grigi (ottenuta da GEE)

**Output:**
   - Maschera di Canny binarizzata (0 == bordo ; 255 == non bordo)

## 1.3 Suddivisione training i/o in Sub-Tiles 

Lo script `new_subtiler_wOverlap.py` è progettato per preparare le immagini in tile da 256x256 pixel per il training della rete U-Net, partendo da immagini binarizzate con `canny_binarizer.py`. Lo script supporta immagini sia a 1 che a 3 canali e salva le subtiles in una nuova cartella generata automaticamente. Questa operazione facilita la creazione di un dataset composto da diverse immagini di input.

**Funzionalità:**
- **Supporto Multicanale**: Gestisce immagini a 1 o 3 canali.
- **Generazione Automatica di Cartelle**: Crea automaticamente una cartella nella stessa directory dell'immagine di input.
- **Naming Intelligente**: Le subtiles vengono nominate in modo sistematico basato sulla loro posizione nella griglia di tiles, con formati come `prefixes_subtile_0_0`, `prefixes_subtile_0_N`, fino a `prefixes_subtile_M_N`; dove `N` è pari al numero di colonne dell'immagine diviso per 256, mentre `M` è pari al numero di righe dell'immagine di input diviso per 256.
- **Gestione dell'Overlap**: Permette l'estrazione di subtiles considerando un overlap tra di esse.

Per utilizzare lo script, è necessario specificare il file di input, la dimensione delle tiles e la dimensione dell'overlap. 

Rispetto alla versione precedente, questa versione aggiunge automaticamente un prefisso al nome delle tile, fornito in una lista di prefissi corrispondenti alla lista delle immagini di input.

**Esempio:**
```python
input_files = [
    input_img_path_1,
    input_img_path_2, 
    input_img_path_3, 
    input_img_path_4, ...
]

prefix_name_list = ['tilename1_data1', 'tilename1_data2', 'tilename1_data3', 'tilename1_data4', ...]
```

nota: la data del prefisso è aggiunta anche all'interno del nome della cartella contenente le subtile di ogni immagine, questo per far si che venga prodotta una cartella diversa per ogni input canny, che è ripetuto in numero di volte pari al numero di immagini input a 3ch.

## 2. **Segmentazione con UNet**
 
## 2.1 U-Net Training 

Lo script Python `unet_v2_1.py` utilizza TensorFlow per addestrare una rete neurale U-Net. La rete è ottimizzata per funzionare su hardware GPU, sfruttando la precisione mista e la gestione dinamica della memoria per migliorare efficienza e prestazioni.

### 2.1.1 Funzionalità
- **Training con Precisione Mista**: Impiega tipi di dati a 16-bit e 32-bit durante il training, riducendo il consumo di memoria e accelerando il processo.
- **Gestione Dinamica della Memoria GPU**: Configura TensorFlow per allocare dinamicamente la memoria GPU, prevenendo l'allocazione di tutta la memoria inizialmente.
- **Logging e Checkpointing Avanzati**: Implementa checkpointing automatico e logging dettagliato per monitorare il training e facilitare la ripresa del processo in caso di interruzione.


### 2.1.2 Prerequisiti
- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- tifffile
- scikit-image
- PIL


### 2.1.3 Preparazione del Dataset

In primis tutte le coppie i/o di subtiles prodotte nei passi precedenti vanno spostate in cartelle uniche per gli input e per gli outpu(labels), nella sezione input del codice UNet vengono infatti specificati due path.

Il dataset deve essere composto da immagini  a 3 canali e le corrispondenti maschere di segmentazione. Le immagini vengono caricate e pre-elaborate come segue:

- **Caricamento**: Le immagini e le maschere vengono caricate utilizzando `tifffile` per supportare formati TIFF a canali plurimi.
- **Normalizzazione**: Le immagini vengono normalizzate dividendo per il massimo valore specifico del canale per portarle nel range [0, 1].
- **Splitting Manuale**: Il dataset viene diviso in set di training e validazione utilizzando una porzione predefinita del dataset originale per la validazione. Questo splitting è basato su un indice casuale che separa il 10% dei dati per la validazione.

### 2.1.4 Architettura della Rete
La rete U-Net è configurata con blocchi convoluzionali che comprendono:

- Doppie Convoluzioni 2D con attivazione ReLU e normalizzazione.
- Max Pooling  per ridurre le dimensioni spaziali.
- Convoluzioni Trasposte per l'upsampling e ricomposizione delle dimensioni iniziali.
- Concatenazione con i corrispondenti output del percorso di contrazione per preservare le informazioni contestuali.

L'ultimo strato utilizza una convoluzione 2D per mappare le caratteristiche all'immagine di segmentazione finale.

### 2.1.5 Configurazione dell'Ottimizzatore
Il modello utilizza l'ottimizzatore Adam con i seguenti parametri:

- **Learning Rate**: 0.001 iniziale, con riduzione dinamica basata sul plateau del validation loss, fino a 10e-5
- **Loss Function**: Binary Focal Crossentropy, per gestire il disbilanciamento tra le classi.

### 2.1.6 Training
Il modello viene addestrato con le seguenti specifiche:

- **Batch Size**: 4 (variabile sulla base dell'hardware a disposizione)
- **Epochs**: 100 (70 sono più che sufficienti)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint per salvare i migliori pesi, e TensorBoard per il monitoring.


### 2.1.7 Gestione degli Input

Per l'addestramento della rete U-Net, gli input devono essere organizzati in modo specifico. Il codice prevede la specificazione di directory per le immagini di input e per le maschere (labels) attraverso le variabili `input_images_directory` e `canny_masks_directory`. Queste directory devono contenere file immagine nel formato TIFF, pronti per essere caricati e preprocessati dallo script.

Ecco come vengono specificate nel codice:

```python
base_directory = "path/to/dataset/base_directory"
input_images_directory = os.path.join(base_directory, "input_images/*.tif")
canny_masks_directory = os.path.join(base_directory, "labels/*.tif")
```

Queste variabili puntano alle directory dove sono situati rispettivamente gli input e le maschere.

### 2.1.8 Salvataggio dei Log di TensorBoard e dei Pesi del Modello

Il salvataggio dei log e dei pesi del modello è essenziale per monitorare il progresso dell'addestramento e per poter riprendere l'addestramento da un certo punto in caso di interruzione. Nel tuo script, queste operazioni sono gestite come segue:

- **Directory dei Log**: I log di TensorBoard vengono salvati in una sottodirectory all'interno della directory `logs_unet`, la quale è a sua volta situata nella `base_directory`. La sottodirectory per ogni sessione di training è nominata con un timestamp per garantire l'unicità e per facilitare l'identificazione della sessione di addestramento corrispondente.

```python
logs_directory = os.path.join(base_directory, "logs_unet")
current_run_directory = os.path.join(logs_directory, f"BFC-{timestamp}")
os.makedirs(current_run_directory, exist_ok=True)
```

- **Directory dei Pesi**: Analogamente ai log, i pesi del modello vengono salvati in una directory chiamata `weights_unet`, situata anch'essa nella `base_directory`. Anche qui, una sottodirectory specifica per ogni sessione di training viene creata usando lo stesso timestamp utilizzato per i log.

```python
weights_directory = os.path.join(base_directory, "weights_unet")
weights_run_directory = os.path.join(weights_directory, f"BFC-{timestamp}")
model_checkpoint_path = os.path.join(weights_run_directory, 'U-Net-Weights-BFCE.h5')
os.makedirs(weights_run_directory, exist_ok=True)
```

Ogni modello viene salvato con il nome `U-Net-Weights-BFCE.h5` all'interno della sua specifica sottodirectory identificata dal timestamp, facilitando così il recupero dei pesi per sessioni future o per l'analisi post-training.

Questa struttura organizzativa non solo mantiene i file ordinati e facilmente accessibili, ma permette anche di eseguire più sessioni di addestramento in parallelo senza rischio di sovrascrivere dati importanti.

### 2.1.9 Valutazione e Metriche
Il modello viene valutato utilizzando metriche standard come IoU (Jaccard index), Precision, Recall e Dice Coefficient. Queste metriche sono calcolate per ogni batch e visualizzate al termine di ogni epoca per il set di training e di validazione.

### 2.1.10 Visualizzazione dei Risultati
Grafici della precisione, Loss, IoU e altre metriche vengono generati usando Matplotlib per visualizzare la performance del modello nel corso del training.



## 2.2 Fine-Tuning di U-Net pre-addestrata

### 2.2.1 Descrizione
Lo script Python `Fine_tuning.py`  è destinato al fine-tuning di una rete U-Net già addestrata, per migliorarne la precisione su un nuovo dataset. A differenza del training da zero, il fine-tuning adatta un modello pre-addestrato per affinare ulteriormente le sue capacità predittive, sfruttando il sapere già acquisito.

### 2.2.2 Differenze Chiave dal Training da Zero
- **Punto di Partenza**: Inizia con un modello che ha già imparato pattern significativi da un dataset simile o correlato, invece di iniziare con pesi casuali.
- **Learning Rate**: Utilizza un learning rate molto più basso (0.0001 nel fine-tuning vs. tipicamente 0.001 nel training da zero), per fare aggiustamenti più sottili ai pesi e prevenire la perdita di informazioni apprese precedentemente.
- **Epochs**: Generalmente, meno epoche sono necessarie nel fine-tuning perché il modello non deve imparare da capo.
- **Callbacks**: Configurazioni simili per le callbacks, ma con una soglia di tolleranza più bassa per l'early stopping e il reduce learning rate, riflettendo l'aspettativa di progressi più marginali e raffinati.

### 2.2.3 Prerequisiti
Elenco delle dipendenze software come Python, TensorFlow, OpenCV, etc., rimane invariato rispetto allo script di training da zero.

### 2.2.4 Configurazione del Fine-Tuning
#### Caricamento del Modello
Carica un modello pre-addestrato specificando il percorso ai pesi salvati. Questo passo è cruciale per iniziare il fine-tuning su una base già solida.

####  Preparazione dei Dati
- **Input e Normalizzazione**: Le immagini sono caricate e normalizzate nello stesso modo del training da zero, ma la coerenza con il preprocessing del modello originale è vitale.

####  Configurazione della Rete
- **Ottimizzatore**: Usa `Adam` con un `learning_rate` di 0.0001 per minimizzare il rischio di disturbare l'apprendimento pregresso.
- **Loss e Metriche**: Configura le funzioni di perdita e le metriche come nel training originale per mantenere la coerenza nelle valutazioni.

### 2.2.5 Gestione degli Input
Specifica i percorsi e le procedure di caricamento e normalizzazione, come descritto nella sezione di preparazione dei dati dello script di training da zero.

### 2.2.6 Log e Salvataggio dei Pesi
- **Logs**: Salva i log di TensorBoard per monitorare il fine-tuning secondo la stessa logica di directory descritta nello script di training da zero.
- **Salvataggio dei Pesi**: I pesi sono salvati in una directory unica identificata da un timestamp, per distinguere i set di pesi fine-tuned da quelli del training originale.

### 2.2.7 Output del Training
Genera e visualizza grafici di accuratezza, perdita, IoU, coefficiente di Dice, precisione e recall, permettendo una comparazione diretta con i risultati pre-fine-tuning.

## 2.3 Inferenza con U-Net su Nuovi Dati

Lo script `inference.py` esegue l'inferenza su un nuovo set di dati utilizzando un modello U-Net precedentemente addestrato.  L'obiettivo è generare maschere di segmentazione per singole immagini Sentinel-2 di input, applicando lo stesso preprocessing utilizzato nel training per garantire la coerenza dei risultati.

### 2.3.1 Prerequisiti
- Python 3.x
- TensorFlow 2.x
- tifffile
- NumPy

### 2.3.2 Configurazione
- **Disabilitazione GPU**: Per impostazione predefinita, lo script esegue l'inferenza utilizzando la CPU. Questo può essere modificato rimuovendo o commentando la linea `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`.
- **Modello Pre-addestrato**: Assicurati che il percorso al modello salvato sia corretto e accessibile dallo script.

### 2.3.3 Preparazione del Dataset
Il dataset per l'inferenza deve essere preparato con lo stesso preprocessing usato per il training:
- **Caricamento Immagini**: Le immagini devono essere caricate dalla directory specificata.
- **Normalizzazione**: Le immagini vengono normalizzate nello stesso modo in cui sono state normalizzate durante il training.

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

### 2.3.4 Esecuzione dell'Inferenza
- **Carica e Preprocessa le Nuove Immagini**: Utilizza la funzione `load_and_preprocess_images` per preparare i dati.
- **Esecuzione del Modello**: Applica il modello alle immagini preprocessate per generare le previsioni.

### 2.3.5 Salvataggio delle Maschere Predette
Ogni maschera predetta viene salvata in una directory di output specificata, utilizzando un formato che facilita l'identificazione:

```python
for i, predicted_mask in enumerate(predicted_masks):
    filename = os.path.basename(image_paths[i])
    output_path = os.path.join(output_directory, f"predicted_{filename}")
    tiff.imwrite(output_path, predicted_mask)
```
Le subtile predette avranno quindi la stessa logica di nominazione che hanno quelle di input. Questo è importante perchè è tramite la nomenclatura delle subtile che sarà facilmente ricostruire la maschera di output completa con lo script `ReMosaicker_overlap.py` descritto successivamente.
Al termine dell'inferenza, le maschere predette saranno salvate nella directory specificata. Lo script fornisce feedback stampando il percorso della directory di output.



## 2.4 Ricostruzione di Immagini integrali da Subtiles Predette con U-Net

### Descrizione
Lo script `ReMosaiker_overlap_v2.py` Python è progettato per ricomporre un'immagine completa a partire da subtiles generate dal processo di inferenza utilizzando la rete U-Net. `ReMosaiker_overlap_v2.py` gestisce la ricomposizione considerando gli overlap tra le subtiles basandosi sulla denominazione delle subtiles. In particolare la denominazione delle subtile contenute nella cartella di input deve essere del seguente tipo:

- predicted_20211018_subtile_m_n.tif

Dove ricordiamo che 'm' ed 'n' rappresentano rispettivamente la posizione in termine di riga e colonna del mosaico di tiles da 256x256 pixel.

Nota: lo script legge le posizione a partire dalla terza posizione, dove le posizioni sono separate da '_'

### Prerequisiti
- Python
- GDAL
- NumPy
- glob

### Configurazione
Per utilizzare questo script, è necessario specificare:
- **cartella delle Subtiles**: Il percorso della directory contenente le subtiles.
- **dimensione delle Tiles**: La dimensione delle subtiles (ad esempio, 256x256 pixels).
- **dimensione dell'Overlap**: La dimensione dell'overlap tra le subtiles (ad esempio, 32 pixels).
- **file di Output**: Il percorso e il nome del file TIFF di output.

ad esempio:

```python
subtiles_folder = "path_to_your_subtiles_folder"
original_geotiff = "path_to_original_geotiff"
output_file = "path_to_output_file.tif"
tile_size = 256
overlap_size = 32
reconstruct_image(subtiles_folder, tile_size, overlap_size, output_file, original_geotiff)
```



## 2.5 Pulizia della maschera predetta con trasformazioni morfologiche

Lo script `unet_output_cleaner.py` è destinato alla pulizia della maschera binaria dei bordi ottenuta dal modello UNet, dopo aver ricostruito l'immagine integrale. Prima di applicare ulteriori algoritmi di segmentazione come il watershed, è essenziale ripulire la maschera binaria da elementi di rumore e da bordi non effettivi attraverso l'uso di trasformazioni morfologiche.

### Input

Il file di input è una maschera binaria con valori 0 e 1 dove:
- **0**: indica la presenza di un bordo
- **1**: indica l'assenza di un bordo

### Output

Il file di output è una maschera binaria con valori 0 e 255 dove:
- **0**: indica la presenza di un bordo
- **255**: indica l'assenza di un bordo

### Trasformazioni Morfologiche Applicate

Le seguenti trasformazioni morfologiche sono eseguite sull'immagine, con l'obiettivo di migliorare la qualità della maschera per successive elaborazioni:

1. **Opening**: Un'apertura morfologica (erosione seguita da una dilatazione) utilizzando un elemento strutturante di forma circolare con raggio di 2.5 pixel. Questo passaggio aiuta a rimuovere piccoli punti luminosi e a connettere piccole crepe scure.
2. **Small Object Removal**: Rimozione di oggetti piccoli che sono più piccoli di 200 pixel, con connettività di tipo 2. Questo passaggio è fondamentale per eliminare componenti isolati che possono essere interpretati come rumore.
3. **Small Object Removal su negativo**: Inversione della maschera per lavorare sui non-bordi come oggetti, seguita da una rimozione di piccoli oggetti dalla maschera invertita, impostando una soglia di dimensione minima di 80 pixel e connettività di tipo 2.

### Utilizzo

Per utilizzare questo script, specificare il percorso del file di input come argomento del comando e eseguire lo script. Il risultato sarà la generazione di un file GeoTIFF pulito nel percorso specificato.


## 3. **Segmentazione Watershed e Poligonizzazione**
   
## 3.1 Segmentazione Growing-Regions con Watershed iterativo

### 3.1.1 Descrizione

Lo script `iterativeWS_v2.py` implementa un approccio iterativo all'algoritmo di segmentazione watershed per l'identificazione di segmenti chiusi, ideale per la successiva poligonizzazione dei campi agricoli. Partendo dall'identificazione dei campi più grandi, passondo poi a quelli sempre più piccoli, l'aplicazione iterativa di questo metodo mira a minimizzare il rischio di oversegmentazione, migliorando la qualità e l'accuratezza della segmentazione in scenari complessi di immagini agricole.

Dopo la generazione di una maschera di bordi binaria tramite un modello UNet e successive pulizie morfologiche, l'algoritmo watershed viene applicato per identificare aree chiuse che rappresentano campi agricoli. Utilizzando un approccio iterativo, il processo inizia con la ricerca di massimi locali su distanze maggiori per identificare i campi più grandi, per poi diminuire progressivamente la distanza minima di ricerca, consentendo di identificare campi più piccoli senza frammentare eccessivamente quelli più grandi.

### 3.1.2 Input

- **File GeoTIFF**: Una maschera di bordi binaria pulita, ottenuta e preparata tramite processi precedenti. I bordi sono rappresentati con il valore 0, mentre le aree non bordo sono rappresentate con il valore 255.

### 3.1.3 Output

- **File GeoTIFF**: Una serie di file GeoTIFF per ciascun round di segmentazione, contenenti labels uniche per ciascun campo identificato. Ogni label corrisponde a un segmento unico individuato durante quel round specifico di segmentazione.

### 3.1.4 Watershed Iterativo

1. **distance_transform_edt**: Viene calcolata la trasformazione di distanza dall'immagine binaria, che serve per identificare i punti centrali dei potenziali campi, trovando i punti più distanti dai bordi e che rispettino una distanza reciproca minima.
2. **Ricerca Massimi Locali**: Utilizzando `peak_local_max` di `skimage.feature`, vengono identificati i massimi locali che servono come marcatori per il watershed.
3. **Analisi Componenti Connesse**: I picchi locali vengono analizzati per definire componenti connesse, che fungono da marcatori iniziali per il watershed.
4. **Segmentazione Watershed**: Utilizzando la maschera binaria come maschera e i marcatori identificati come inizializzatori, il watershed è applicato per segmentare l'immagine.
5. **Iterazione con Distanze Decrescenti**: Il processo viene ripetuto con distanza minima tra picchi picchi locali decrescenti per minimizzare la oversegmentazione.
 

## 3.2 Poligonizzazione dei Segmenti Watershed

Questo script converte i segmenti chiusi e unici, identificati attraverso l'algoritmo di segmentazione watershed, in vettori poligonali. L'obiettivo è facilitare analisi successive e operazioni di GIS, convertendo le maschere raster in formati vettoriali più utilizzabili per applicazioni di mappatura e monitoraggio agricolo.

### 3.2.1 Descrizione

Dopo aver completato la segmentazione watershed dei campi agricoli, il passo successivo è poligonizzare questi segmenti. Questo script poligonizza il raster dei segmenti unici, ovvero i labels unici generati dall'algoritmo watershed, in un formato vettoriale (Shapefile), che è più adatto per analisi successive.

Il processo utilizza la libreria GDAL per leggere un raster di input, che rappresenta la maschera di segmentazione watershed, e produce un file Shapefile che contiene poligoni corrispondenti a ciascun segmento unico identificato.

### 3.2.2 Dettagli Tecnici

- **Maschera di Input**: Un raster dove ogni valore unico rappresenta un segmento distinto.
- **File Shapefile**: Uno Shapefile che contiene i poligoni di ciascun segmento identificato. Ogni poligono ha un attributo 'Label' che corrisponde al label del segmento nel raster di input.





