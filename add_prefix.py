# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:36:57 2024

@author: Alvise Ferrari

Al fine di poter costituire un dataset con input multipli, a cui però corrisponde sempre lo stesso output 
(ricordiamo che la maschera di Canny è costituita sovrapponendo applicando l'omonimo filtro a immagini multiple),
si è definita la seguente semplice funzione che aggiunge un determinato prefisso ai nomi di tutti i file contenuti 
all'interno di una cartella specificata. In questa fase sperimentale sarà necessario aggiungere tale prefisso
a tutte le cartelle contenenti le subtile di input; inoltre sarà necessario moltiplicare la cartella contenente 
le subtile di output, per un numero di volte pari al numero delle immagini di input, ed applicare fittizziamente
il prefisso corrispondente a ciascun input, a ognuna delle cartelle di output. Infine tutte le coppie i/o possono 
essere organizzate in una unica directory che verrà utilizzata per training o fine tuning.

"""

import os

def add_prefix_to_files(directory, prefix):
    # Change the current working directory to the specified directory
    os.chdir(directory)
    
    # List all files in the directory
    files = os.listdir()
    
    # Iterate over each file
    for filename in files:
        # Construct new filename with the prefix
        new_filename = prefix + filename
        
        # Rename the file
        os.rename(filename, new_filename)
        print(f"Renamed '{filename}' to '{new_filename}'")

# Example usage
directory_path = r"D:\Lavoro_e_Studio\Assegno_Ricerca_Sapienza\UNET_fields_segentation\Nuovo_addestramento_igarss2024\Iowa_15TWG\2021\IOWA_15TWG_canny_2021_NDVIth025_NDWIth025_sigma1dot5_optimized_thresh_tiles_woverlap_20211018"
prefix = '20211018_'  # Replace with your desired prefix
add_prefix_to_files(directory_path, prefix)
