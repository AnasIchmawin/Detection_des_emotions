<div style="text-align: center;">

# <span style="color:#fffff">DETECTION DES EMOTIONS</span>

### PYTHON PROJECT

</div><br>

| Realisé par: | Encadré par: |
|--------------|--------------|
| - IMANI Mourad | - ENNAJI Fatima Ezzohra |
| - ICHMAWIN Anas | |
| - BOUCHTA Othmane | |
| - FATIH Mohamed-Amine | |


## Introduction
La reconnaissance d'émotions à partir d'images constitue un domaine de recherche et d'application passionnant, alliant les disciplines de l'apprentissage profond, de la vision par ordinateur et de la psychologie. Ce projet s'inscrit dans ce contexte stimulant en visant à développer un modèle de réseau neuronal convolutionnel (CNN) capable de classifier les émotions humaines à partir d'images faciales en niveaux de gris

## EXPLORATION DES DONNÉES
![alt text](Picture4.png)

## PRÉTRAITEMENT DES DONNÉES
- <span style="font-size: larger;"><u>Première étape :</u> rendre les images sous forme carré </span>

![alt text](Picture5.png)

```python
def Rendre_carré(path_image):
    image = cv2.imread(path_image)
    height, width = image.shape[:2]
    crop_size = min(width, height)
    start_row = (height - crop_size) // 2
    start_col = (width - crop_size) // 2
    return image[start_row:start_row+crop_size, start_col:start_col+crop_size]
```

- <span style="font-size: larger;"><u>Deuxième étape:</u> 
redimensionnement
 </span>
 
![alt text](Picture6.png)

```python
def preprocess_image(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)
    return img_resized
```

- <span style="font-size: larger;"><u>Troixième étape:</u> cropping
 </span>
 
![alt text](Picture7.png)

```python
def Cropping(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return image.crop((10, 5, width-10, height-5))
```

## CRÉATION D’UN MODÈLE CNN
- <span style="font-size: larger;"><u>Définition :</u>
 </span><br>
    Un modèle CNN, ou Convolutional Neural Network (Réseau de Neurones Convolutif), est un type spécifique de réseau neuronal profond conçu principalement pour traiter et classer des données visuelles, telles que des images. Les CNN ont été largement utilisés et ont obtenu des succès significatifs dans des tâches telles que la reconnaissance d'objets, la détection de visages, la segmentation d'images, et bien d'autres.

## CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES:
- <span style="font-size: larger;"><u>les bibliothèques utilisées:</u>
 </span><br>

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib.pyplot as plt
import cv2
import os
```
- <span style="font-size: larger;"><u>Code source:</u>
 </span><br>

```python
data_path = "E:\\nouveau_pretraitement_1_1_2"
emotions_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
# Charger les images et les étiquettes
images = []
labels = []

for emotion_label in emotions_labels:
    folder_path = os.path.join(data_path, emotion_label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #convertir en niveaux de gris
        images.append(img)
        labels.append(emotions_labels.index(emotion_label))
```
