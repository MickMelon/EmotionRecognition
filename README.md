# Emotion Recognition Application
This emotion recognition application was developed as part of the CMP304 Artificial Intelligence 
module undertaken at Abertay University.

The emotions that this application recognises are:
1. Anger
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sadness
7. Surprise

# Running
Run the file `er_main.py` and you will be presented with a menu indicating the options that the
application provides. 

These options are:
1. Process dataset
2. Train from full dataset
3. Train and predict from 80/20 dataset
4. Predict one

## Process dataset
This will put a folder through the pre-processing steps and output into the specified folder. Both
folders must have the emotion folders in them already.

## Train from full dataset
This will train a FaceRecognizer model with all the images in the specified folder.

## Train and predict from 80/20 dataset
This will randomly take 80% of each emotion and use it for training and 20% of each emotion and 
use it for prediction.

## Predict one
This will predict only one image using the already trained model.