import cv2 as cv
import er_processor as processor
import er_trainer as trainer
import er_classifier as classifier
import glob
import random
import sys
import os
import time

# The emotions used for emotion recognition.
EMOTIONS = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

# Makes training and prediction sets along with labels for all the emotion 
# pictures in the specified sourcefolder. 
def make_sets(sourcefolder):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in EMOTIONS:
        training, prediction = get_files(sourcefolder, emotion)

        # For every training image, convert it to grayscale and add it to the training data
        # array along with an entry to the training labels array indicating the emotion.
        for item in training:
            image = cv.imread(item)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(EMOTIONS.index(emotion))

        # For every prediction image, convert it to grayscale and add it to the prediction data
        # array along with an entry to the prediction labels array indicating the emotion.
        for item in prediction:
            image = cv.imread(item)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(EMOTIONS.index(emotion))
        
    return training_data, training_labels, prediction_data, prediction_labels

# Gets all the files for a given emotion in the given folder and splits them into training and 
# prediction sets, 80% and 20% respectively.
def get_files(sourcefolder, emotion):
    files = glob.glob("%s/%s/*" %(sourcefolder, emotion))
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction

# Testing
def test():
    facerecogniser = cv.face.LBPHFaceRecognizer_create()
    average = 0
    loops = 10

    for i in range(loops):   
        start = time.time()
        training_data, training_labels, prediction_data, prediction_labels = make_sets("dataset")
        trainer.train(facerecogniser, training_data, training_labels)
        end = time.time()

        totaltraintime = end - start

        start = time.time()
        result = classifier.predict_set(facerecogniser, prediction_data, prediction_labels)
        end = time.time()

        totalpredicttime = end - start
        print("Training took %s on %s files and Prediction took %s on %s files " %(totaltraintime, len(training_data), totalpredicttime, len(prediction_data)))
        average += result
        print("Prediction Accuracy: %s%%" %round(result))#

    average /= loops
    average = round(average)
    print("Completed. Average accuracy %s%% " %average)

def process():
    start = time.time()
    processor.run_processor("mug_dataset", "dataset")
    end = time.time()
    totaltime = end - start
    print("Processing took %s" %totaltime)

process()
test()
