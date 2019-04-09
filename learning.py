import cv2 as cv
import glob
import random
import numpy as np

emotions = ["neutral", "angry", "sad", "happy", "disgust", "fear", "surprise"]
fishface = cv.face.FisherFaceRecognizer_create()
data = {}

def get_files(emotion):
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    
    # Split the files 80/20 so that 80% is used for training and 20% used for learning
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:
        training, prediction = get_files(emotion)

        for item in training:
            image = cv.imread(item)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv.imread(item)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def train(training_data, training_labels):
    fishface.train(training_data, np.array(training_labels))
    fishface.save("test.yml")
    
def run():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    train(training_data, training_labels)