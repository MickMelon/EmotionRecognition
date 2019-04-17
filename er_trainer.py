import cv2 as cv
import numpy as np

def train(facerecogniser, training_data, training_labels):
    facerecogniser.train(training_data, np.array(training_labels))
    facerecogniser.save("trained.yml")