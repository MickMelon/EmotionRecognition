import cv2 as cv
import numpy as np

fishface = cv.face.FisherFaceRecognizer_create()

def train(training_data, training_labels):
    fishface.train(training_data, np.array(training_labels))
    fishface.save("trained.yml")