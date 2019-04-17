import cv2 as cv
import numpy as np

# Train the given FaceRecogniser instance with the given training data and labels.
#
# facerecogniser: The instance of the FaceRecogniser class.
# training_data: The array of training images.
# training_labels: The array of labels associated with the training images.
def train(facerecogniser, training_data, training_labels):
    # Train the FaceRecogniser.
    facerecogniser.train(training_data, np.array(training_labels))

    # Save the trained model.
    facerecogniser.save("trained.yml")