import cv2 as cv

# The emotions defined for emotion recognition.
EMOTIONS = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

# Predicts a set of data with the given data and labels.
#
# facerecogniser: The instance of FaceRecogniser used.
# prediction_data: The array of face images to be used for prediction.
# prediction_labels: The array of labels associated with each face image.
def predict_set(facerecogniser, prediction_data, prediction_labels):
    # Read in the already trained model to be used for prediction.
    facerecogniser.read("trained.yml")

    count = 0
    correct = 0
    incorrect = 0

    # Begin predicting each image in the array.
    for image in prediction_data:
        print("** Predicting No %i **" %count)

        # Make prediction.
        pred, conf = facerecogniser.predict(image)
        print("PRED: %s | CONF: %s" %(pred, conf))

        # Check if the prediction was correct.
        if pred == prediction_labels[count]:
            print("Correct with confidence %s [%s]" %(conf, EMOTIONS[pred]))
            correct += 1
        else:
            print("Incorrect. Guessed: %s / Actual: %s" %(EMOTIONS[pred], EMOTIONS[prediction_labels[count]]))
            incorrect += 1
        count += 1

    # Return the accuracy as a percentage.
    return (100 * correct) / (correct + incorrect)

# Predicts a one image with the given image and label.
#
# facerecogniser: The instance of FaceRecogniser used.
# image: The face image to be used for prediction
# label: The label associated with the face image.
def predict_one(facerecogniser, image, label):
    # Read in the already trained model to be used for prediction.
    facerecogniser.read("trained.yml")

    # Make the prediction.
    pred, conf = facerecogniser.predict(image)
    print("Guessed %s" %EMOTIONS[pred])

    # Check if the prediction was correct.
    if pred == label:
        print("Got it correct")
        return True

    # The prediction wasn't correct if the code gets here.
    return False

