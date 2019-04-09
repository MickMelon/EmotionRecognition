import cv2 as cv

emotions = ["neutral", "angry", "sad", "happy", "disgust", "fear", "surprise"]
fishface = cv.face.FisherFaceRecognizer_create()

def predict_set(prediction_data, prediction_labels):
    fishface.read("trained.yml")

    count = 0
    correct = 0
    incorrect = 0

    for image in prediction_data:
        print("** Predicting No %i **" %count)
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[count]:
            print("Correct with confidence %s" %conf)
            correct += 1
        else:
            print("Incorrect")
            incorrect += 1
        count += 1

    return (100 * correct) / (correct + incorrect)

def predict_one(image, label):
    fishface.read("test.yml")
    pred, conf = fishface.predict(image)
    if pred == label:
        print("Got it correct")
    return pred, conf

