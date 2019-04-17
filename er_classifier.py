import cv2 as cv

emotions = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

fishface = cv.face.EigenFaceRecognizer_create()

def predict_set(prediction_data, prediction_labels):
    fishface.read("trained.yml")

    count = 0
    correct = 0
    incorrect = 0

    for image in prediction_data:
        print("** Predicting No %i **" %count)
        pred, conf = fishface.predict(image)
        print("PRED: %s | CONF: %s" %(pred, conf))
        if pred == prediction_labels[count]:
            print("Correct with confidence %s [%s]" %(conf, emotions[pred]))
            correct += 1
        else:
            print("Incorrect. Guessed: %s / Actual: %s" %(emotions[pred], emotions[prediction_labels[count]]))
            incorrect += 1
        count += 1

    return (100 * correct) / (correct + incorrect)

def predict_one(image, label):
    fishface.read("trained.yml")
    pred, conf = fishface.predict(image)
    print("Guessed %s" %emotions[pred])
    if pred == label:
        print("Got it correct")
    return pred, conf

