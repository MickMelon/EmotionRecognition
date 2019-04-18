import cv2 as cv
import sys

# The emotions defined for emotion recognition.
EMOTIONS = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

neutralw = 0
angerw = 0
sadnessw = 0
happyw = 0
fearw = 0
surprisew = 0
disgustw = 0

neutralt = 0
angert = 0
sadnesst = 0
happyt = 0
feart = 0
surpriset = 0
disgustt = 0

def countwrong(emotion):
    global neutralw, angerw, sadnessw, happyw, fearw, surprisew, disgustw

    if emotion == "neutral":
        neutralw += 1
    elif emotion == "anger":
        angerw += 1
    elif emotion == "sadness":
        sadnessw += 1
    elif emotion == "happy":
        happyw += 1
    elif emotion == "fear":
        fearw += 1
    elif emotion == "surprise":
        surprisew += 1
    elif emotion == "disgust":
        disgustw += 1

def totalforeachemotion(prediction_labels):
    global neutralt, angert, sadnesst, happyt, feart, surpriset, disgustt

    for label in prediction_labels:
        if EMOTIONS[label] == "neutral":
            neutralt += 1
        elif EMOTIONS[label] == "anger":
            angert += 1
        elif EMOTIONS[label] == "sadness":
            sadnesst += 1
        elif EMOTIONS[label] == "happy":
            happyt += 1
        elif EMOTIONS[label] == "fear":
            feart += 1
        elif EMOTIONS[label] == "surprise":
            surpriset += 1
        elif EMOTIONS[label] == "disgust":
            disgustt += 1

def printresults():
    #global neutralt, angert, sadnesst, happyt, feart, surpriset, disgustt
    #global neutralw, angerw, sadnessw, happyw, fearw, surprisew, disgustw

    neutralp = round((neutralw / neutralt) * 100)
    angerp = round((angerw / angert) * 100)
    sadnessp = round((sadnessw / sadnesst) * 100)
    happyp = round((happyw / happyt) * 100)
    fearp = round((fearw / feart) * 100)
    surprisep = round((surprisew / surpriset) * 100)
    disgustp = round((disgustw / disgustt) * 100)
    
    print("Neutral: %s of %s incorrect (%s%%)" %(neutralw, neutralt, neutralp))
    print("Anger: %s of %s incorrect (%s%%)" %(angerw, angert, angerp))
    print("Sadness: %s of %s incorrect (%s%%)" %(sadnessw, sadnesst, sadnessp))
    print("Happy: %s of %s incorrect (%s%%)" %(happyw, happyt, happyp))
    print("Fear: %s of %s incorrect (%s%%)" %(fearw, feart, fearp))
    print("Surprise: %s of %s incorrect (%s%%)" %(surprisew, surpriset, surprisep))
    print("Disgust: %s of %s incorrect (%s%%)" %(disgustw, disgustt, disgustp))

# Predicts a set of data with the given data and labels.
#
# facerecogniser: The instance of FaceRecogniser used.
# prediction_data: The array of face images to be used for prediction.
# prediction_labels: The array of labels associated with each face image.
def predict_set(facerecogniser, prediction_data, prediction_labels):
    # Read in the already trained model to be used for prediction.
    #facerecogniser.read("trained.yml")

    count = 0
    correct = 0
    incorrect = 0

   # sys.exit()

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
            countwrong(EMOTIONS[pred])

            # flag image
            cv.imwrite("incorrect/%sGuessed_%sActual%s.jpg" %(EMOTIONS[pred], EMOTIONS[prediction_labels[count]], count), image)

            incorrect += 1
        count += 1

    totalforeachemotion(prediction_labels)
    printresults()

    print("%s incorrect in total" %incorrect)

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

