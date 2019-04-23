import cv2 as cv
import sys

# The emotions defined for emotion recognition.
EMOTIONS = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

# The count of all the wrong guesses
neutral_wrong = 0
anger_wrong = 0
sadness_wrong = 0
happy_wrong = 0
fear_wrong = 0
surprise_wrong = 0
disgust_wrong = 0

# The total of each emotion in the prediction set
neutral_total = 0
anger_total = 0
sadness_total = 0
happy_total = 0
fear_total = 0
surprise_total = 0
disgust_total = 0

# Add plus one to the wrong count for the given emotion
def countwrong(emotion):
    global neutral_wrong, anger_wrong, sadness_wrong, happy_wrong, fear_wrong, surprise_wrong, disgust_wrong

    if emotion == "neutral":
        neutral_wrong += 1
    elif emotion == "anger":
        anger_wrong += 1
    elif emotion == "sadness":
        sadness_wrong += 1
    elif emotion == "happy":
        happy_wrong += 1
    elif emotion == "fear":
        fear_wrong += 1
    elif emotion == "surprise":
        surprise_wrong += 1
    elif emotion == "disgust":
        disgust_wrong += 1

# Get the total amount of faces for each emotion in the prediction set.
def totalforeachemotion(prediction_labels):
    global neutral_total, anger_total, sadness_total, happy_total, fear_total, surprise_total, disgust_total

    for label in prediction_labels:
        if EMOTIONS[label] == "neutral":
            neutral_total += 1
        elif EMOTIONS[label] == "anger":
            anger_total += 1
        elif EMOTIONS[label] == "sadness":
            sadness_total += 1
        elif EMOTIONS[label] == "happy":
            happy_total += 1
        elif EMOTIONS[label] == "fear":
            fear_total += 1
        elif EMOTIONS[label] == "surprise":
            surprise_total += 1
        elif EMOTIONS[label] == "disgust":
            disgust_total += 1

# Print the results from the prediction.
def printresults():
    neutralp = round((neutral_wrong / neutral_total) * 100)
    angerp = round((anger_wrong / anger_total) * 100)
    sadnessp = round((sadness_wrong / sadness_total) * 100)
    happyp = round((happy_wrong / happy_total) * 100)
    fearp = round((fear_wrong / fear_total) * 100)
    surprisep = round((surprise_wrong / surprise_total) * 100)
    disgustp = round((disgust_wrong / disgust_total) * 100)
    
    print("Neutral: %s of %s incorrect (%s%%)" %(neutral_wrong, neutral_total, neutralp))
    print("Anger: %s of %s incorrect (%s%%)" %(anger_wrong, anger_total, angerp))
    print("Sadness: %s of %s incorrect (%s%%)" %(sadness_wrong, sadness_total, sadnessp))
    print("Happy: %s of %s incorrect (%s%%)" %(happy_wrong, happy_total, happyp))
    print("Fear: %s of %s incorrect (%s%%)" %(fear_wrong, fear_total, fearp))
    print("Surprise: %s of %s incorrect (%s%%)" %(surprise_wrong, surprise_total, surprisep))
    print("Disgust: %s of %s incorrect (%s%%)" %(disgust_wrong, disgust_total, disgustp))

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

    # Ensure all the wrong variables are set to 0
    neutral_wrong = 0
    anger_wrong = 0
    sadness_wrong = 0
    happy_wrong = 0
    fear_wrong = 0
    surprise_wrong = 0
    disgust_wrong = 0

    # Ensure all the total variables are set to 0
    neutral_total = 0
    anger_total = 0
    sadness_total = 0
    happy_total = 0
    fear_total = 0
    surprise_total = 0
    disgust_total = 0

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
    if EMOTIONS[pred] == label:
        return True, EMOTIONS[pred]

    # The prediction wasn't correct if the code gets here.
    return False, EMOTIONS[pred]

