import cv2 as cv
import er_processor as processor
import er_trainer as trainer
import er_classifier as classifier
import glob
import random

emotions = ["neutral", "angry", "sad", "happy", "fear", "surprise", "disgust"]


#pp.preprocessing()
#prediction_data, prediction_labels = l.run()
#result = c.predict(prediction_data, prediction_labels)
#print("Completed! Result was %i" %result)

#image = pp.one("009_su_003_0023.jpg")
#pred, conf = c.predict_one(image)
#print("Predicted %s with %s confidence" %(emotions[pred], conf))

# get face from image and crop and gray it
# do this for all 

# get all the images
#   for every image
#   preprocesser.process
#       detect
#       processimage
# 
# split images into training and prediction
# train classifier
# classify

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

def get_files(emotion):
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction

# First process the original face set
#processor.process("mug_dataset")

# Make the sets used for this
training_data, training_labels, prediction_data, prediction_labels = make_sets()

#trainer.train(training_data, training_labels)
result = classifier.predict_set(prediction_data, prediction_labels)
print("Result was %s" %result)