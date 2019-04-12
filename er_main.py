import cv2 as cv
import er_processor as processor
import er_trainer as trainer
import er_classifier as classifier
import glob
import random
import sys
import os

emotions = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]


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

def make_sets(sourcefolder):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:
        training, prediction = get_files(sourcefolder, emotion)

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

def get_files(sourcefolder, emotion):
    files = glob.glob("%s/%s/*" %(sourcefolder, emotion))
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction

# First process the original face set
#processor.process("mug_dataset")

# Make the sets used for this
training_data, training_labels, prediction_data, prediction_labels = make_sets("dataset")

trainer.train(training_data, training_labels)
result = classifier.predict_set(prediction_data, prediction_labels)
print("Accuracy: %s" %result)
#processor.process("mug_dataset", "dataset")
#image = cv.imread("dataset/neutral/3.jpg")
#gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#result = classifier.predict_one(gray, "neutral")

######

#cv.EYE_SX
#
#def emotionsFoldersExist(folder):
#    for emotion in emotions:
#        if not os.path.isdir(folder + "/" + emotion):
#            print("ERROR: Could not find %s folder in %s" %(emotion, folder))
#            return False
#    return True#
#
#if len(sys.argv) == 1:
#    print("USAGE: python er_main.py [action]")
#elif sys.argv[1] == "process":
#    if len(sys.argv) == 4:
#        source = sys.argv[2]
#        target = sys.argv[3]
#        if os.path.isdir(source) and os.path.isdir(target):            
#            if emotionsFoldersExist(source) and emotionsFoldersExist(target):
#                print("Beginning to process...")
#                processor.process(source, target) 
#                print("Processing finished!")
#        else:
#            print("ERROR: Source or target folder does not exist")
#    else:
#        print("USAGE: python er_main.py process [sourcefolder] [targetfolder]")
#elif sys.argv[1] == "train":
#    if len(sys.argv) == 3:
#        source = sys.argv[3]
#        if os.path.isdir(source):
#            training_data, training_labels, prediction_data, prediction_labels = make_sets(source)
#    #train
#elif sys.argv[1] == "predict":
#    print("Predict")
#    classifier.predict_set(data, labels)
#    #predict
##
#else:
#   print("USAGE: python er_main.py [action]")