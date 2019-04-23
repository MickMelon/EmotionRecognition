import cv2 as cv
import er_processor as processor
import er_trainer as trainer
import er_classifier as classifier
import glob
import random
import sys
import os
import time

# The emotions used for emotion recognition.
EMOTIONS = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

# The instance of the FaceRecognizer class used for training and prediction.
facerecogniser = cv.face.LBPHFaceRecognizer_create()

# Makes training and prediction sets along with labels for all the emotion 
# pictures in the specified sourcefolder. 
def make_sets(sourcefolder):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in EMOTIONS:
        training, prediction = get_files(sourcefolder, emotion)

        # For every training image, convert it to grayscale and add it to the training data
        # array along with an entry to the training labels array indicating the emotion.
        for item in training:
            image = cv.imread(item)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            training_data.append(gray)
            training_labels.append(EMOTIONS.index(emotion))

        # For every prediction image, convert it to grayscale and add it to the prediction data
        # array along with an entry to the prediction labels array indicating the emotion.
        for item in prediction:
            image = cv.imread(item)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            
            prediction_data.append(gray)
            prediction_labels.append(EMOTIONS.index(emotion))
        
    return training_data, training_labels, prediction_data, prediction_labels

# Gets all the files for a given emotion in the given folder and splits them into training and 
# prediction sets, 80% and 20% respectively.
def get_files(sourcefolder, emotion):
    files = glob.glob("%s/%s/*" %(sourcefolder, emotion))
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction

# Makes a training set from all the images in the source folder. In contrast to make_sets() that
# makes sets for 80/20 training and prediction.
def make_full_training_set(sourcefolder):
    data = []
    labels = []

    for emotion in EMOTIONS:
        files = glob.glob("%s/%s/*" %(sourcefolder, emotion))
        for file in files:
            image = cv.imread(file)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            data.append(gray)
            labels.append(EMOTIONS.index(emotion))

    return data, labels

# Main Menu
choice = ""
while choice != "exit":
    print("# Emotion Recognition #")
    print("1. Process dataset")
    print("2. Train from full dataset")
    print("3. Train and predict from 80/20 dataset")
    print("4. Predict one")
    print("Choose menu item >>")
    choice = input()

    if choice == "1": # Process dataset
        print("Enter source folder >>")
        source = input()
        print("Enter target folder >>")
        target = input()

        start = time.time()
        processor.run_processor(source, target)
        end = time.time()
        totaltime = end - start
        print("Processing took %s" %totaltime)

    elif choice == "2": # Train from full dataset
        print("Enter dataset location >>")
        location = input()

        data, labels = make_full_training_set(location)
        trainer.train(facerecogniser, data, labels)
        print("Training for the full dataset was completed.")    

    elif choice == "3": # Train and predict from 80/20 dataset
        print("Enter dataset folder (must already be processed) >>")
        dataset = input()
        print("Enter amount of loops >>")
        loops = int(input())
        if loops < 1: 
            loops = 1

        average = 0
        
        # Loop for as many times as specified. For each loop, make sets from the specified dataset
        # folder, train, then predict, all while timing how long it takes.
        for i in range(loops):   
            # Training
            start = time.time()
            training_data, training_labels, prediction_data, prediction_labels = make_sets(dataset)
            trainer.train(facerecogniser, training_data, training_labels)
            end = time.time()
            totaltraintime = end - start

            # Prediction
            start = time.time()
            result = classifier.predict_set(facerecogniser, prediction_data, prediction_labels)
            end = time.time()

            # Display the results from the loop
            totalpredicttime = end - start
            print("Training took %s on %s files and Prediction took %s on %s files " %(totaltraintime, len(training_data), totalpredicttime, len(prediction_data)))
            average += result
            print("Prediction Accuracy: %s%%" %round(result))#

        # Display the overall results
        average /= loops
        average = round(average)
        print("Completed. Average accuracy %s%% " %average)

    elif choice == "4": # predict one
        print("Enter file location >>")
        location = input()
        print("Enter emotion >>")
        emotion = input()

        # Process the input image
        processor.process_one(location)
        processed = cv.imread("test/processed.png")
        processed = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
        print("Finish process")

        # Predict the input image and print the result
        result, pred = classifier.predict_one(facerecogniser, processed, emotion)
        if result:
            print("Correct")
        else:
            print("Incorrect, guessed %s" %pred)