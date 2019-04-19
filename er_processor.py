import cv2 as cv
import glob
import numpy as np
import math
import crop_face
from PIL import Image
import sys
import time

# The emotions defined for emotion recognition.
EMOTIONS = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

# The OpenCV HAAR Cascade Classifiers used for face and eye detection.
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_casecade = cv.CascadeClassifier('haarcascade_eye.xml')

# Constant variables to specify desired values.
DESIRED_FACE_WIDTH = 150
DESIRED_FACE_HEIGHT = 150
DESIRED_LEFT_EYE_X = 0.20
DESIRED_LEFT_EYE_Y = 0.26
DESIRED_RIGHT_EYE_X = 1.0 - 0.20
DESIRED_RIGHT_EYE_Y = 1.0 - 0.26

# Shows the specified image in a window.
def show_image(image):
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Loads all the images in a given folder for the given emotion.
def load_images(folder, emotion):
    return glob.glob("%s//%s//*" %(folder, emotion))

# Carries out all the pre-processing steps on the given file.
def process(file):
    # Read the image and convert it to grayscale
    image = cv.imread(file)
    cv.imwrite("test/1_original.jpg", image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite("test/2_gray.jpg", gray)
    # Detect a face in the image
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)

    # Check if a face has been found
    if len(face) == 1:
        # Specify the variables for better readability
        x, y, w, h = face[0][0], face[0][1], face[0][2], face[0][3]

        # Crop the face from the original image and resize it to the desired width and height.
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv.resize(roi_gray, (DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT))
        cv.imwrite("test/3_croppedface.jpg", roi_gray)

        # Detect eyes in the cropped face image.
        eyes = eye_casecade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=10, minSize=(20, 20), maxSize=(35, 35))

        #eyeNo = 0
        #lol = 255
        #for (ex,ey,ew,eh) in eyes:
        #    roi_eye = gray[y:ey+eh, x:ex+ew]
        #    print("eye %s ex %s ey %s lol %s" %(eyeNo, ex, ey, lol))
        #    cv.rectangle(roi_gray, (ex, ey), (ex+ew, ey+eh), (lol,lol,lol), 2)
        #    eyeNo += 1
        #    lol = 0

        #show_image(roi_gray)
        #sys.exit()
        
        # Check if two eyes were found
        if len(eyes) == 2:
            # Get left and right eyes. The eye with the highest X value is the right eye.
            if eyes[0][0] > eyes[1][0]:
                rightEye = eyes[0]
                leftEye = eyes[1]
            else:
                rightEye = eyes[1]
                leftEye = eyes[0]

            # Get the center position of the eyes because the eye classifier creates a rectangle
            # around the eyes, so we need to get the center of that rectangle.
            rightEyeX = int(round(rightEye[0] + (rightEye[2] / 2)))
            rightEyeY = int(round(rightEye[1] + (rightEye[3] / 2)))
            leftEyeX = int(round(leftEye[0] + (leftEye[2] / 2)))
            leftEyeY = int(round(leftEye[1] + (leftEye[3] / 2)))

            # Perform pre-processing steps on the image.
            roi_gray = geometrical_transformation(roi_gray, rightEyeX, rightEyeY, leftEyeX, leftEyeY)
            cv.imwrite("test/4_geo.jpg", roi_gray)
            roi_gray = histogram_equalisation(roi_gray)
            cv.imwrite("test/5_histo.jpg", roi_gray)
            roi_gray = smoothing(roi_gray)
            cv.imwrite("test/6_smooth.jpg", roi_gray)
            roi_gray = elliptical_mask(roi_gray)
            cv.imwrite("test/7_mask.jpg", roi_gray)

            #sys.exit()

            # Return the fully pre-processed face image.
            roi_gray = cv.resize(roi_gray, (70, 70))
            cv.imwrite("test/8_finalresize.jpg", roi_gray)
            #sys.exit()
            return roi_gray

    # No face or eyes were detected, so return an empty string to indicate this.
    return ""

# Apply an elliptical mask on the image to remove unwanted corners of the face image. This is done 
# by drawing an ellipse and applying it to the face image.
def elliptical_mask(image):
    # Create a blank image with the same size as the input image.
    mask = np.zeros_like(image)

    # Calculate the size of the ellipse.
    sizeX = round(DESIRED_FACE_WIDTH * 0.5)
    sizeY = round(DESIRED_FACE_HEIGHT * 0.8)

    # Calculate the center of the image.
    centerX = int(round(DESIRED_FACE_WIDTH / 2))
    centerY = int(round(DESIRED_FACE_HEIGHT / 2))

    # Draw the ellipse onto the blank image.
    mask = cv.ellipse(mask, center=(centerX, centerY), axes=(sizeX,sizeY), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)

    # Perform a bitwise AND operation to apply the mask to the image.
    np.bitwise_and(image, mask)

    # Change the colour of the corner regions from black to grey so there is less contrast to
    # the rest of the face. 
    image[mask == 0] = 128

    return image

# Perform a bilateral filter on the input image to reduce the effect of pixel noise. A filter 
# strength of 20 was chosen to cover heavy pixel noise caused by histogram equalisation.
def smoothing(image):
    return cv.bilateralFilter(image, 0, 20, 20)

def gaussian_blur(image):
    return cv.GaussianBlur(image, (5,5), 0)

# Perform histogram equalisation on the input image to standardise the brightness and contrast.
def histogram_equalisation(image):
    return cv.equalizeHist(image)

# Perform geometrical transformation on the input image in order to normalise it so that face images
# have reduced variability. 
# This involves:
# 1. Rotating the face so that the two eyes are horizontal.
# 2. Scale the face so that the distance between the two eyes is always the same.
# 3. Translate the face so that the two eyes are always centered horizontally and are of
#    the desired height.
# 4. Crop the unneccessary outer parts of the image (background, hair, ears, and chin)
def geometrical_transformation(gray, rightEyeX, rightEyeY, leftEyeX, leftEyeY):
    # Get the center between the two eyes
    eyesCenterX = (leftEyeX + rightEyeX) * 0.5
    eyesCenterY = (leftEyeY + rightEyeY) * 0.5

    # Get the angle between the two eyes
    dy = rightEyeY - leftEyeY
    dx = rightEyeX - leftEyeX
    length = math.sqrt(dx*dx + dy*dy)
    angle = math.atan2(dy, dx) * 180.0/np.pi

    # Get the amount that the image needs to be scaled by in order to be the desired
    # fixed face height and width.
    desiredLength = (DESIRED_RIGHT_EYE_X - 0.16)
    scale = desiredLength * DESIRED_FACE_WIDTH / length

    # Get transformation matrix for desired angle and size
    rotationMatrix = cv.getRotationMatrix2D((eyesCenterX, eyesCenterY), angle, 1.5) #1.55

    # Move the center of the eyes so that they are at the desired center.
    ex = DESIRED_FACE_WIDTH * 0.5 - eyesCenterX
    ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenterY
    rotationMatrix[0,2] += ex
    rotationMatrix[1,2] += ey

    # Transform the face to the desired angle, size, and position. 
    return cv.warpAffine(gray, rotationMatrix, (DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT))

def process_one(file):
    print("here")
    result = process(file)
    if result == "":
        print("File %s could not be processed." %file)
        return ""
    
    cv.imwrite("test/processed.png", result)
    print("File %s processed successfully" %file)

# Runs the processor.
def run_processor(sourcefolder, targetfolder):
    totaltime = 0
    totalfiles = 0

    errorcount = 0

    for emotion in EMOTIONS:
        print("Processing emotion %s" %emotion)
        images = load_images(sourcefolder, emotion)
        if len(images) == 0:
            print("No files could be found for %s" %emotion)
        fileNumber = 0
        
        for file in images:
            start = time.time()

            print("Processing file %s" %file)
            result = process(file)
            if result == "": 
                print("File %s could not be processed." %file)
                errorcount += 1
                continue
            cv.imwrite("%s/%s/%s.png" %(targetfolder, emotion, fileNumber), result)
            print("File %s processed successfully" %file)
            fileNumber += 1
            totalfiles += 1

            end = time.time()
            timetaken = end - start
            totaltime += timetaken

    average = totaltime / totalfiles
    print("%s processed, Average time per image processed %s" %(totalfiles, average))
    print("Failed to process %s images" %errorcount)