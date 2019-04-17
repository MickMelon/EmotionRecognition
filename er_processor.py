import cv2 as cv
import glob
import numpy as np
import math
import crop_face
from PIL import Image
import sys

# The emotions defined for emotion recognition.
EMOTIONS = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

# The OpenCV HAAR Cascade Classifiers used for face and eye detection.
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_casecade = cv.CascadeClassifier('haarcascade_eye.xml')

# Constant variables to specify desired values.
DESIRED_FACE_WIDTH = 150
DESIRED_FACE_HEIGHT = 150
DESIRED_LEFT_EYE_X = 0.22
DESIRED_LEFT_EYE_Y = 0.2
DESIRED_RIGHT_EYE_X = 1.0 - 0.22
DESIRED_RIGHT_EYE_Y = 1.0 - 0.2

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
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Detect a face in the image
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)

    # Check if a face has been found
    if len(face) == 1:
        # Specify the variables for better readability
        x, y, w, h = face[0][0], face[0][1], face[0][2], face[0][3]

        # Crop the face from the original image and resize it to the desired width and height.
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv.resize(roi_gray, (DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT))

        # Detect eyes in the cropped face image.
        eyes = eye_casecade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(25, 25), maxSize=(30, 30))
        
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
            roi_gray = histogram_equalisation(roi_gray)
            roi_gray = smoothing(roi_gray)
            roi_gray = elliptical_mask(roi_gray)

            # Return the fully pre-processed face image.
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
    rotationMatrix = cv.getRotationMatrix2D((eyesCenterX, eyesCenterY), angle, 1.5)

    # Move the center of the eyes so that they are at the desired center.
    ex = DESIRED_FACE_WIDTH * 0.5 - eyesCenterX
    ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenterY
    rotationMatrix[0,2] += ex
    rotationMatrix[1,2] += ey

    # Transform the face to the desired angle, size, and position. 
    return cv.warpAffine(gray, rotationMatrix, (DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT))

# Runs the processor.
def run_processor(sourcefolder, targetfolder):
    for emotion in EMOTIONS:
        print("Processing emotion %s" %emotion)
        images = load_images(sourcefolder, emotion)
        if len(images) == 0:
            print("No files could be found for %s" %emotion)
        fileNumber = 0
        for file in images:
            print("Processing file %s" %file)
            result = process(file)
            if result == "": 
                print("File %s could not be processed." %file)
                continue
            cv.imwrite("%s/%s/%s.jpg" %(targetfolder, emotion, fileNumber), result)
            print("File %s processed successfully" %file)
            fileNumber += 1