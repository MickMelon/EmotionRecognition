import cv2 as cv
import glob
import numpy as np
import math
import crop_face
from PIL import Image
import sys

emotions = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_casecade = cv.CascadeClassifier('haarcascade_eye.xml')

DESIRED_FACE_WIDTH = 150
DESIRED_FACE_HEIGHT = 150
DESIRED_LEFT_EYE_X = 0.22
DESIRED_LEFT_EYE_Y = 0.2
DESIRED_RIGHT_EYE_X = 1.0 - 0.22
DESIRED_RIGHT_EYE_Y = 1.0 - 0.2

def show_image(image):
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def load_images(folder, emotion):
    return glob.glob("%s//%s//*" %(folder, emotion))

def crop(imageLocation, leftEyeX, leftEyeY, rightEyeX, rightEyeY):
    image = Image.open(imageLocation)
    filename = imageLocation[:-4] + "_cropped.jpg"
    crop_face.CropFace(image, eye_left=(rightEyeX, rightEyeY), eye_right=(leftEyeX, leftEyeY), offset_pct=(0.2,0.2), dest_sz=(100,100)).save(filename)
    return filename

def detect_face(file):
    print("Beginning to detect a face...")
    frame = cv.imread(file)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)

    if len(face) == 1:
        x, y, w, h = face[0][0], face[0][1], face[0][2], face[0][3]
        roi_gray = gray[y:y+h, x:x+w]
        #roi_color = frame[y:y+h, x:x+w]
        roi_gray = cv.resize(roi_gray, (DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT))
        eyes = eye_casecade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(25, 25), maxSize=(30, 30))
        
        if len(eyes) != 2:
            return ""

        #cv.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        #eyeNo = 0
        #lol = 255
        #eye0=right
        #for (ex,ey,ew,eh) in eyes:
        #    roi_eye = gray[y:ey+eh, x:ex+ew]
        #    print("eye %s ex %s ey %s lol %s" %(eyeNo, ex, ey, lol))
        #    cv.rectangle(roi_gray, (ex, ey), (ex+ew, ey+eh), (lol,lol,lol), 2)
        #    eyeNo += 1
        #    lol = 0

        # Get left and right eyes
        if eyes[0][0] > eyes[1][0]:
            rightEye = eyes[0]
            leftEye = eyes[1]
        else:
            rightEye = eyes[1]
            leftEye = eyes[0]

        rightEyeX = int(round(rightEye[0] + (rightEye[2] / 2)))
        rightEyeY = int(round(rightEye[1] + (rightEye[3] / 2)))

        leftEyeX = int(round(leftEye[0] + (leftEye[2] / 2)))
        leftEyeY = int(round(leftEye[1] + (leftEye[3] / 2)))

        print("%s %s %s %s" %(rightEyeX, rightEyeY, leftEyeX, leftEyeY))

        #cv.circle(roi_gray, (leftEyeX, leftEyeY), 1, (255,255,0), -1)
        #cv.circle(roi_gray, (rightEyeX, rightEyeY), 1, (255,255,0), -1)

        #cv.line(roi_gray, (rightEyeX, rightEyeY), (leftEyeX, leftEyeY), (255,0,0),1)

        roi_gray = geometrical_transformation(roi_gray, rightEyeX, rightEyeY, leftEyeX, leftEyeY)

        

        
        roi_gray = histogram_equalisation(roi_gray)
        roi_gray = smoothing(roi_gray)
        #roi_gray = cv.GaussianBlur(roi_gray, (5,5), 0)

        roi_gray = e_mask(roi_gray)
        #roi_gray = elliptical_mask(roi_gray)
        
        #show_image(roi_gray)
        #sys.exit()

        return roi_gray

    print("No face detected")
    return ""

def e_mask(img):
    mask = np.zeros_like(img)

    sizeX = round(150 * 0.5)
    sizeY = round(150 * 0.8)

    centerX = int(round(DESIRED_FACE_WIDTH / 2))
    centerY = int(round(DESIRED_FACE_HEIGHT / 2))

    mask = cv.ellipse(mask, center=(centerX, centerY), axes=(sizeX,sizeY), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
    np.bitwise_and(img, mask)

    img[mask == 0] = 128
    return img

def smoothing(img):
    return cv.bilateralFilter(img, 0, 20, 20)

def histogram_equalisation(img):
    rows, cols = img.shape
    print(img.shape)
    wholeFace = cv.equalizeHist(img)
    midX = round(cols / 2)
    #xy wh
    #yx hw
    #leftSide = img[0:rows, 0:midX]
    #rightSide = img[0:rows, rows-midX:cols]
    #print("here")
    
    #leftSide = cv.equalizeHist(leftSide)
    #rightSide = cv.equalizeHist(rightSide)

    #show_image(leftSide)
    #show_image(rightSide)

    #test = leftSide + rightSide
    #show_image(test)

    #img = blend(wholeFace, leftSide, rightSide, midX, img)
    #show_image(img)
    return wholeFace

def blend(wholeFace, leftSide, rightSide, midX, img):
    cols, rows = img.shape
    for y in range(0, rows):
        for x in range(0, cols):
            if x < cols / 4:
                v = leftSide[x, y]
            elif x < (cols * 2 / 4):
                lv = leftSide[x, y]
                wv = wholeFace[x, y]
                f = (x - cols * 1 / 4) / (cols / 4)
                v = round((1.0 - f) * lv + (f) * wv)
            elif x < (cols * 3 / 4):
                rv = rightSide[x-midX, y]
                wv = wholeFace[x, y]
                f = (x - cols * 2 / 4) / (cols / 4)
                v = round((1.0 - f) * wv + (f) * rv)
            else:
                v = rightSide[x-midX, y]
            img[x,y] = v
        break

    return img

def geometrical_transformation(gray, rightEyeX, rightEyeY, leftEyeX, leftEyeY):
    # Rotate the face so eyes are horizontal
    # Scale face so distance between eyes is always the same
    # Translate so eyes are always centered horizontally and at desired height
    # Crop outer parts of face (image background, hair, forehead*, ears, chin*)

    # Get the center between the two eyes
    eyesCenterX = (leftEyeX + rightEyeX) * 0.5
    eyesCenterY = (leftEyeY + rightEyeY) * 0.5

    #cv.circle(gray, (int(eyesCenterX), int(eyesCenterY)), 1, (255,255,0), -1)

    # Get the angle between the two eyes
    dy = rightEyeY - leftEyeY
    dx = rightEyeX - leftEyeX
    length = math.sqrt(dx*dx + dy*dy)
    angle = math.atan2(dy, dx) * 180.0/np.pi

    # Desired positions
    desiredLength = (DESIRED_RIGHT_EYE_X - 0.16)
    scale = desiredLength * DESIRED_FACE_WIDTH / length

    # Get transformation matrix for desired angle and size
    rotationMatrix = cv.getRotationMatrix2D((eyesCenterX, eyesCenterY), angle, 1.5)
    ex = DESIRED_FACE_WIDTH * 0.5 - eyesCenterX
    ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenterY
    rotationMatrix[0,2] += ex
    rotationMatrix[1,2] += ey

    rows, cols = gray.shape
    gray = cv.warpAffine(gray, rotationMatrix, (rows, cols))
    return gray

def process(sourcefolder, targetfolder):
    for emotion in emotions:
        images = load_images(sourcefolder, emotion)
        if len(images) == 0:
            print("No files could be found for %s" %emotion)
        fileNumber = 0
        for file in images:
            result = detect_face(file)
            if result == "": continue
            cv.imwrite("%s/%s/%s.jpg" %(targetfolder, emotion, fileNumber), result)
            fileNumber += 1