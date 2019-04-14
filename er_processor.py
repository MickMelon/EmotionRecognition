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
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_casecade.detectMultiScale(roi_gray, 1.3, 10)
        rightEye = eyes[1]
        leftEye = eyes[0]

        cv.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

        for (ex,ey,ew,eh) in eyes:
            roi_eye = gray[y:ey+eh, x:ex+ew]
            cv.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        roi_gray = cv.resize(roi_gray, (DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT))

        show_image(roi_gray)

        roi_gray = histogram_equalisation(roi_gray)
        #sys.exit()
        show_image(roi_gray) 

        roi_gray = smoothing(roi_gray)
        show_image(roi_gray) 

        roi_gray = cv.GaussianBlur(roi_gray, (5,5), 0)
        print("gauss")
        show_image(roi_gray)

        roi_gray = elliptical_mask(roi_gray)
        show_image(roi_gray)  

        sys.exit()
        
       # dst, gray = geometrical_transformation(gray, rightEye, leftEye)

        return face, mask

    print("No face detected")
    return "", ""

def elliptical_mask(img):
    mask = np.zeros_like(img)
    rows, cols = mask.shape
    print("rows %s cols %s" %(rows, cols))

    faceCenterX = round(cols * 0.5)
    faceCenterY = round(rows * 0.5)

    mask = cv.ellipse(mask, center=(faceCenterY, faceCenterX), axes=(125,175), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
    result = np.bitwise_and(img, mask)

    img[mask == 0] = 128

    return img

def e_mask(img):
    mask = np.zeros((DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT), np.uint8)

    rows, cols = mask.shape
    faceCenterX = round(cols * 0.5)
    faceCenterY = round(rows * 0.5)
    mask = cv.ellipse(mask, center=(faceCenterY, faceCenterX), axes=(125,175), angle=0, startAngle=0, endAngle=360, color=(0,255,0), thickness=-1)
   # mask = cv.ellipse(mask,(130,130),(100,50),0,0,180,255,-1)
    #result = np.bitwise_and(img, mask)
    show_image(mask)

    #img[mask == 0] = 128
    print("t")

    return mask

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

def geometrical_transformation(gray, leftEye, rightEye):
    # Rotate the face so eyes are horizontal
    # Scale face so distance between eyes is always the same
    # Translate so eyes are always centered horizontally and at desired height
    # Crop outer parts of face (image background, hair, forehead*, ears, chin*)

    # Get the center between the two eyes
    eyesCenterX = (leftEye[0] + rightEye[0]) * 0.5
    eyesCenterY = (leftEye[1] + rightEye[1]) * 0.5

    # Get the angle between the two eyes
    dy = rightEye[1] - leftEye[1]
    dx = rightEye[0] - leftEye[0]
    length = math.sqrt(dx*dx + dy*dy)
    angle = cv.fastAtan2(dy, dx) * 180.0/np.pi

    

    # Desired positions
    DESIRED_LEFT_EYE_X = 0.16
    DESIRED_LEFT_EYE_Y = 0.14
    DESIRED_RIGHT_EYE_X = 1.0 - 0.16
    DESIRED_RIGHT_EYE_Y = 1.0 - 0.14
    DESIRED_FACE_WIDTH = 70
    DESIRED_FACE_HEIGHT = 70
    desiredLength = (DESIRED_RIGHT_EYE_X - 0.16)
    scale = desiredLength * DESIRED_FACE_WIDTH / length

    # Get transformation matrix for desired angle and size
    rotationMatrix = cv.getRotationMatrix2D((eyesCenterX, eyesCenterY), angle, scale)
    ex = DESIRED_FACE_WIDTH * 0.5 - eyesCenterX
    ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenterY
    rotationMatrix[0,2] += ex
    rotationMatrix[1,2] += ey

    rows, cols = gray.shape
    dst = cv.warpAffine(gray, rotationMatrix, (150, 150))
    return dst, gray

def process(sourcefolder, targetfolder):
    for emotion in emotions:
        images = load_images(sourcefolder, emotion)
        if len(images) == 0:
            print("No files could be found for %s" %emotion)
        fileNumber = 0
        for file in images:
            facefeatures, gray = detect_face(file)
            for (x, y, w, h) in facefeatures:
                gray = gray[y:y+h, x:x+w]
                try:
                    out = cv.resize(gray, (150, 150))
                    cv.imwrite("%s/%s/%s.jpg" %(targetfolder, emotion, fileNumber), out)
                except:
                    print("EXCEPTION OCCURED")
                    pass
            fileNumber += 1