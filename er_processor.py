import cv2 as cv
import glob
import numpy as np
import math
import crop_face
from PIL import Image

emotions = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_casecade = cv.CascadeClassifier('haarcascade_eye.xml')

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
        eyes = eye_casecade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))
        rightEye = eyes[1]
        leftEye = eyes[0]

        for (ex,ey,ew,eh) in eyes:
            roi_eye = gray[y:ey+eh, x:ex+ew]

        result = histogram_equalisation(gray)
        blur = smoothing(result)
        mask = elliptical_mask(blur)     
        
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

def smoothing(img):
    blur = cv.bilateralFilter(img, 9, 75, 75)
    return blur

def histogram_equalisation(img):
    result = cv.equalizeHist(img)
    return result

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