import cv2 as cv
import glob

emotions = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

faceDet = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

def load_images(folder, emotion):
    return glob.glob("%s//%s//*" %(folder, emotion))

def detect_face(file):
    print("Beginning to detect a face...")
    frame = cv.imread(file)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face = faceDet.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    print(face)

    if len(face) == 1:
        return face, gray

    print("No face detected")
    return "", ""

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