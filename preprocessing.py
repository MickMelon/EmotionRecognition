import cv2 as cv
import glob

# Declare the emotions used for recognition
emotions = ["neutral", "anger", "sadness", "happy", "fear", "surprise", "disgust"]

# Declare the HAAR cascade classifier used for detecting faces
faceDet = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the files for a given emotion
def load_files(emotion):
    print("Beginning to load files for %s" %emotion)
    return glob.glob("googleset//%s//*" %emotion)

# Detect a face in an image using the HAAR cascade classifier
def detect_face(file):
    print("Beginning to detect a face...")
    frame = cv.imread(file)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv.CASCADE_SCALE_IMAGE)

    if len(face) == 1:
        return face, gray

    print("No face detected")
    return "", ""

# Carry out preprocessing steps on the image extracted from the HAAR cascade classifier
def preprocess(facefeatures, gray):
    print("Beginning preprocessing steps for an image")
    for (x, y, w, h) in facefeatures:
        gray = gray[y:y+h, x:x+w]
        try:
            output = cv.resize(gray, (350, 350))
            return output
        except:
            print("Something strange happened")
            pass

    return ""

# Saves the given file to the dataset
def save_file(file, emotion, fileNumber):
    cv.imwrite("dataset/%s/%s.jpg" %(emotion, fileNumber), file)

# Executes the preprocessing step
def preprocessing():
    # Detect the face and crop it into a new image
    # Convert it to grayscale
    # Resize it to a standard size
    # Histogram equalization to smooth out lighting differences
    # Apply biltateral filter to smooth out small details
    print("Beginning preprocessing...")
    for emotion in emotions:
        files = load_files(emotion)
        i = 0

        if len(files) == 0:
            print("No files could be found for %s" %emotion)
        for file in files:
            facefeatures, gray = detect_face(file)
            output = preprocess(facefeatures, gray)
            if output == "":
                print("wtf")
                continue
            save_file(output, emotion, i)
            i += 1
    print("Ended preprocessing!")
    return
