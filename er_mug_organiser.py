# Organises the MUG images into emotions folders.
import glob
import os

def organise(folder):
    files = glob.glob("%s/*" %folder)
    fileNumber = 0
    print("wtf")
    for file in files:
        if "an" in file:            
            #anger
            os.rename(file, "mug_dataset/anger/%i.jpg" %fileNumber)
        elif "di" in file:
            #disgust
            os.rename(file, "mug_dataset/disgust/%i.jpg" %fileNumber)
        elif "fe" in file:
            #fear
            os.rename(file, "mug_dataset/fear/%i.jpg" %fileNumber)
        elif "ha" in file:
            #happy
            os.rename(file, "mug_dataset/happy/%i.jpg" %fileNumber)
        elif "ne" in file:
            #neutral
            os.rename(file, "mug_dataset/neutral/%i.jpg" %fileNumber)
        elif "sa" in file:
            #sadness
            os.rename(file, "mug_dataset/sadness/%i.jpg" %fileNumber)
        elif "su" in file:
            #surprised
            os.rename(file, "mug_dataset/surprise/%i.jpg" %fileNumber)

        fileNumber += 1
        print("loop")
        
organise("mugset")