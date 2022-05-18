from genericpath import isdir
from os import listdir, mkdir, path
import re
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from PIL import Image
import threading

directory = "video"

#Train set destination
destinationDirectory = "m2fred_face"

#comparison set destination
destinationComparisonDirectory = "dataset_test/m2fred_face"

#Save images in this resolution
required_size=(160, 160)

THREADS = 30

folders = list(filter(lambda file: file.find("_0") >0, listdir(directory)))
for i, f in enumerate(folders):
    folders[i] = f.split("_0")[0]


#Path is the video input path
#trainDestination is the destination folder where images will be saved
#comparisonDestination is the comparison destination folder where images will be saved 
#onlyFirstCapture: Only the first image will be saved (used for the test set)
def frameExtractor(path, trainDestination, comparisonDestination, onlyFirstCapture=False):
    vidcap = cv2.VideoCapture(path)

    fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    #Duration is divided into 11 parts. In this way there will be 11 images for each video
    #10 images will be used for the train, 1 for the comparison
    duration = frame_count/fps * 1000 / 11

    i=0
    count=0

    while True:
        success,image = vidcap.read()
        if success == False:
            break
        
        #Convert images from BGR -> RGB. 
        # By default cv2 loads image in BGR.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Convert image in numpy array
        pixels = np.asarray(image)

        #Extract face details with MTCNN
        detector = MTCNN()
        results = detector.detect_faces(image)
        if len(results) > 0:
            x1, y1, width, height = results[0]['box']
            # deal with negative pixel index
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            if i < 10:
                image.save("{}-{}.png".format(trainDestination,i))
            else:
                image.save("{}-0.png".format(comparisonDestination))
            i+=1
            if onlyFirstCapture == True:
                break
        count+=1
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*duration))    # Change the current position of the video file of count*duration milliseconds
    #Exit semaphore
    semaphore.release()
    
#Semaphore for multi-thread
semaphore = threading.Semaphore(THREADS) 

for identity in folders:
    trainDestinationFolder = path.join(destinationDirectory, identity)
    if isdir(trainDestinationFolder) == False:
            mkdir(trainDestinationFolder)

    comparisonDestinationFolder = path.join(destinationComparisonDirectory, identity)
    if isdir(comparisonDestinationFolder) == False:
        mkdir(comparisonDestinationFolder)

    videosUnmasked = listdir(path.join(directory, identity+"_0"))
    for v in videosUnmasked:
        video = path.join(directory, identity+"_0", v)

        #Extract from the video's title the session and acquisition number. 
        # 043_3_1_2 => session #3 and acquisition #2 
        videoType = re.findall(r'\d{3}_(\d)_\d_(\d)', v)[0]

        #Lock semaphore
        semaphore.acquire()
        threading.Thread(target=frameExtractor, args=(video, 
            "{}\\{}-{}".format(trainDestinationFolder,videoType[0], videoType[1]),
            "{}\\{}-0".format(comparisonDestinationFolder,videoType[0]),
        )).start()
    
    #Load video with mask. Load only videos with acquisition number 1
    videosMasked = list(filter(lambda file: re.findall("_\d_\d_1\.", file), listdir(path.join(directory, identity+"_1"))))
    for v in videosMasked:
        video = path.join(directory, identity+"_1", v)

        #Extract from the video's title the session and acquisition number. 
        # 043_3_1_2 => session #3 and acquisition #2 
        videoType = re.findall(r'\d{3}_(\d)_\d_(\d)', v)[0]
        semaphore.acquire()
        threading.Thread(target=frameExtractor, args=(video, 
            "{}\\{}-1".format(comparisonDestinationFolder,videoType[0]),
            None,
            True,)).start()
    
#Wait and join all the remaining threads
main_thread = threading.currentThread()
for t in threading.enumerate():
    if t is not main_thread:
        t.join()      

