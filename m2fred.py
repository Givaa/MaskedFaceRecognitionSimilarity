from genericpath import isfile
from ntpath import join
from os import listdir, path
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from siamese_network import SiameseClass

#Training dataset
directory = "m2fred_face"

#Dataset used for testing
directoryTest = "dataset_test/m2fred_face"

#keras model save path
directoryModel = "model_save/{}/".format(directory)

#Create a Siamese Network instance.
siamese = SiameseClass(directory=directory, 
model_save_path=directoryModel,
epochs=10)

#Train the model
#siamese.train(save_weights=True)

#Plot accuracy, loss
siamese.plot_accuracy()
siamese.plot_loss()
siamese.plot_f1()
siamese.plot_mse()
siamese.plot_precision()
siamese.plot_recall()


identities = listdir(join(directoryTest))
i=1
#Loop on test identities.
for identity in identities:
        identifyFolderTest = path.join(directoryTest,identity)

        #Load images without mask
        unmaskedFiles = list(filter(lambda file: file.find("-0-0") >0, listdir(identifyFolderTest)))

        for unmaskedFile in unmaskedFiles:
                videoSession = unmaskedFile.split("-")[0]

                #Masked image file name.
                maskedFile = "{}-1-0.png".format(videoSession)

                if(isfile(path.join(directoryTest,identity, maskedFile))):
                        #Compare image of the same session. Unm
                        unmaskedPath = path.join(directoryTest,identity, unmaskedFile)
                        maskedPath = path.join(directoryTest,identity, maskedFile)
                        score = siamese.predict(path1=unmaskedPath,
                                path2=maskedPath, visualize_result=False, save_image_path="result/m2fred/{}.png".format(i))
                        i+=1
                        print("Identity: {}. Session: {}. Score: {}".format(identity, videoSession, score))
print("FATTO")
