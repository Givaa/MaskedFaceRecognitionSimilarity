from genericpath import isfile
from ntpath import join
from os import listdir, path
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from siamese_network import SiameseClass

directory = "m2fred_face"
directoryTest = "dataset_test/m2fred_face"
directoryModel = "model_save/{}/".format(directory)

#truncate_dataset=None dà problemi allocazione memoria
siamese = SiameseClass(directory=directory, 
model_save_path=directoryModel,
epochs=20)

#Commentato perchè già addestrato
#siamese.train(save_model=True)

#labels
classes = siamese.class_names
#siamese.plot_accuracy()
#siamese.plot_loss()

#result = siamese.predict(path1="celeb_dataset_masked_person/unmasked/202599.jpg",
#                path2="celeb_dataset_masked_unmasked/mask1/202599.jpg", visualize_result=True, save_image_path="foto.png")

identities = listdir(join(directoryTest))

for identity in identities:
        identifyFolderTest = path.join(directoryTest,identity)
        unmaskedFiles = list(filter(lambda file: file.find("-1-1") >0, listdir(identifyFolderTest)))

        for unmaskedFile in unmaskedFiles:
                videoSession = unmaskedFile.split("-")[0]
                maskedFile = "{}-1-1.png".format(videoSession)
                if(isfile(maskedFile)):
                        score = siamese.predict(path1=path.join(directoryTest,identity, unmaskedFile),
                                path2=path.join(directoryTest,identity, maskedFile), visualize_result=True)
                        print("Identità: {}. Session: {}. Score: {}".format(identity, videoSession, score))
print("FATTO")
