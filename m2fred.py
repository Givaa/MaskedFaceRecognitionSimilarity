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
directoryTest = "dataset_test/m2fred"
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
        identifyFolder = path.join(directory,identity)
        for i in [1,2,3,4]:
                testFile = path.join(identifyFolderTest,"{}.png".format(i))
                file = path.join(identifyFolder,"{}.png".format(i))
                if isfile(file) and isfile(testFile):
                        score = siamese.predict(path1=file,
                                path2=testFile, visualize_result=True)
                        print("Identità: {}. File: {}. Score: {}".format(identity, i, score))
print("FATTO")
