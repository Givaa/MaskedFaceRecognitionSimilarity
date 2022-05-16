from genericpath import isfile
from ntpath import join
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from siamese_network import SiameseClass

directory = "face_mask_detection_database"
directoryTest = "dataset_test/face_mask_detection_database"
directoryModel = "model_save/{}/".format(directory)

siamese = SiameseClass(directory=directory, 
model_save_path=directoryModel,
epochs=10)

#Commentato perchè già addestrato
#siamese.train(save_model=True)

#labels
classes = siamese.class_names
#siamese.plot_accuracy()
#siamese.plot_loss()

#result = siamese.predict(path1="celeb_dataset_masked_person/unmasked/202599.jpg",
#                path2="celeb_dataset_masked_unmasked/mask1/202599.jpg", visualize_result=True, save_image_path="foto.png")

identities = [f for f in listdir(join(directoryTest,"without_mask")) if isfile(join(directoryTest,"without_mask", f))]

for fileName in identities:
        identity = fileName.split(".")[0]
        #TODO Che criterio usiamo?
print("FATTO")
