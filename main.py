from genericpath import isfile
from ntpath import join
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from siamese_network import SiameseClass

directory = "celeb_dataset_masked"
directoryModel = "model_save/celeb_dataset_masked/"

#truncate_dataset=None dà problemi allocazione memoria
siamese = SiameseClass(directory=directory, model_save_path=directoryModel, epochs=10, truncate_dataset=10000)

#Commentato perchè già addestrato
#siamese.train(save_model=True)

#labels
classes = siamese.class_names
#siamese.plot_accuracy()
#siamese.plot_loss()

result = siamese.predict(path1="celeb_dataset_masked/unmasked/202599.jpg",
                path2="celeb_dataset_masked/mask1/202599.jpg", visualize_result=True, save_image_path="foto.png")

#identities = [f for f in listdir(join(directory,"unmasked")) if isfile(join(directory,"unmasked", f))]
#for fileName in identities[:-100]:
#    identity = fileName.split(".")[0]
#    for maskType in classes:
#        print("")
#        #TODO
#        #siamese.predict(path1="celeb_dataset_masked/unmasked/"+id,
#        #        path2="celeb_dataset_masked/mask1/"+id)

print("FATTO")
