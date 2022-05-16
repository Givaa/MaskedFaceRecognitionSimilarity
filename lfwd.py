from genericpath import isfile
from ntpath import join
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from siamese_network import SiameseClass

directory = "lfwd_dataset"
directoryTest = "dataset_test/lfwd_dataset"
directoryModel = "model_save/{}/".format(directory)

#truncate_dataset=None dà problemi allocazione memoria
siamese = SiameseClass(directory=directory, 
model_save_path=directoryModel,
epochs=10, truncate_dataset=10000)

#Commentato perchè già addestrato
#siamese.train(save_model=True)

#labels
classes = siamese.class_names
#siamese.plot_accuracy()
#siamese.plot_loss()

#result = siamese.predict(path1="lfwd_dataset/unmasked/202599.jpg",
#                path2="celeb_dataset_masked_unmasked/mask1/202599.jpg", visualize_result=True, save_image_path="foto.png")

identities = [f for f in listdir(join(directoryTest,"LFW_without_Mask")) if isfile(join(directoryTest,"LFW_without_Mask", f))]

for fileName in identities:
        identity = fileName.split(".")[0]
        unmasked = siamese.predict(path1="{}/LFW_without_Mask/{}".format(directoryTest,fileName),
                path2="{}/LFW_without_Mask/{}".format(directoryTest,fileName))
        masked = siamese.predict(path1="{}/LFW_without_Mask/{}".format(directoryTest,fileName),
                path2="{}/Masked_LFW_Dataset/{}".format(directoryTest,fileName))
       
        print("Identità: {}. Unmasked: {:.5f}. Mask1: {:.5f}".format(identity,unmasked, masked))
print("FATTO")
