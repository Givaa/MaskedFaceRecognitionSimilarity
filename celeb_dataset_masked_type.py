from genericpath import isfile
from ntpath import join
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from siamese_network import SiameseClass

directory = "celeb_dataset_masked_type"
directoryTest = "dataset_test/celeb_dataset"
directoryModel = "model_save/{}/".format(directory)

#truncate_dataset=None dà problemi allocazione memoria
siamese = SiameseClass(directory=directory, 
model_save_path=directoryModel,
epochs=10 , truncate_dataset=4300)

#Commentato perchè già addestrato
#siamese.train(save_model=False)

#labels
classes = siamese.class_names
#siamese.plot_accuracy()
#siamese.plot_loss()

#result = siamese.predict(path1="celeb_dataset_masked_type/unmasked/202599.jpg",
#                path2="celeb_dataset_masked_unmasked/mask1/202599.jpg", visualize_result=True, save_image_path="foto.png")

identities = [f for f in listdir(join(directoryTest,"unmasked")) if isfile(join(directoryTest,"unmasked", f))]

for fileName in identities:
        identity = fileName.split(".")[0]
        unmasked = siamese.predict(path1="{}/unmasked/{}".format(directoryTest,fileName),
                path2="{}/unmasked/{}".format(directoryTest,fileName))
        mask1 = siamese.predict(path1="{}/unmasked/{}".format(directoryTest,fileName),
                path2="{}/mask1/{}".format(directoryTest,fileName))
        mask2 = siamese.predict(path1="{}/unmasked/{}".format(directoryTest,fileName),
                path2="{}/mask2/{}".format(directoryTest,fileName))
        mask3 = siamese.predict(path1="{}/unmasked/{}".format(directoryTest,fileName),
                path2="{}/mask3/{}".format(directoryTest,fileName))
        mask4 = siamese.predict(path1="{}/unmasked/{}".format(directoryTest,fileName),
                path2="{}/mask4/{}".format(directoryTest,fileName))
        print("Identità: {}. Unmasked: {:.5f}. Mask1: {:.5f}. Mask2: {:.5f}. Mask3: {:.5f}. Mask4: {:.5f}".format(identity,unmasked, mask1,mask2,mask3,mask4))
print("FATTO")
