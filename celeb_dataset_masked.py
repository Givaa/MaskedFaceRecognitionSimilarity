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
siamese = SiameseClass(directory=directory, 
model_save_path=directoryModel,
epochs=10, truncate_dataset=10000)

#Commentato perchè già addestrato
#siamese.train(save_model=True)

#labels
classes = siamese.class_names
#siamese.plot_accuracy()
#siamese.plot_loss()

#result = siamese.predict(path1="celeb_dataset_masked/unmasked/202599.jpg",
#                path2="celeb_dataset_masked/mask1/202599.jpg", visualize_result=True, save_image_path="foto.png")

identities = [f for f in listdir(join(directory,"unmasked")) if isfile(join(directory,"unmasked", f))]
random.shuffle(identities)
for fileName in identities[:20]:
    identity = fileName.split(".")[0]
    for maskType in classes:
        mask1 = siamese.predict(path1="celeb_dataset_masked/unmasked/{}".format(fileName),
                path2="celeb_dataset_masked/mask1/{}".format(fileName), visualize_result=True, save_image_path="result/mask1/{}.png".format(identity))
        mask2 = siamese.predict(path1="celeb_dataset_masked/unmasked/{}".format(fileName),
                path2="celeb_dataset_masked/mask2/{}".format(fileName), visualize_result=True, save_image_path="result/mask2/{}.png".format(identity))
        mask3 = siamese.predict(path1="celeb_dataset_masked/unmasked/{}".format(fileName),
                path2="celeb_dataset_masked/mask3/{}".format(fileName), visualize_result=True, save_image_path="result/mask3/{}.png".format(identity))
        mask4 = siamese.predict(path1="celeb_dataset_masked/unmasked/{}".format(fileName),
                path2="celeb_dataset_masked/mask4/{}".format(fileName), visualize_result=True, save_image_path="result/mask4/{}.png".format(identity))
    print("Identità: {}. Mask1: {:.5f}. Mask2: {:.5f}. Mask3: {:.5f}. Mask4: {:.5f}".format(identity,mask1,mask2,mask3,mask4))
print("FATTO")
