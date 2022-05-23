from genericpath import isfile
from ntpath import join
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from siamese_network import SiameseClass

#Training dataset
directory = "celeb_dataset_masked_type"

#Dataset used for testing
directoryTest = "dataset_test/celeb_dataset"

#keras model save path
directoryModel = "model_save/{}/".format(directory)

#Create a Siamese Network instance. truncate_dataset=10000 will load only the first 10000 identities due to memory restrictions.
siamese = SiameseClass(directory=directory, 
model_save_path=directoryModel,
epochs=10 , truncate_dataset=10000)

#Train the model
#siamese.train(save_weights=True)

#Labels
classes = siamese.class_names

#Plot accuracy, loss
siamese.plot_accuracy()
siamese.plot_loss()
siamese.plot_f1()
siamese.plot_mse()
siamese.plot_precision()
siamese.plot_recall()

identities = [f for f in listdir(join(directoryTest,"unmasked")) if isfile(join(directoryTest,"unmasked", f))]

#Loop on test identities.
for fileName in identities:
        identity = fileName.split(".")[0]

        #Compare unmasked identity with type1 mask
        mask1 = siamese.predict(path1="{}/unmasked/{}".format(directoryTest,fileName),
                path2="{}/mask1/{}".format(directoryTest,fileName), visualize_result=False, save_image_path="result/celeb_dataset/mask1.png")

        #Compare unmasked identity with type2 mask
        mask2 = siamese.predict(path1="{}/unmasked/{}".format(directoryTest,fileName),
                path2="{}/mask2/{}".format(directoryTest,fileName), visualize_result=False, save_image_path="result/celeb_dataset/mask2.png")

        #Compare unmasked identity with type3 mask
        mask3 = siamese.predict(path1="{}/unmasked/{}".format(directoryTest,fileName),
                path2="{}/mask3/{}".format(directoryTest,fileName), visualize_result=False, save_image_path="result/celeb_dataset/mask3.png")

        #Compare unmasked identity with type4 mask
        mask4 = siamese.predict(path1="{}/unmasked/{}".format(directoryTest,fileName),
                path2="{}/mask4/{}".format(directoryTest,fileName), visualize_result=False, save_image_path="result/celeb_dataset/mask4.png")
        print("Identity: {}. Mask1: {:.5f}. Mask2: {:.5f}. Mask3: {:.5f}. Mask4: {:.5f}".format(identity, mask1,mask2,mask3,mask4))
print("FATTO")
