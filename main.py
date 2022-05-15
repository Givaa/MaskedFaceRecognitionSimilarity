import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from siamese_network import SiameseClass

directory = "celeb_dataset"
directoryModel = "model_save/celeb_dataset"
siamese = SiameseClass(directory=directory, model_save_path=directoryModel)
#siamese.train(save_model=True)
result = siamese.predict(path1="celeb_dataset/mask1/201209.jpg",
                path2="celeb_dataset/mask1/201209.jpg", visualize_result=True, save_image_path="foto.png")

print("FATTO")
