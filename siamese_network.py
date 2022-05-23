import pickle
from genericpath import isdir, isfile
from ntpath import join
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from utility import euclidean_distance, f1_m, get_class_from_file, load_img, loss, make_pairs, mean_squared_error, plt_metric, precision_m, recall_m, visualize

class SiameseClass:
    def __init__(self, directory = "/dataset", model_save_path=None, epochs=10, target_shape = (128, 128), margin=1, truncate_dataset=None):
        """
        Init a Siamese network

        :param directory: training dataset directory path
        :param model_save_path: directory where keras saves the layer weights
        :param epochs: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
        :param target_shape: image dimension
        :param margin: Integer, defines the baseline for distance for which pairs should be classified as dissimilar. - (default is 1).
        :param truncate_dataset: Integer. Loads only the first truncate_dataset entries. Useful to reduce the memory usage.
        :return: SiameseClass
        """ 
        self.directory = directory
        self.model_save_path = model_save_path
        self.epochs = epochs
        self.target_shape = target_shape
        self.class_names=[]

        directories = [f for f in listdir(directory) if isdir(join(directory, f))]
        self.class_names = directories

        if truncate_dataset == None and len(directories) > 0:
            truncate_dataset = [f for f in listdir(join(directory,directories[0])) if isfile(join(directory,directories[0], f))]
            self.batch_size = len(truncate_dataset)
        elif truncate_dataset != None:
            self.batch_size = truncate_dataset
        else:
            self.batch_size = 32

        self.margin=margin
        self.siamese = self.siamese_boilerplate()

        if model_save_path != None and isdir(model_save_path):
            self.siamese.load_weights(model_save_path)
            self.history = pickle.load(open(self.model_save_path+"history", "rb"))
        else:
            print("No model found. Model needs to be trained.")
            

    def train(self, save_weights = False):
        """
        Train Siamese network

        :param save_weights: boolean. Saves the model weights in model_save_path
        :return: None
        """ 
        train_ds = keras.utils.image_dataset_from_directory(
            self.directory,
            batch_size=self.batch_size,
            color_mode="rgb",
            image_size=self.target_shape,
            labels='inferred', label_mode='int',
        )

        x_train_val =[]
        y_train_val =[]

        self.class_names = train_ds.class_names

        for images, labels in train_ds.take(1):  # only take first element of dataset
            x_train_val = images.numpy()
            y_train_val = labels.numpy()

        x_train_val  = x_train_val.astype("float32")
        y_train_val  = y_train_val.astype("uint8")

        lenght = len(x_train_val)
        train = int(lenght*.7)
        test = int(lenght*.9)
        val = int(lenght)

        self.x_train, self.x_test, self.x_val = x_train_val[:train], x_train_val[train:test], x_train_val[test:val]
        self.y_train, self.y_test, self.y_val = y_train_val[:train], y_train_val[train:test], y_train_val[test:val]


        pairs_train, labels_train = make_pairs(self.x_train, self.y_train)

        # make validation pairs
        pairs_val, labels_val = make_pairs(self.x_val, self.y_val)

        # make test pairs
        pairs_test, labels_test = make_pairs(self.x_test, self.y_test)

        x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
        x_train_2 = pairs_train[:, 1]

        x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (60000, 28, 28)
        x_val_2 = pairs_val[:, 1]

        x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (12800, 28, 28)
        x_test_2 = pairs_test[:, 1]

        siamese = self.siamese_boilerplate()
        
        history = siamese.fit(
            [x_train_1, x_train_2],
            labels_train,
            validation_data=([x_val_1, x_val_2], labels_val),
            batch_size=20,
            epochs=self.epochs,
        )

        
        if save_weights == True:
            siamese.save_weights(self.model_save_path)
            with open(self.model_save_path+"history", 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            #TODO SAVE CLASSNAME

        self.history = history
        self.siamese = siamese

    def siamese_boilerplate(self):
        input = tf.keras.layers.Input((128, 128, 3))
        x = tf.keras.layers.BatchNormalization()(input)
        x = tf.keras.layers.Conv2D(4, (5, 5), activation="tanh")(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(16, (5, 5), activation="tanh")(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(10, activation="tanh")(x)
        embedding_network = keras.Model(input, x)


        input_1 = tf.keras.layers.Input((128, 128, 3))
        input_2 = tf.keras.layers.Input((128, 128, 3))

        # As mentioned above, Siamese Network share weights between
        # tower networks (sister networks). To allow this, we will use
        # same embedding network for both tower networks.
        tower_1 = embedding_network(input_1)
        tower_2 = embedding_network(input_2)

        merge_layer = tf.keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
        normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
        output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(normal_layer)
        siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

        siamese.compile(loss=loss(margin=self.margin), optimizer="RMSprop", metrics=["accuracy", f1_m,precision_m, recall_m, mean_squared_error])
        siamese.summary()
        return siamese

    def plot_accuracy(self):
        """
        Plot accuracy
        """ 
        plt_metric(history=self.history, metric="accuracy", title="Accuracy")

    def plot_loss(self):
        """
        Plot Constrastive loss metrics
        """ 
        plt_metric(history=self.history, metric="loss", title="Constrastive Loss")

    def plot_f1(self):
        """
        Plot F1 score
        """ 
        plt_metric(history=self.history, metric="f1_m", title="F1 score")

    def plot_precision(self):
        """
        Plot precision score
        """ 
        plt_metric(history=self.history, metric="precision_m", title="Precision")

    def plot_recall(self):
        """
        Plot recall metric
        """ 
        plt_metric(history=self.history, metric="recall_m", title="Recall")

    def plot_mse(self):
        """
        Plot Mean Squared Error
        """ 
        plt_metric(history=self.history, metric="mean_squared_error", title="MSE")

    def predict(self, path1="", path2="", visualize_result=False, save_image_path=None):
        """
        Predict the difference between two images

        :param path1: First image path
        :param path2: Second image path
        :param visualize_result: boolean. Plot the result with the two images
        :param save_image_path: save the plot result as image
        :return: SiameseClass
        """ 
        file1 = load_img(path1, (128, 128))
        class1 = get_class_from_file(path1)

        file2 = load_img(path2, (128, 128))
        class2 = get_class_from_file(path2)

        results = self.siamese.predict([file1,file2])

        if visualize_result == True:
            fig = plt.figure()
            plt.title("Pred: {:.5f}".format(results[0][0]))
            plt.axis('off')
            ax = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(file1[0].astype("uint8"))
            ax.set_title(class1)
            ax.set_axis_off()
            ax = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(file2[0].astype("uint8"))
            imgplot.set_clim(0.0, 0.7)
            ax.set_title(class2)
            ax.set_axis_off()
            #plt.show()
            if save_image_path != None:
                fig.savefig(save_image_path)

        return results[0][0]