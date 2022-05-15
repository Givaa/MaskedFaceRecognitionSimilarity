from genericpath import isdir, isfile
from ntpath import join
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from utility import euclidean_distance, get_class_from_file, load_img, loss, make_pairs, plt_metric, visualize

class SiameseClass:

    def __init__(self, directory = "/dataset", model_save_path=None, epochs=10, target_shape = (128, 128), margin=1):
        self.directory = directory
        self.model_save_path = model_save_path
        self.epochs = epochs
        self.target_shape = target_shape
        self.class_names=[]

        directories = [f for f in listdir(directory) if isdir(join(directory, f))]
        if len(directories) > 0:
            batch_size = [f for f in listdir(join(directory,directories[0])) if isfile(join(directory,directories[0], f))]
            self.batch_size = len(batch_size)
        else:
            self.batch_size = 32

        self.margin=margin
        self.siamese = self.siamese_boilerplate()

        if model_save_path != None and isdir(model_save_path):
            self.siamese.load_weights(model_save_path)
        else:
            print("No model found. Model needs to be trained.")
            

    def train(self, save_model = False):
        train_ds = keras.utils.image_dataset_from_directory(
            self.directory,
            batch_size=self.batch_size,
            color_mode="rgb",
            image_size=self.target_shape,
            labels='inferred', label_mode='int',
        )
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]

        x_train_val =[]
        y_train_val =[]

        self.class_names = train_ds.class_names

        for images, labels in train_ds.take(1):  # only take first element of dataset
            x_train_val = images.numpy()
            y_train_val = labels.numpy()

        x_train_val  = x_train_val.astype("uint8")

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
            batch_size=32,
            epochs=self.epochs,
        )

        
        if save_model == True:
            siamese.save_weights(self.model_save_path)
            #TODO SAVE CLASSNAME

        self.history = history
        self.siamese = siamese

        #predictions = siamese.predict([x_test_1, x_test_2])
        #visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)

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

        siamese.compile(loss=loss(margin=self.margin), optimizer="RMSprop", metrics=["accuracy"])
        siamese.summary()
        return siamese

    def plot_accuracy(self):
        plt_metric(history=self.history.history, metric="accuracy", title="Model accuracy")

    def plot_loss(self):
        plt_metric(history=self.history.history, metric="loss", title="Constrastive Loss")

    def predict(self, path1="", path2="", visualize_result=False, save_image_path=None):
        file1 = load_img(path1, (128, 128))
        class1 = get_class_from_file(path1)

        file2 = load_img(path2, (128, 128))
        class2 = get_class_from_file(path2)

        results = self.siamese.predict([file1,file2])


        #fig, ax = plt.subplots()
        #ax.set_title('Pred: ')#{:.5f}', results[0][0])
        #ax.imshow(tf.concat([file1[0], file2[0]], axis=1), cmap="gray")

        if visualize_result == True:
            fig = plt.figure()
            plt.title("Pred: {:.5f}".format(results[0][0]))
            plt.axis('off')
            ax = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(file1[0])
            ax.set_title(class1)
            ax.set_axis_off()
            ax = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(file2[0])
            imgplot.set_clim(0.0, 0.7)
            ax.set_title(class2)
            ax.set_axis_off()
            plt.show()
            if save_image_path != None:
                fig.savefig(save_image_path)

        return results[0][0]