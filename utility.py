import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")

def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


    def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
        """Creates a plot of pairs and labels, and prediction if it's test dataset.

        Arguments:
            pairs: Numpy Array, of pairs to visualize, having shape
                   (Number of pairs, 2, 28, 28).
            to_show: Int, number of examples to visualize (default is 6)
                    `to_show` must be an integral multiple of `num_col`.
                     Otherwise it will be trimmed if it is greater than num_col,
                     and incremented if if it is less then num_col.
            num_col: Int, number of images in one row - (default is 3)
                     For test and train respectively, it should not exceed 3 and 7.
            predictions: Numpy Array of predictions with shape (to_show, 1) -
                         (default is None)
                         Must be passed when test=True.
            test: Boolean telling whether the dataset being visualized is
                  train dataset or test dataset - (default False).

        Returns:
            None.
        """

        # Define num_row
        # If to_show % num_col != 0
        #    trim to_show,
        #       to trim to_show limit num_row to the point where
        #       to_show % num_col == 0
        #
        # If to_show//num_col == 0
        #    then it means num_col is greater then to_show
        #    increment to_show
        #       to increment to_show set num_row to 1
        num_row = to_show // num_col if to_show // num_col != 0 else 1

        # `to_show` must be an integral multiple of `num_col`
        #  we found num_row and we have num_col
        #  to increment or decrement to_show
        #  to make it integral multiple of `num_col`
        #  simply set it equal to num_row * num_col
        to_show = num_row * num_col

        # Plot the images
        fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
        for i in range(to_show):

            # If the number of rows is 1, the axes array is one-dimensional
            if num_row == 1:
                ax = axes[i % num_col]
            else:
                ax = axes[i // num_col, i % num_col]

            ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
            ax.set_axis_off()
            if test:
                ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
            else:
                ax.set_title("Label: {}".format(labels[i]))
        if test:
            plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
        else:
            plt.tight_layout(rect=(0, 0, 1.5, 1.5))
        plt.show()

def load_img(image_path="", target_size=(200,200)):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    return input_arr.astype("uint8")


def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()


def get_class_from_file(path):
    path = Path(path)
    return path.parts[-2]
