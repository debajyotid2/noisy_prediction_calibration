"""
Basic convolutional network for image classification.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, InputLayer, Conv2DTranspose
from tensorflow.keras.layers import Dropout, Dense, Flatten, Reshape


class MyModel(Model):
    """
    Convolutional variational autoencoder.
    """

    def __init__(
        self
    ):
        super().__init__()
    
    def call(self, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Returns model outputs.
        """
