"""
Basic convolutional network for image classification.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Dropout,
    InputLayer,
    BatchNormalization,
)


def softplus(x: tf.Tensor, beta: float) -> tf.Tensor:
    """
    softplus(x) = 1/beta * log(1+ exp(beta * x))
    """
    if beta == 0.0:
        return 0.0
    return 1.0 / beta * tf.math.log(1 + tf.exp(beta * x))


class CNN(Model):
    """
    Convolutional image classifier.
    """

    def __init__(
        self,
        *,
        n_classes: int = 10,
        dim: int = 16,
        img_height: int = 28,
        img_width: int = 28,
        n_channels: int = 1,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.dim = dim
        self.model = tf.keras.Sequential(
            layers=[
                InputLayer((img_height, img_width, n_channels)),
                Conv2D(dim / 2, kernel_size=3, padding="same", activation="relu"),
                BatchNormalization(),
                Conv2D(dim, kernel_size=3, padding="same", activation="tanh"),
                Flatten(),
                Dense(img_height * img_width),
                Dropout(dropout_p),
                Dense(dim * dim),
            ]
        )
        self.output_layer = Dense(n_classes)

    @tf.function
    def call(
        self, input_tensor: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Returns model outputs.
        """
        model_out = self.model(input_tensor, training=training)
        logits = self.output_layer(model_out)
        return model_out, logits


class CVAE(Model):
    """
    Convolutional variational autoencoder for noisy labels.
    """

    def __init__(
        self,
        *,
        n_classes: int = 10,
        dim: int = 16,
        img_height: int = 28,
        img_width: int = 28,
        n_channels: int = 1,
        dropout_p: float = 0.2,
        softplus_beta: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.softplus_beta = softplus_beta
        self.forward_encoder = CNN(
            n_classes=n_classes,
            dim=dim,
            img_height=img_height,
            img_width=img_width,
            n_channels=n_channels,
            dropout_p=dropout_p,
        )
        self.encoder = Dense(n_classes)
        self.decoder = Dense(n_classes)

    def encode(self, images: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """
        Encodes.
        """
        _, logits = self.forward_encoder(images)
        joined = tf.concat([logits, labels], axis=1)
        encoded = self.encoder(joined) + 1.0 + 1.0 / self.n_classes
        return softplus(encoded, self.softplus_beta)

    def reparametrize(self, encoded: tf.Tensor) -> tf.Tensor:
        """
        Reparametrizes.
        """
        noise = tf.random.uniform(shape=encoded.shape)
        return tf.exp(
            (tf.ones_like(encoded) / encoded) * tf.math.log(encoded * noise)
            + tf.math.lgamma(encoded)
        )

    def decode(self, inputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """
        Decodes.
        """
        _, logits = self.forward_encoder(inputs)
        joined = tf.concat([logits, labels], axis=1)
        return self.decoder(joined)

    @tf.function
    def call(self, images: tf.Tensor, preds: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Expects dataset images and predicted labels from a classifier.
        Returns generated labels and encoded preds.
        """
        encoded = self.encode(images, preds)
        z = self.reparametrize(encoded)
        generated = self.decode(images, z)
        return generated, encoded
