"""
Basic convolutional network for image classification.
"""

from typing import Any, Callable, TypeVar
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Dense,
    Flatten,
    Dropout,
    Layer,
    Reshape,
    InputLayer,
    BatchNormalization,
    LeakyReLU,
)


def softplus(x: tf.Tensor, beta: float) -> tf.Tensor:
    """
    softplus(x) = 1/beta * log(1+ exp(beta * x))
    """
    if beta == 0.0:
        return 0.0
    return 1.0 / beta * tf.math.log(1 + tf.exp(beta * x))


class ConvBlock(Layer):
    """
    Basic building block of a convolutional net.
    """

    def __init__(self, dim: int, lr_alpha: float = 0.01):
        super().__init__()
        self.layer = tf.keras.Sequential(
            layers=[
                Conv2D(dim, kernel_size=3, strides=1, padding="same"),
                BatchNormalization(),
                LeakyReLU(alpha=lr_alpha),
            ]
        )

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Call method.
        """
        return self.layer(inputs, training=training)


class CNN_CIFAR(Model):
    """
    CNN classifier for CIFAR dataset.
    """

    def __init__(
        self,
        *,
        n_classes: int = 10,
        img_height: int = 32,
        img_width: int = 32,
        n_channels: int = 3,
        dim: int = 128,
        lr_alpha: float = 0.01,
        n_blocks_p_dim: int = 3,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.dim = dim

        # Loss and accuracy trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_metric_noisy = tf.keras.metrics.Accuracy(name="accuracy_noisy")
        self.accuracy_metric_clean = tf.keras.metrics.Accuracy(name="accuracy_clean")

        layers = []

        layers.append(InputLayer(input_shape=(img_height, img_width, n_channels)))
        layers.append(ConvBlock(dim))
        for _ in range(n_blocks_p_dim - 1):
            layers.append(ConvBlock(dim))
        layers.append(ConvBlock(2 * dim))
        layers.extend([MaxPooling2D(pool_size=2, strides=2), Dropout(rate=dropout_p)])

        for _ in range(n_blocks_p_dim - 1):
            layers.append(ConvBlock(2 * dim))
        layers.append(ConvBlock(4 * dim))
        layers.extend([MaxPooling2D(pool_size=2, strides=2), Dropout(rate=dropout_p)])

        layers.append(ConvBlock(2 * dim))
        layers.append(ConvBlock(dim))

        self.convnet = tf.keras.Sequential(layers=layers)

        convnet_outdim = self._get_convnet_out_dims(img_height, img_width, n_channels)
        self.avgpool = AveragePooling2D(pool_size=convnet_outdim[2], padding="same")
        pool_outdim = self._get_avgpool_out_dims(img_height, img_width, n_channels)
        self.reshape = Reshape(
            (
                -1,
                pool_outdim[-1],
            )
        )
        self.classifier = Dense(n_classes)

        # Embedding dim
        self.emb_dim = pool_outdim[-1]

    def _get_convnet_out_dims(
        self, img_height: int, img_width: int, n_channels: int
    ) -> tf.TensorShape:
        """
        Get shape of output of convnet for deciding avg pool kernel
        size.
        """
        random_inputs = tf.random.normal(shape=(1, img_height, img_width, n_channels))
        conv_out = self.convnet(random_inputs)
        return conv_out.shape

    def _get_avgpool_out_dims(
        self, img_height: int, img_width: int, n_channels: int
    ) -> tf.TensorShape:
        """
        Get shape of output of avg pooling layer for deciding reshape
        size.
        """
        random_inputs = tf.random.normal(shape=(1, img_height, img_width, n_channels))
        pool_out = self.avgpool(self.convnet(random_inputs))
        return pool_out.shape

    @tf.function
    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Call method.
        """
        conv_out = self.convnet(inputs, training=training)
        pooled = self.reshape(self.avgpool(conv_out))
        return pooled, self.classifier(pooled)

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """
        Returns list of metrics.
        """
        return [
            self.loss_tracker,
            self.accuracy_metric_noisy,
            self.accuracy_metric_clean,
        ]

    @tf.function
    def train_step(
        self, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor], training: bool = True
    ) -> dict[str, tf.Tensor]:
        """
        Training step.
        """
        x_batch, y_batch_noisy, y_batch_clean = data

        with tf.GradientTape() as tape:
            _, logits = self(x_batch, training=training)
            loss = self.loss(y_batch_noisy, logits)
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        y_pred = tf.argmax(logits, axis=-1)
        y_noisy = tf.argmax(y_batch_noisy, axis=-1)

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_metric_noisy.update_state(y_pred, y_noisy)
        self.accuracy_metric_clean.update_state(y_pred, y_batch_clean)

        return dict(
            train_loss=self.loss_tracker.result(),
            train_acc_noisy=self.accuracy_metric_noisy.result(),
            train_acc_clean=self.accuracy_metric_clean.result(),
        )

    @tf.function
    def test_step(
        self, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """
        Validation step.
        """
        x_batch, y_batch_noisy, y_batch_clean = data
        _, logits = self(x_batch, training=False)
        y_pred = tf.argmax(logits, axis=-1)
        loss = self.loss(y_batch_noisy, logits)
        y_pred = tf.argmax(logits, axis=-1)
        y_noisy = tf.argmax(y_batch_noisy, axis=-1)

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_metric_noisy.update_state(y_pred, y_noisy)
        self.accuracy_metric_clean.update_state(y_pred, y_batch_clean)

        return dict(
            loss=self.loss_tracker.result(),
            acc_noisy=self.accuracy_metric_noisy.result(),
            acc_clean=self.accuracy_metric_clean.result(),
        )


class CNN_MNIST(Model):
    """
    Convolutional image classifier for MNIST and FMNIST.
    """

    def __init__(
        self,
        *,
        n_classes: int = 10,
        dim: int = 16,
        img_height: int = 28,
        img_width: int = 28,
        n_channels: int = 1,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.dim = dim

        # Loss and accuracy trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_metric_noisy = tf.keras.metrics.Accuracy(name="accuracy_noisy")
        self.accuracy_metric_clean = tf.keras.metrics.Accuracy(name="accuracy_clean")

        # Model
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

        # Embedding dim
        self.emb_dim = dim * dim

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

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """
        Returns list of metrics.
        """
        return [
            self.loss_tracker,
            self.accuracy_metric_noisy,
            self.accuracy_metric_clean,
        ]

    @tf.function
    def train_step(
        self, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor], training: bool = True
    ) -> dict[str, tf.Tensor]:
        """
        Training step.
        """
        x_batch, y_batch_noisy, y_batch_clean = data

        with tf.GradientTape() as tape:
            _, logits = self(x_batch, training=training)
            loss = self.loss(y_batch_noisy, logits)
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        y_pred = tf.argmax(logits, axis=-1)
        y_noisy = tf.argmax(y_batch_noisy, axis=-1)

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_metric_noisy.update_state(y_pred, y_noisy)
        self.accuracy_metric_clean.update_state(y_pred, y_batch_clean)

        return dict(
            train_loss=self.loss_tracker.result(),
            train_acc_noisy=self.accuracy_metric_noisy.result(),
            train_acc_clean=self.accuracy_metric_clean.result(),
        )

    @tf.function
    def test_step(
        self, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """
        Validation step.
        """
        x_batch, y_batch_noisy, y_batch_clean = data
        _, logits = self(x_batch, training=False)
        y_pred = tf.argmax(logits, axis=-1)
        loss = self.loss(y_batch_noisy, logits)
        y_pred = tf.argmax(logits, axis=-1)
        y_noisy = tf.argmax(y_batch_noisy, axis=-1)

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_metric_noisy.update_state(y_pred, y_noisy)
        self.accuracy_metric_clean.update_state(y_pred, y_batch_clean)

        return dict(
            loss=self.loss_tracker.result(),
            acc_noisy=self.accuracy_metric_noisy.result(),
            acc_clean=self.accuracy_metric_clean.result(),
        )


CNNS = {"cifar10": CNN_CIFAR, "mnist": CNN_MNIST, "fashion mnist": CNN_MNIST}


def load_cnn(dataset_name: str) -> TypeVar:
    """
    Loads the appropriate CNN class as per the dataset.
    """
    if dataset_name not in CNNS:
        raise ValueError(f"Invalid dataset {dataset_name}.")
    return CNNS[dataset_name]


class CVAE(Model):
    """
    Convolutional variational autoencoder for noisy labels.
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        n_classes: int = 10,
        dim: int = 16,
        img_height: int = 28,
        img_width: int = 28,
        n_channels: int = 1,
        dropout_p: float = 0.2,
        softplus_beta: float = 0.1,
        kld_reg: float = 1.0,
        prior_norm: float = 5.0,
    ):
        super().__init__()

        self.kld_reg = kld_reg
        self.prior_norm = prior_norm
        self.n_classes = n_classes
        self.softplus_beta = softplus_beta

        # loss function
        self.bce_with_logits = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        # loss trackers
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.dist_loss_tracker = tf.keras.metrics.Mean(name="dist_loss")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # models
        self.forward_encoder = load_cnn(dataset_name)(
            n_classes=n_classes,
            dim=dim,
            img_height=img_height,
            img_width=img_width,
            n_channels=n_channels,
            dropout_p=dropout_p,
        )
        self.encoder = tf.keras.Sequential(
            layers=[InputLayer((2 * n_classes,)), Dense(n_classes)]
        )
        self.decoder = tf.keras.Sequential(
            layers=[InputLayer((2 * n_classes,)), Dense(n_classes)]
        )

    def encode(self, images: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """
        Encodes.
        """
        _, logits = self.forward_encoder(images)
        joined = tf.concat([tf.squeeze(logits), tf.squeeze(labels)], axis=1)
        encoded = self.encoder(joined) + 1.0 + 1.0 / self.n_classes
        return softplus(encoded, self.softplus_beta)

    def reparametrize(self, encoded: tf.Tensor) -> tf.Tensor:
        """
        Reparametrizes.
        """
        noise = tf.random.uniform(shape=tf.shape(encoded))
        return tf.exp(
            (tf.ones_like(encoded) / encoded) * tf.math.log(encoded * noise)
            + tf.math.lgamma(encoded)
        )

    def decode(self, inputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """
        Decodes.
        """
        _, logits = self.forward_encoder(inputs)
        joined = tf.concat([tf.squeeze(logits), tf.squeeze(labels)], axis=1)
        return self.decoder(joined)

    @tf.function
    def call(
        self, data: tuple[tf.Tensor, tf.Tensor], training: bool = True
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Expects dataset images and predicted labels from a classifier.
        Returns generated labels and encoded preds.
        """
        images, preds = data
        encoded = self.encode(images, preds)
        z = self.reparametrize(encoded)
        generated = self.decode(images, z)
        return generated, encoded

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """
        Returns all metrics.
        """
        return [self.rec_loss_tracker, self.dist_loss_tracker, self.loss_tracker]

    def _kl_divergence(
        self, alpha_prior: tf.Tensor, alpha_inferred: tf.Tensor
    ) -> float:
        """
        KL divergence loss.
        """
        kld = (
            tf.math.lgamma(alpha_prior)
            - tf.math.lgamma(alpha_inferred)
            + (alpha_inferred - alpha_prior) * tf.math.digamma(alpha_inferred)
        )
        return tf.reduce_sum(kld, axis=1)

    def compute_loss(
        self,
        y_pred: tf.Tensor,
        y_gen: tf.Tensor,
        y_prior: tf.Tensor,
        alpha_inferred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute reconstruction and distribution losses.
        """
        rec_loss = tf.reduce_sum(self.bce_with_logits(y_pred, y_gen))
        alpha_prior = (
            self.prior_norm * y_prior
            + 1.0
            + 1.0 / tf.cast(tf.shape(y_gen)[-1], tf.float32)
        )
        dist_loss = self.kld_reg * tf.reduce_sum(
            self._kl_divergence(alpha_prior, alpha_inferred)
        )
        loss = rec_loss + dist_loss
        return loss, rec_loss, dist_loss

    @tf.function
    def train_step(
        self, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """
        Training step.
        """
        x_batch, y_pred, y_prior = data
        with tf.GradientTape() as tape:
            y_gen, encoded_ypred = self((x_batch, y_pred), training=True)
            loss, rec_loss, dist_loss = self.compute_loss(
                y_pred, y_gen, y_prior, encoded_ypred
            )
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        # Update metrics
        self.rec_loss_tracker.update_state(rec_loss)
        self.dist_loss_tracker.update_state(dist_loss)
        self.loss_tracker.update_state(loss)

        return dict(
            rec_loss=self.rec_loss_tracker.result(),
            dist_loss=self.dist_loss_tracker.result(),
            train_loss=self.loss_tracker.result(),
        )

    @tf.function
    def test_step(
        self, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """
        Evaluation step.
        """
        x_batch, y_pred, y_prior = data
        y_gen, encoded_ypred = self((x_batch, y_pred), training=True)
        loss, rec_loss, dist_loss = self.compute_loss(
            y_pred, y_gen, y_prior, encoded_ypred
        )

        # Update metrics
        self.rec_loss_tracker.update_state(rec_loss)
        self.dist_loss_tracker.update_state(dist_loss)
        self.loss_tracker.update_state(loss)

        return dict(
            rec_loss=self.rec_loss_tracker.result(),
            dist_loss=self.dist_loss_tracker.result(),
            loss=self.loss_tracker.result(),
        )


class ClassifierPostprocessorEnsemble(Model):
    """
    Final ensemble of classifier and post-processor that is used
    to generate predictions.
    """

    def __init__(
        self,
        classifier: tf.keras.Model,
        post_processor: Callable[Any, Any],
        n_classes: int = 10,
    ):
        super().__init__()
        self.classifier = classifier
        self.post_processor = post_processor
        self.n_classes = n_classes

    def call(self, inputs: Any, training: bool = False) -> tf.Tensor:
        """
        Call method.
        """
        x_batch, _, _ = inputs
        batch_size = tf.shape(x_batch)[0]

        # Calculate p(y_hat | x)
        _, logits = self.classifier(x_batch, training=training)
        probs = tf.nn.softmax(tf.squeeze(logits))

        # Calculate p(y| y_hat, x)
        ae_probs = tf.Variable(
            tf.zeros(shape=(batch_size, self.n_classes, self.n_classes))
        )
        for class_val in range(self.n_classes):
            labels = class_val * tf.ones(shape=(batch_size,), dtype=tf.int32)
            labels_one_hot = tf.one_hot(labels, self.n_classes)
            labels_one_hot = tf.cast(labels_one_hot, tf.float32)

            _, alpha_inferred = self.post_processor((x_batch, labels_one_hot))
            alpha_inferred -= 1.0
            ae_probs[:, :, class_val].assign(
                alpha_inferred
                / tf.reshape(tf.reduce_sum(alpha_inferred, axis=1), shape=(-1, 1))
            )

        # Reshape probs to match dimensions of ae_probs
        probs = tf.expand_dims(probs, axis=1)
        probs = tf.repeat(probs, self.n_classes, axis=1)

        # Generate final labels
        final_probs = ae_probs * probs
        pseudolabels = tf.argmax(tf.reduce_sum(final_probs, axis=2), axis=1)
        return pseudolabels
