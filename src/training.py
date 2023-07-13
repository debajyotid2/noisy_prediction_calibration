"""
Functions in this module are for the training, validation and testing
of datasets in the algorithm utilizing the loss-based filtering of batches.
"""

import logging
from pathlib import Path
import tensorflow as tf


logging.getLogger(__name__)


def kl_divergence(alpha_prior: tf.Tensor, alpha_inferred: tf.Tensor):
    """
    KL divergence loss.
    """
    kld = (
        tf.math.lgamma(alpha_prior)
        - tf.math.lgamma(alpha_inferred)
        + (alpha_inferred - alpha_prior) * tf.math.digamma(alpha_inferred)
    )
    return tf.reduce_sum(kld, axis=1)


class TrainStepClassifier:
    """
    Training step tf function for classifier.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        loss_func: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
    ):
        self.model = model
        self.loss_func
        self.optimizer = optimizer

    @tf.function
    def __call__(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        """
        Training step.
        """
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch)
            loss = self.loss_func(y_batch, y_pred)
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss


class ValidationStepClassifier:
    """
    Class for tf function validation step for classifier.
    """

    def __init__(self, model: tf.keras.Model, loss_func: tf.keras.losses.Loss):
        self.model = model
        self.loss_func = loss_func

    @tf.function
    def __call__(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        """
        Validation step.
        """
        y_pred = self.model(x_batch)
        loss = self.loss_func(y_batch, y_pred)
        return loss


class TrainStepAE:
    """
    Training step tf function for autoencoder.
    """

    def __init__(
        self,
        autoencoder: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        kld_reg: float,
        prior_norm: float,
    ):
        self.ae = autoencoder
        self.optimizer = optimizer
        self.bce_with_logits = tf.keras.losses.BinaryCrossEntropy(
            reduction=tf.keras.losses.Reduction.SUM
        )
        self.kld_reg = kld_reg
        self.prior_norm = prior_norm

    @tf.function
    def __call__(
        self, x_batch: tf.Tensor, y_pred: tf.Tensor, y_prior: tf.Tensor
    ) -> tf.Tensor:
        """
        Training step.
        """
        with tf.GradientTape() as tape:
            y_gen, encoded_ypred = self.ae(x_batch, y_pred)
            rec_loss = self.bce_with_logits(y_pred, y_gen)
            alpha_prior = self.prior_norm * y_prior
            dist_loss = kl_divergence(alpha_prior, encoded_ypred)
            loss = rec_loss + self.kld_reg * dist_loss
        grad = tape.gradient(loss, self.ae.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.ae.trainable_variables))
        return loss


def train_model(
    *,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    loss_func: tf.keras.losses.Loss,
    num_epochs: int = 1,
    train_log_dir: Path,
):
    """
    Trains a variational autoencoder.
    """
    len_train_ds = len(list(train_ds.unbatch()))
    len_val_ds = len(list(val_ds.unbatch()))

    train_step = TrainStepClassifier(model, optimizer, loss_func)
    val_step = ValidationStepClassifier(model, loss_func)

    # Tensorboard logging
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

    for epoch in range(num_epochs):
        # training
        train_loss = 0.0
        for batch_id, (x_batch, y_batch) in enumerate(train_ds):
            loss = train_step(x_batch, y_batch)
            train_loss += loss * len(x_batch)
        train_loss /= len_train_ds

        # validation
        val_loss = 0.0
        for idx, (x_batch, y_batch) in enumerate(val_ds):
            loss = val_step(x_batch, y_batch)
            val_loss += loss
        val_loss /= float(len_val_ds)

        # Log metrics
        with train_summary_writer.as_default():
            tf.summary.scalar("train_loss", train_loss, step=epoch)
            tf.summary.scalar("val_loss", val_loss, step=epoch)

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}],"
            f" train loss: {train_loss:.4f},"
            f" validation loss: {val_loss:.4f},"
        )


def train_npc(
    autoencoder: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    train_npc_ds: tf.data.Dataset,
    train_log_dir: Path,
    num_epochs: int = 1,
    kld_reg: float = 1.0,
    prior_norm: float = 5.0,
):
    """
    Training algorithm for the noisy prediction calibration algorithm.
    """
    # Tensorboard logging
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

    train_step = TrainStepAE(autoencoder, optimizer, kld_reg, prior_norm)
    for epoch in range(num_epochs):
        train_loss = 0.0
        for idxs, x_batch, y_pred, y_prior in train_npc_ds:
            # Generate alpha prior
            loss = train_step(x_batch, y_pred, y_prior)
            train_loss += loss

        # Log metrics
        with train_summary_writer.as_default():
            tf.summary.scalar("train_loss", train_loss, step=epoch)

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}]," f" train loss: {train_loss:.4f},"
        )
