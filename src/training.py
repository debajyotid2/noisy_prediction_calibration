"""
Functions in this module are for the training, validation and testing
of datasets in the algorithm utilizing the loss-based filtering of batches.
"""

import logging
from typing import Any
from pathlib import Path
import numpy as np
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


class ClassifierTrainStep:
    """
    Training step tf function for classifier.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_func: tf.keras.losses.Loss,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func

    @tf.function
    def __call__(
        self, x_batch: tf.Tensor, y_batch: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Training step.
        """
        with tf.GradientTape() as tape:
            _, logits = self.model(x_batch)
            loss = self.loss_func(y_batch, logits)
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        y_pred = tf.argmax(tf.nn.softmax(logits), axis=-1)
        y_batch = tf.argmax(y_batch, axis=-1)
        correct = tf.where(y_pred == y_batch, 1.0, 0.0)
        return loss, tf.reduce_sum(correct)


class ClassifierValidationStep:
    """
    Class for tf function validation step for classifier.
    """

    def __init__(self, model: tf.keras.Model, loss_func: tf.keras.losses.Loss):
        self.model = model
        self.loss_func = loss_func

    @tf.function
    def __call__(
        self, x_batch: tf.Tensor, y_batch: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Validation step.
        """
        _, logits = self.model(x_batch)
        y_pred = tf.argmax(tf.nn.softmax(logits), axis=-1)
        loss = self.loss_func(y_batch, logits)
        y_pred = tf.argmax(tf.nn.softmax(logits), axis=-1)
        y_batch = tf.argmax(y_batch, axis=-1)
        correct = tf.where(y_pred == y_batch, 1.0, 0.0)
        return loss, tf.reduce_sum(correct)


class AETrainStep:
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
        self.bce_with_logits = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self.kld_reg = kld_reg
        self.prior_norm = prior_norm

    @tf.function
    def __call__(
        self, x_batch: tf.Tensor, y_pred: tf.Tensor, y_prior: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Training step.
        """
        with tf.GradientTape() as tape:
            y_gen, encoded_ypred = self.ae(x_batch, y_pred)
            rec_loss = tf.reduce_mean(self.bce_with_logits(y_pred, y_gen))
            alpha_prior = self.prior_norm * y_prior + 1.0 + 1.0 / y_gen.shape[-1]
            dist_loss = self.kld_reg * tf.reduce_mean(
                kl_divergence(alpha_prior, encoded_ypred)
            )
            loss = rec_loss + dist_loss
        grad = tape.gradient(loss, self.ae.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.ae.trainable_variables))
        return loss, rec_loss, dist_loss


def train_classifier(
    *,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    loss_func: tf.keras.losses.Loss,
    num_epochs: int = 1,
    train_log_dir: Path,
) -> np.ndarray[Any, Any]:
    """
    Trains a variational autoencoder.
    """
    len_train_ds = len(list(train_ds.unbatch()))
    len_val_ds = len(list(val_ds.unbatch()))

    train_step = ClassifierTrainStep(model, optimizer, loss_func)
    val_step = ClassifierValidationStep(model, loss_func)

    # Tensorboard logging
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

    for epoch in range(num_epochs):
        # training
        train_loss = 0.0
        train_acc = 0.0
        for batch_id, (x_batch, y_batch) in enumerate(train_ds):
            loss, correct = train_step(x_batch, y_batch)
            train_loss += loss * len(x_batch)
            train_acc += correct
        train_loss /= len_train_ds
        train_acc /= len_train_ds

        # validation
        val_loss = 0.0
        val_acc = 0.0
        for idx, (x_batch, y_batch) in enumerate(val_ds):
            loss, correct = val_step(x_batch, y_batch)
            val_loss += loss
            val_acc += correct
        val_loss /= len_val_ds
        val_acc /= len_val_ds

        # Log metrics
        with train_summary_writer.as_default():
            tf.summary.scalar("train_loss", train_loss, step=epoch)
            tf.summary.scalar("train_acc", train_acc, step=epoch)
            tf.summary.scalar("val_loss", val_loss, step=epoch)
            tf.summary.scalar("val_acc", val_acc, step=epoch)

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}],"
            f" train loss: {train_loss:.4f},"
            f" train accuracy: {train_acc * 100:.2f} %,"
            f" validation loss: {val_loss:.4f},"
            f" validation accuracy: {val_acc * 100:.2f} %"
        )

    # Gather predictions after model training for NPC dataset
    preds = []
    for x_batch, _ in train_ds:
        _, logits = model(x_batch)
        y_pred = tf.argmax(tf.nn.softmax(logits), axis=-1)
        preds.append(y_pred.numpy())
    return np.hstack(preds)


def train_npc(
    *,
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

    train_step = AETrainStep(autoencoder, optimizer, kld_reg, prior_norm)
    for epoch in range(num_epochs):
        train_loss = 0.0
        rec_loss_train = 0.0
        dist_loss_train = 0.0
        rec_losses = []
        dist_losses = []
        for x_batch, y_pred, y_prior in train_npc_ds:
            # Generate alpha prior
            loss, rec_loss, dist_loss = train_step(x_batch, y_pred, y_prior)
            train_loss += loss
            rec_loss_train += rec_loss
            dist_loss_train += dist_loss
            rec_losses.append(rec_loss)
            dist_losses.append(dist_loss)
        breakpoint()

        # Log metrics
        with train_summary_writer.as_default():
            tf.summary.scalar("train_loss", train_loss, step=epoch)
            tf.summary.scalar("rec_loss", rec_loss_train, step=epoch)
            tf.summary.scalar("dist_loss", dist_loss_train, step=epoch)

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}],"
            f" train loss: {train_loss:.4f},"
            f" reconstruction loss: {rec_loss_train:.4f},"
            f" distribution loss: {dist_loss_train:.4f},"
        )
