"""
Functions in this module are for the training, validation and testing
of datasets in the algorithm utilizing the loss-based filtering of batches.
"""

import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .helpers import plot_images


logging.getLogger(__name__)


class TrainStep:
    """
    Training step tf function.
    """

    def __init__(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer):
        self.model = model
        self.optimizer = optimizer
    
    @tf.function
    def __call__(
        self,
        x_batch: tf.Tensor,
    ) -> tf.Tensor:
        """
        Training step.
        """
        with tf.GradientTape() as tape:
            # model training logic
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss


class ValidationStep:
    """
    Class for tf function validation step.
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model

    @tf.function
    def __call__(self, x_batch: tf.Tensor) -> tf.Tensor:
        """
        Validation step.
        """
        # model validation logic
        return loss


def train_model(
    *,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_epochs: int = 1,
    train_log_dir: Path,
):
    """
    Trains a variational autoencoder.
    """
    len_train_ds = len(list(train_ds.unbatch()))
    len_val_ds = len(list(val_ds.unbatch()))

    train_step = TrainStep(model, optimizer)
    val_step = ValidationStep(model)

    # Tensorboard logging
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

    for epoch in range(num_epochs):
        # training
        train_loss = 0.0
        for batch_id, (x_batch, _) in enumerate(train_ds):
            loss = train_step(x_batch)
            train_loss += loss * len(x_batch)
        train_loss /= len_train_ds

        # validation
        val_loss = 0.0
        for idx, (x_batch, y_batch) in enumerate(val_ds):
            loss = val_step(x_batch)
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
