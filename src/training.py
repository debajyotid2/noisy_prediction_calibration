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
