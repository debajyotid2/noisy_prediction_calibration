"""
Functions for preprocessing and transforming image datasets.
"""
import random
import logging
from pathlib import Path
from typing import Any
import numpy as np
import tensorflow as tf

logging.getLogger(__name__)

# Dataset loaders
DATASETS = {
    "mnist": tf.keras.datasets.mnist,
    "cifar10": tf.keras.datasets.cifar10,
    "fashion mnist": tf.keras.datasets.fashion_mnist,
}


def get_mean_std(
    images: tf.Tensor,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Calculate mean and standard deviation from a batch of
    image tensors for normalization.
    """
    mean = ()
    std = ()
    for i in range(images.shape[-1]):
        mean += (np.mean(images[:, :, :, i]),)
        std += (np.std(images[:, :, :, i]),)
    return mean, std


def standardize(
    images: tf.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]
) -> tf.Tensor:
    """
    Standardize a batch of images according to the given
    mean and standard deviation.

    normalized = (image - mean) / standard dev
    """
    for i in range(images.shape[-1]):
        images[:, :, :, i] = (images[:, :, :, i] - mean[i]) / std[i]
    return images


def train_val_split(
    x_train: np.ndarray[Any, Any], y_train: np.ndarray[Any, Any], val_frac: float = 0.2
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    """
    Splits the data into training and validation sets according to the
    supplied fraction of data for validation (val_frac).
    """
    perm_idx = np.random.permutation(x_train.shape[0])
    val_idx = perm_idx[: int(val_frac * x_train.shape[0])]
    train_idx = perm_idx[int(val_frac * x_train.shape[0]) :]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    return x_train, y_train, x_val, y_val


def make_dataset(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    batch_size: int = 32,
) -> tf.data.Dataset:
    """
    Makes a Tensorflow batched dataset from numpy images and labels.
    Discards additional samples such that the dataset length is an
    exact multiple of the batch size.
    """
    num_samples = int(len(x) / batch_size) * batch_size
    logging.debug(f"{len(x) - num_samples} samples discarded.")

    dataset = tf.data.Dataset.from_tensor_slices((x[:num_samples], y[:num_samples]))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def load_data(
    name: str,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    """
    Loads the data from tf.keras.datasets given dataset name.
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset {name} not loadable.")
    loader = DATASETS[name]
    (x_train, y_train), (x_test, y_test) = loader.load_data()
    return x_train, y_train, x_test, y_test
