"""
Functions for preprocessing and transforming image datasets.
"""
from enum import Enum, auto
import logging
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

# Class transition matrix for asymmetric noise
TRANSITION = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}


class NoiseType(Enum):
    """
    Class for different noisy types.
    """

    SYMMETRIC = auto()
    ASYMMETRIC = auto()


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


def convert_to_one_hot(
    image: tf.Tensor, noisy_label: tf.Tensor, clean_label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Convert categorical label to one-hot format.
    """
    noisy_label = tf.one_hot(noisy_label, 10)
    return image, tf.cast(noisy_label, tf.float32), tf.cast(clean_label, tf.float32)


def convert_to_one_hot_with_prior(
    image: tf.Tensor, pred: tf.Tensor, prior: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Convert categorical label to one-hot format for predictions and priors.
    """
    pred = tf.one_hot(pred, 10)
    prior = tf.one_hot(prior, 10)
    return image, tf.cast(pred, tf.float32), tf.cast(prior, tf.float32)


def generate_noisy_labels(
    noise_rate: float,
    gt_labels: np.ndarray[Any, Any],
    noise_mode: NoiseType = NoiseType.SYMMETRIC,
    transition: dict[int, int] = TRANSITION,
) -> np.ndarray[Any, Any]:
    """
    Creates noisy labels by randomly sampling from ground truth labels
    according to the type of noise and noise rate (% of noise).
    """
    rng = np.random.default_rng()

    noisy_labels = np.copy(gt_labels)
    idxs = np.arange(gt_labels.shape[0])
    rng.shuffle(idxs)

    noisy_label_idxs = idxs[: int(noise_rate * len(gt_labels))]

    if noise_mode == NoiseType.SYMMETRIC:
        noisy_labels[noisy_label_idxs] = np.random.randint(
            0, 10, size=noisy_label_idxs.shape
        )
    else:
        for idx in noisy_label_idxs:
            noisy_labels[idx] = transition[gt_labels[idx].item()]
    return noisy_labels


def train_val_split_idxs(
    num_datapoints: int, val_frac: float = 0.2
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Generates indices to split data into training and validation sets according
    to the supplied fraction of data for validation (val_frac).
    """
    perm_idx = np.random.permutation(num_datapoints)
    val_idxs = perm_idx[: int(val_frac * num_datapoints)]
    train_idxs = perm_idx[int(val_frac * num_datapoints) :]
    return train_idxs, val_idxs


def make_dataset(
    x: np.ndarray[Any, Any],
    y_noisy: np.ndarray[Any, Any],
    y_clean: np.ndarray[Any, Any],
    batch_size: int = 32,
) -> tf.data.Dataset:
    """
    Makes a Tensorflow batched dataset from numpy images and labels.
    Discards additional samples such that the dataset length is an
    exact multiple of the batch size.
    """
    num_samples = int(len(x) / batch_size) * batch_size
    logging.debug(f"{len(x) - num_samples} samples discarded.")

    dataset = tf.data.Dataset.from_tensor_slices(
        (x[:num_samples], y_noisy[:num_samples], y_clean[:num_samples])
    ).map(convert_to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def make_npc_dataset(
    *,
    x: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    y_prior: np.ndarray[Any, Any],
    batch_size: int = 32,
) -> tf.data.Dataset:
    """
    Makes a Tensorflow batched dataset from numpy images, predicted labels and
    prior labels.  Discards additional samples such that the dataset length is
    an exact multiple of the batch size.
    """
    num_samples = int(len(x) / batch_size) * batch_size
    logging.debug(f"{len(x) - num_samples} samples discarded.")

    dataset = tf.data.Dataset.from_tensor_slices(
        (x[:num_samples], y_pred[:num_samples], y_prior[:num_samples])
    ).map(convert_to_one_hot_with_prior, num_parallel_calls=tf.data.AUTOTUNE)
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
