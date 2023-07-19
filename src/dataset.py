"""
Functions for preprocessing and transforming image datasets.
"""
from enum import Enum, auto
import logging
from typing import Any
from pathlib import Path
from math import inf
import numpy as np
import tensorflow as tf
from scipy.stats import truncnorm
from . import npz_ops

logging.getLogger(__name__)

# Dataset loaders
DATASETS = {
    "mnist": tf.keras.datasets.mnist,
    "cifar10": tf.keras.datasets.cifar10,
    "fashion mnist": tf.keras.datasets.fashion_mnist,
}

# Class transition matrix for asymmetric noise
TRANSITION_MNIST = {0: 0, 1: 1, 2: 7, 3: 8, 4: 4, 5: 6, 6: 5, 7: 7, 8: 8, 9: 9}
TRANSITION_FMNIST = {0: 6, 1: 1, 2: 4, 3: 3, 4: 4, 5: 7, 6: 6, 7: 7, 8: 8, 9: 9}
TRANSITION_CIFAR10 = {0: 0, 1: 1, 2: 0, 3: 5, 4: 7, 5: 3, 6: 6, 7: 7, 8: 8, 9: 1}

_TRANSITIONS = {
    "mnist": TRANSITION_MNIST,
    "cifar10": TRANSITION_CIFAR10,
    "fashion mnist": TRANSITION_FMNIST,
}


class NoiseType(Enum):
    """
    Class for different noisy types.
    """

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    INSTANCE_DEPENDENT = "idn"
    SIMILARITY_REFLECTED = "sridn"


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


def _generate_symmetric_noise(
    x: np.ndarray[Any, Any],
    y_gt: np.ndarray[Any, Any],
    noise_rate: float,
    noisy_label_idxs: np.ndarray[Any, Any],
    sridn_cls_prob_path: Path,
    transition: dict[int, int] = TRANSITION_MNIST,
    num_classes: int = 10,
) -> np.ndarray[Any, Any]:
    """
    Generates symmetric noise.
    """
    noisy_labels = np.copy(y_gt)
    noisy_labels[noisy_label_idxs] = np.random.randint(
        0, num_classes, size=noisy_label_idxs.shape
    )
    return noisy_labels


def _generate_asymmetric_noise(
    x: np.ndarray[Any, Any],
    y_gt: np.ndarray[Any, Any],
    noise_rate: float,
    noisy_label_idxs: np.ndarray[Any, Any],
    sridn_cls_prob_path: Path,
    transition: dict[int, int] = TRANSITION_MNIST,
    num_classes: int = 10,
) -> np.ndarray[Any, Any]:
    """
    Generates asymmetric noise.
    """
    noisy_labels = np.copy(y_gt)
    for idx in noisy_label_idxs:
        noisy_labels[idx] = transition[y_gt[idx].item()]
    return noisy_labels


def _generate_instance_dependent_noise(
    x: np.ndarray[Any, Any],
    y_gt: np.ndarray[Any, Any],
    noise_rate: float,
    noisy_label_idxs: np.ndarray[Any, Any],
    sridn_cls_prob_path: Path,
    transition: dict[int, int] = TRANSITION_MNIST,
    num_classes: int = 10,
) -> np.ndarray[Any, Any]:
    """
    Generates instance dependent noise.
    """
    rng = np.random.default_rng()

    images = np.copy(x)
    noisy_labels = np.copy(y_gt)
    flip_rates = truncnorm.rvs(
        -noise_rate / 0.1,
        (1.0 - noise_rate) / 0.1,
        loc=noise_rate,
        scale=0.1,
        size=(len(y_gt),),
    )
    dim_weights = rng.normal(
        size=(num_classes, images.shape[1] * images.shape[2], num_classes)
    )
    for i in range(y_gt.shape[0]):
        if i not in noisy_label_idxs:
            continue
        p = images[i].reshape(1, -1) @ dim_weights[y_gt[i]]
        p = np.squeeze(p, axis=0)

        p[y_gt[i]] = -inf
        p = flip_rates[i] * np.exp(p) / np.sum(np.exp(p))
        p[y_gt[i]] += 1 - flip_rates[i]
        noisy_labels[i] = np.argmax(np.random.multinomial(1, p))
    return noisy_labels


def _generate_sr_instance_dependent_noise(
    x: np.ndarray[Any, Any],
    y_gt: np.ndarray[Any, Any],
    noise_rate: float,
    noisy_label_idxs: np.ndarray[Any, Any],
    sridn_cls_prob_path: Path,
    transition: dict[int, int] = TRANSITION_MNIST,
    num_classes: int = 10,
) -> np.ndarray[Any, Any]:
    """
    Generates similarity reflected instance dependent noise.
    """

    noisy_labels = np.copy(y_gt)
    if not sridn_cls_prob_path.exists():
        raise RuntimeError(f"{sridn_cls_prob_path.resolve()} does not exist.")
    probs_classifier = npz_ops.load_from_npz(sridn_cls_prob_path)
    cls_pred_idxs = np.argsort(np.max(probs_classifier, axis=-1))
    noisy_label_idxs = cls_pred_idxs[: int(noise_rate * len(cls_pred_idxs))]
    noisy_labels[noisy_label_idxs] = np.argmax(probs_classifier[noisy_label_idxs])
    return noisy_labels


"""
Noisy label generator dictionary.
"""
_NOISY_LABEL_GEN = {
    NoiseType.SYMMETRIC: _generate_symmetric_noise,
    NoiseType.ASYMMETRIC: _generate_asymmetric_noise,
    NoiseType.INSTANCE_DEPENDENT: _generate_instance_dependent_noise,
    NoiseType.SIMILARITY_REFLECTED: _generate_sr_instance_dependent_noise,
}


def generate_noisy_labels(
    *,
    noise_rate: float,
    x: np.ndarray[Any, Any],
    y_gt: np.ndarray[Any, Any],
    cache_path: Path,
    noise_mode: NoiseType = NoiseType.SYMMETRIC,
    dataset_name: str = "mnist",
    num_classes: int = 10,
) -> np.ndarray[Any, Any]:
    """
    Creates noisy labels by randomly sampling from ground truth labels
    according to the type of noise and noise rate (% of noise).
    """
    rng = np.random.default_rng()

    idxs = np.arange(y_gt.shape[0])
    rng.shuffle(idxs)

    noisy_label_idxs = idxs[: int(noise_rate * len(y_gt))]

    noisy_lbl_path = cache_path / f"{dataset_name}-{noise_mode.value}-noisy.npz"
    if noisy_lbl_path.exists():
        noisy_labels = npz_ops.load_from_npz(noisy_lbl_path)
    else:
        noisy_labels = _NOISY_LABEL_GEN[noise_mode](
            x,
            y_gt,
            noise_rate,
            noisy_label_idxs,
            cache_path / "sridn_cls_probs.npz",
            _TRANSITIONS[dataset_name],
            num_classes,
        )
        cache_path.mkdir(exist_ok=True)
        npz_ops.compress_to_npz(noisy_labels, noisy_lbl_path)

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
