"""
Prior generation.
"""
import logging
from typing import Any
from pathlib import Path
from time import perf_counter
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from . import npz_ops

logging.getLogger(__name__)


def generate_prior(
    *,
    classifier: tf.keras.Model,
    dataset: tf.data.Dataset,
    n_classes: int,
    n_neighbors: int = 10,
    dataset_name: str,
    noise_mode: str,
    noise_rate: float,
    cache_dir: Path
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Generates prior using KNN. Dataset must have images and predicted labels.
    (y hat). Ensure that the dataset has batch size 1.
    """
    prob_cache_path = cache_dir / f"knn_probs-{dataset_name}-{noise_mode}-{noise_rate}.npz"
    
    # Load from saved cache if it exists
    if prob_cache_path.exists():
        padded = npz_ops.load_from_npz(prob_cache_path)
        output_labels = np.argmax(padded, axis=-1)
        return output_labels, padded

    start_time = perf_counter()

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")

    embeddings = []
    classes = []
    len_train = 0

    # Get classifier predictions
    for x_batch, y_preds, _ in dataset:
        len_train += x_batch.shape[0]
        # Make class to idx hash map
        class_idx_dict = dict.fromkeys(list(range(n_classes)), list())
        for idx, y_pred in enumerate(y_preds):
            label = tf.argmax(y_pred, axis=-1)
            class_idx_dict[label.numpy().item()].append(idx)

        # Get embeddings and logits
        embedding, logits = classifier(x_batch)
        probs = tf.nn.softmax(logits)
        probs = probs.numpy()
        probs = np.squeeze(probs)

        # Gather embeddings for given class
        for class_id in range(n_classes):
            if len(class_idx_dict[class_id]) == 0:
                continue
            class_idxs = np.asarray(class_idx_dict[class_id], dtype=np.int32)
            gathered = np.argsort(probs[class_idxs, class_id])
            embeddings.append(embedding.numpy()[class_idxs[gathered[-1]]])
            classes.append(class_id)

    logging.info(f"Initial embeddings generated in {(perf_counter() - start_time)/60:.2f} minutes.")

    # Fit classifier embeddings to classes in KNN

    start_time = perf_counter()
    classes = np.asarray(classes, dtype=np.int32)
    embeddings = np.asarray(embeddings, dtype=np.int32)
    embeddings = np.squeeze(embeddings)
    knn.fit(embeddings, classes)
    del embeddings

    logging.info(f"KNN trained in {(perf_counter() - start_time)/60:.2f} minutes.")
    # Generate predictions using KNN for classes
    # NOTE: New embeddings must be generated for prediction
    start_time = perf_counter()
    embeddings = np.zeros((len_train, int(classifier.emb_dim)))
    for idx, (x_batch, _, _) in enumerate(dataset):
        img_emb, _ = classifier(x_batch)
        embeddings[
            idx * x_batch.shape[0] : (idx + 1) * x_batch.shape[0]
        ] = np.squeeze(img_emb.numpy())

    logging.info(f"New embeddings generated in {(perf_counter() - start_time)/60:.2f} minutes.")
    
    start_time = perf_counter()
    output_probs = knn.predict_proba(embeddings)
    padded = np.zeros((output_probs.shape[0], n_classes))
    padded[:, knn.classes_] = output_probs
    output_labels = np.argmax(padded, axis=-1)

    logging.info(f"Prior generated in {(perf_counter() - start_time)/60:.2f} minutes.")

    npz_ops.compress_to_npz(padded, prob_cache_path)

    return output_labels, padded
