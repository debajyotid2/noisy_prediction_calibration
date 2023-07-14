"""
Prior generation.
"""
import logging
from typing import Any
from time import perf_counter
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier


logging.getLogger(__name__)


def generate_prior(
    *,
    classifier: tf.keras.Model,
    dataset: tf.data.Dataset,
    n_classes: int,
    n_neighbors: int = 10,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Generates prior using KNN. Dataset must have images and predicted labels.
    (y hat)
    """
    start_time = perf_counter()

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")

    embeddings = []
    classes = []
    len_train = 0

    # Get classifier predictions
    for x_batch, labels in dataset:
        len_train += x_batch.shape[0]
        # Make class to idx hash map
        class_idx_dict = dict.fromkeys(list(range(n_classes)), list())
        for idx, label in enumerate(labels):
            label = tf.argmax(label, axis=-1)
            class_idx_dict[label.numpy()].append(idx)

        # Get embeddings and logits
        embedding, logits = classifier(x_batch)
        probs = tf.nn.softmax(logits)
        probs = probs.numpy()

        # Gather embeddings for given class
        for class_id in range(n_classes):
            if len(class_idx_dict[class_id]) == 0:
                continue
            class_idxs = np.asarray(class_idx_dict[class_id], dtype=np.int32)
            gathered = np.argsort(probs[class_idxs, class_id])
            embeddings.append(embedding[class_idxs[gathered[-1]]])
            classes.append(class_id)

    # Fit classifier embeddings to classes in KNN
    classes = np.asarray(classes, dtype=np.int32)
    embeddings = np.asarray(embeddings, dtype=np.int32)
    knn.fit(embeddings, classes)
    del embeddings

    # Generate predictions using KNN for classes
    # NOTE: New embeddings must be generated for prediction
    embeddings = np.zeros((len_train, int(classifier.dim**2.0)))
    for idx, (x_batch, _) in enumerate(dataset):
        img_emb, _ = classifier(x_batch)
        embeddings[
            idx * x_batch.shape[0] : (idx + 1) * x_batch.shape[0]
        ] = img_emb.numpy()

    output_labels = knn.predict(embeddings)
    output_probs = knn.predict_proba(embeddings)
    padded = np.zeros((output_probs.shape[0], n_classes))
    padded[:, knn.classes_] = output_probs

    logging.info(f"Prior generated in {(perf_counter() - start_time)/60:.2f} minutes.")

    return output_labels, padded
