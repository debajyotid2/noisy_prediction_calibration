"""
Variational autoencoder.
"""
import logging
import gc
from pathlib import Path
import hydra
import numpy as np
import tensorflow as tf
from hydra.core.config_store import ConfigStore

from args import Args
from src.dataset import (
    NoiseType,
    load_data,
    get_mean_std,
    train_val_split_idxs,
    standardize,
    generate_noisy_labels,
    make_dataset,
    make_npc_dataset,
)
from src.model import CNN, CVAE, ClassifierPostprocessorEnsemble
from src.prior import generate_prior

cs = ConfigStore.instance()
cs.store(name="args", node=Args)

logging.basicConfig(format="%(asctime)s-%(levelname)s: %(message)s",
                    level=logging.INFO)


@hydra.main(config_path="./hydra_conf", config_name="config", version_base="1.3")
def main(args: Args):
    np.random.seed(args.training.seed)

    x_train, y_train, x_test, y_test = load_data(args.dataset.name)

    # preprocess data

    # ensure NHWC format
    x_train = x_train.reshape(
        -1, args.dataset.img_height, args.dataset.img_width, args.dataset.n_channels
    )
    x_test = x_test.reshape(
        -1, args.dataset.img_height, args.dataset.img_width, args.dataset.n_channels
    )

    x_train = x_train.astype(np.float32) / 255.0
    mean, std = get_mean_std(x_train)
    x_train = standardize(x_train, mean, std)
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))

    x_test = x_test.astype(np.float32) / 255.0
    x_test = standardize(x_test, mean, std)
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

    # generate noisy labels
    y_train_noisy = generate_noisy_labels(0.20, y_train, NoiseType.SYMMETRIC)

    # Make datasets
    train_ds = make_dataset(x_train, y_train_noisy, y_train, args.dataset.batch_size)
    test_ds = make_dataset(x_test, y_test, y_test, args.dataset.batch_size)

    # Initialize classifier and autoencoder
    classifier = CNN(
        n_classes=args.dataset.n_classes,
        dim=args.training.model_dim,
        img_height=args.dataset.img_height,
        img_width=args.dataset.img_width,
        n_channels=args.dataset.n_channels,
    )
    autoencoder = CVAE(
        n_classes=args.dataset.n_classes,
        dim=args.training.model_dim,
        img_height=args.dataset.img_height,
        img_width=args.dataset.img_width,
        n_channels=args.dataset.n_channels,
        dropout_p=args.training.dropout_p,
        softplus_beta=args.npc.softplus_beta,
        kld_reg=args.npc.kld_reg,
        prior_norm=args.npc.prior_norm,
    )
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    clf_optimizer = tf.keras.optimizers.Adam(args.training.learning_rate)
    ae_optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.training.learning_rate, clipnorm=args.npc.clipnorm
    )

    # Train classifier on noisy labels
    classifier.compile(loss=loss_func, optimizer=clf_optimizer, weighted_metrics=[])

    tb_callback_clf = tf.keras.callbacks.TensorBoard(
        log_dir=Path(args.training.log_dir) / "train", histogram_freq=1
    )
    classifier.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.training.num_epochs,
        callbacks=[tb_callback_clf],
    )

    # Gather predictions after model training for NPC dataset
    preds = []
    for x_batch, _, _ in train_ds:
        _, logits = classifier(x_batch)
        y_pred = tf.argmax(logits, axis=-1)
        preds.append(y_pred.numpy())
    y_pred = np.hstack(preds)

    del train_ds
    gc.collect()

    # Create dataset for prior generation
    train_pred_ds = make_dataset(x_train, y_pred, y_pred, args.dataset.batch_size)

    # Generate prior
    prior_labels, prior_probabilities = generate_prior(
        classifier=classifier,
        dataset=train_pred_ds,
        n_classes=args.dataset.n_classes,
        n_neighbors=args.npc.n_neighbors,
    )
    del train_pred_ds
    gc.collect()

    # Create dataset for NPC
    train_idxs, val_idxs = train_val_split_idxs(
        prior_labels.shape[0], args.dataset.val_frac
    )

    train_npc_ds = make_npc_dataset(
        x=x_train[train_idxs],
        y_pred=y_pred[train_idxs],
        y_prior=prior_labels[train_idxs],
        batch_size=args.dataset.batch_size,
    )
    val_npc_ds = make_npc_dataset(
        x=x_train[val_idxs],
        y_pred=y_pred[val_idxs],
        y_prior=prior_labels[val_idxs],
        batch_size=args.dataset.batch_size,
    )

    # Train NPC
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=Path(args.training.log_dir) / "train_npc",
        histogram_freq=1,
    )
    autoencoder.compile(optimizer=ae_optimizer, weighted_metrics=[])
    autoencoder.fit(
        train_npc_ds,
        validation_data=val_npc_ds,
        epochs=args.npc.num_epochs,
        callbacks=[tb_callback],
    )

    # Evaluate final ensemble
    final_accuracy = 0.0
    ensemble = ClassifierPostprocessorEnsemble(
        classifier, autoencoder, args.dataset.n_classes
    )
    for x_batch, _, y_label in test_ds:
        preds = ensemble((x_batch, y_label, y_label))
        preds = tf.cast(preds, tf.float32)
        final_accuracy += tf.reduce_sum(tf.where(preds == y_label, 1.0, 0.0))
    logging.info(f"Accuracy = {final_accuracy / x_test.shape[0] * 100.0:.2f} %")


if __name__ == "__main__":
    main()
