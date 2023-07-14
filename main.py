"""
Variational autoencoder.
"""
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
    train_val_split,
    standardize,
    generate_noisy_labels,
    make_dataset,
    make_npc_dataset,
)
from src.model import CNN, CVAE
from src.prior import generate_prior
from src.training import train_classifier, train_npc

cs = ConfigStore.instance()
cs.store(name="args", node=Args)


@hydra.main(config_path="./hydra_conf", config_name="config", version_base="1.3")
def main(args: Args):
    np.random.seed(args.training.seed)

    x_train, y_train, _, _ = load_data(args.dataset.name)

    # preprocess data

    # ensure NHWC format
    x_train = x_train.reshape(
        -1, args.dataset.img_height, args.dataset.img_width, args.dataset.n_channels
    )
    x_train = x_train.astype(np.float32) / 255.0
    mean, std = get_mean_std(x_train)
    x_train = standardize(x_train, mean, std)

    # generate noisy labels
    y_train_noisy = generate_noisy_labels(0.20, y_train, NoiseType.SYMMETRIC)

    # Make datasets
    x_train, y_train, x_val, y_val = train_val_split(
        x_train, y_train_noisy, args.dataset.val_frac
    )
    train_ds = make_dataset(x_train, y_train, args.dataset.batch_size)
    val_ds = make_dataset(x_val, y_val, args.dataset.batch_size)

    # Tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.training.log_dir)

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
    )
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    clf_optimizer = tf.keras.optimizers.Adam(args.training.learning_rate)
    ae_optimizer = tf.keras.optimizers.Adam(args.training.learning_rate)

    # Train classifier on noisy labels
    y_pred = train_classifier(
        model=classifier,
        optimizer=clf_optimizer,
        loss_func=loss_func,
        train_ds=train_ds,
        val_ds=val_ds,
        num_epochs=args.training.num_epochs,
        train_log_dir=Path(args.training.log_dir) / "train",
    )

    del train_ds
    gc.collect()

    # Create dataset for prior generation
    train_pred_ds = make_dataset(x_train, y_pred)

    # Generate prior
    prior_labels, prior_probabilities = generate_prior(
        classifier=classifier,
        dataset=train_pred_ds,
        n_classes=args.dataset.n_classes,
        n_neighbors=args.npc.n_neighbors,
    )
    del train_pred_ds
    del classifier
    gc.collect()

    # Create dataset for NPC
    train_npc_ds = make_npc_dataset(
        x=x_train, y_pred=y_pred, y_prior=prior_labels, batch_size=args.dataset.batch_size
    )

    # Train NPC
    train_npc(
        autoencoder=autoencoder,
        optimizer=ae_optimizer,
        train_npc_ds=train_npc_ds,
        train_log_dir=Path(args.training.log_dir) / "npc_train",
        num_epochs=args.npc.num_epochs,
        kld_reg=args.npc.kld_reg,
        prior_norm=args.npc.prior_norm,
    )


if __name__ == "__main__":
    main()
