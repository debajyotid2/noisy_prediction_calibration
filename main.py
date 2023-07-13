"""
Variational autoencoder.
"""

from pathlib import Path
import hydra
import numpy as np
import tensorflow as tf
from hydra.core.config_store import ConfigStore

from args import Args
from src.dataset import (
    load_data,
    get_mean_std,
    train_val_split,
    standardize,
    make_dataset,
)
from src.model import CNN
from src.training import train_model

cs = ConfigStore.instance()
cs.store(name="args", node=Args)


@hydra.main(config_path="./hydra_conf", config_name="config", version_base="1.3")
def main(args: Args):
    x_train, y_train, x_test, y_test = load_data(args.dataset.name)
    
    # preprocess data

    # Make datasets
    x_train, y_train, x_val, y_val = train_val_split(
        x_train, y_train, args.dataset.val_frac
    )
    train_ds = make_dataset(x_train, y_train, args.dataset.batch_size)
    val_ds = make_dataset(x_val, y_val, args.dataset.batch_size)
    test_ds = make_dataset(x_test, y_test, args.dataset.batch_size)

    # Tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.training.log_dir)

    # Initialize model
    classifier = CNN(

    # Train

    train_model(
        model=model,
        optimizer=optimizer,
        train_ds=train_ds,
        val_ds=val_ds,
        num_epochs=args.training.num_epochs,
        train_log_dir=Path(args.training.log_dir) / "train",
    )

if __name__ == "__main__":
    main()
