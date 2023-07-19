"""
Functions to compress data to npz and load from saved npz files.
"""
from pathlib import Path
from typing import Any

import logging

import numpy as np

logging.getLogger(__name__)

def compress_to_npz(data: np.ndarray[Any, Any], save_path: Path) -> None:
    """
    Compresses a numpy array to uncompressed npz format and saves to save_path.
    """
    np.savez(save_path, data=data)
    logging.info(f"Saved compressed array to {save_path}.")

def load_from_npz(npz_path: Path) -> np.ndarray[Any, Any]:
    """
    Load a compressed array from npz_path.
    """
    loaded_array = np.load(npz_path)
    logging.info(f"Loaded compressed array from {npz_path}.")
    return loaded_array["data"]


