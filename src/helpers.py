"""
Miscellaneous helper functions.
"""
from pathlib import Path
from typing import Any, Optional
import numpy as np
import matplotlib.pyplot as plt

def plot_images(data: np.ndarray[Any, Any], path: Optional[Path]):
    """
    Plots a 3x3 grid of images from data.
    """
    plt.figure(figsize=(12, 12))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(data[i], cmap="gray")
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
