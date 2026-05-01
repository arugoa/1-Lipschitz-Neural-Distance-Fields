"""Abstract base class for all encoders."""

from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseEncoder(ABC):
    """
    All encoders must implement `encode(imgs_np, device) -> np.ndarray`.
    Input:  imgs_np  (T, H, W, C) uint8 numpy array for one episode
    Output: (T, D)   float32 numpy array of latent features
    """

    @abstractmethod
    def encode(self, imgs_np: np.ndarray, device: str) -> np.ndarray:
        """Encode a single episode of images to a (T, D) feature array."""
        ...

    @abstractmethod
    def output_dim(self) -> int:
        """Return the feature dimensionality D."""
        ...
