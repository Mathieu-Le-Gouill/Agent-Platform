from __future__ import annotations

import numpy as np

__all__ = ["softmax"]


def softmax(x: np.ndarray) -> np.ndarray:
    # Subtract row max before exponentiating to avoid overflow.
    e_x = np.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
