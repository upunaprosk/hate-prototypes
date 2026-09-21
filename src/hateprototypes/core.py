import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(
    x: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """L2-normalize an array along the given axis."""
    x = np.asarray(x)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def build_class_means(
    features: np.ndarray,
    labels: Iterable[int],
) -> dict[int, np.ndarray]:
    """
    Build one normalized prototype per class.

    Each example representation is L2-normalized first.
    The class prototype is the mean representation,
    normalized again to unit length.
    """
    features = np.asarray(features)
    labels = np.asarray(list(labels))

    if features.ndim != 2:
        raise ValueError(
            f"Expected features with shape (n_samples, hidden_size), "
            f"got {features.shape}."
        )

    if len(features) != len(labels):
        raise ValueError(
            "Number of feature vectors and labels must match."
        )

    prototypes: dict[int, np.ndarray] = {}

    for class_id in (0, 1):
        class_features = features[labels == class_id]

        if len(class_features) == 0:
            raise ValueError(
                f"Cannot build prototype for class {class_id}: "
                "no examples were provided."
            )

        class_features = l2_normalize(class_features, axis=1)
        mean = class_features.mean(axis=0)
        prototypes[class_id] = l2_normalize(mean)

    return prototypes


def cosine_classify(
    features: np.ndarray,
    prototype_0: np.ndarray,
    prototype_1: np.ndarray,
) -> np.ndarray:
    """
    Classify representations by cosine similarity to two prototypes.

    Returns
    -------
    np.ndarray
        Predicted binary labels with shape n_samples
    """
    features = np.asarray(features)

    if features.ndim != 2:
        raise ValueError(
            f"Expected features with shape (n_samples, hidden_size), "
            f"got {features.shape}."
        )

    features = l2_normalize(features, axis=1)
    prototype_0 = l2_normalize(np.asarray(prototype_0))
    prototype_1 = l2_normalize(np.asarray(prototype_1))

    scores = np.stack(
        [
            features @ prototype_0,
            features @ prototype_1,
        ],
        axis=1,
    )

    return scores.argmax(axis=1)