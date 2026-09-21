import random

import numpy as np
import pytest
import torch

from hateprototypes.core import (
    build_class_means,
    cosine_classify,
    l2_normalize,
    set_seed,
)


def test_set_seed_is_reproducible():
    set_seed(42)

    python_value_1 = random.random()
    numpy_value_1 = np.random.rand()
    torch_value_1 = torch.rand(1)

    set_seed(42)

    python_value_2 = random.random()
    numpy_value_2 = np.random.rand()
    torch_value_2 = torch.rand(1)

    assert python_value_1 == python_value_2
    assert numpy_value_1 == numpy_value_2
    assert torch.equal(torch_value_1, torch_value_2)


def test_l2_normalize_vector():
    vector = np.array([3.0, 4.0])

    result = l2_normalize(vector)

    np.testing.assert_allclose(
        result,
        np.array([0.6, 0.8]),
        atol=1e-7,
    )


def test_l2_normalize_matrix():
    matrix = np.array(
        [
            [3.0, 4.0],
            [5.0, 12.0],
        ]
    )

    result = l2_normalize(matrix, axis=1)

    norms = np.linalg.norm(result, axis=1)

    np.testing.assert_allclose(
        norms,
        np.ones(2),
        atol=1e-7,
    )


def test_l2_normalize_zero_vector():
    vector = np.zeros(3)

    result = l2_normalize(vector)

    np.testing.assert_allclose(
        result,
        np.zeros(3),
    )


def test_build_class_means():
    features = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )

    labels = [0, 0, 1, 1]

    prototypes = build_class_means(
        features,
        labels,
    )

    assert set(prototypes.keys()) == {0, 1}

    assert prototypes[0][0] > prototypes[0][1]
    assert prototypes[1][1] > prototypes[1][0]

    assert np.isclose(
        np.linalg.norm(prototypes[0]),
        1.0,
    )

    assert np.isclose(
        np.linalg.norm(prototypes[1]),
        1.0,
    )


def test_build_class_means_missing_class():
    features = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
        ]
    )

    labels = [0, 0]

    with pytest.raises(
        ValueError,
        match="class 1",
    ):
        build_class_means(
            features,
            labels,
        )


def test_build_class_means_rejects_wrong_shape():
    features = np.array(
        [1.0, 2.0, 3.0]
    )

    labels = [0, 1, 0]

    with pytest.raises(
        ValueError,
        match="Expected features",
    ):
        build_class_means(
            features,
            labels,
        )


def test_build_class_means_rejects_length_mismatch():
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    labels = [0]

    with pytest.raises(
        ValueError,
        match="must match",
    ):
        build_class_means(
            features,
            labels,
        )


def test_cosine_classify():
    prototype_0 = np.array(
        [1.0, 0.0]
    )

    prototype_1 = np.array(
        [0.0, 1.0]
    )

    features = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [10.0, 1.0],
            [1.0, 10.0],
        ]
    )

    predictions = cosine_classify(
        features,
        prototype_0,
        prototype_1,
    )

    assert predictions.tolist() == [
        0,
        1,
        0,
        1,
    ]


def test_cosine_classification_is_scale_invariant():
    prototype_0 = np.array([1.0, 0.0])
    prototype_1 = np.array([0.0, 1.0])

    features = np.array(
        [
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )

    predictions_1 = cosine_classify(
        features,
        prototype_0,
        prototype_1,
    )

    predictions_2 = cosine_classify(
        features * 100,
        prototype_0 * 5,
        prototype_1 * 10,
    )

    np.testing.assert_array_equal(
        predictions_1,
        predictions_2,
    )


def test_cosine_classify_rejects_wrong_shape():
    features = np.array(
        [1.0, 0.0]
    )

    with pytest.raises(
        ValueError,
        match="Expected features",
    ):
        cosine_classify(
            features,
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        )