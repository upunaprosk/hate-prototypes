import numpy as np
import pandas as pd
import pytest

from hateprototypes.data import (
    normalize_label,
    normalize_labels,
    sample_binary_prototypes,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("hate", 1),
        ("HATE", 1),
        (" unsafe ", 1),
        ("implicit", 1),
        ("implicit_hate", 1),
        ("implicit-hate", 1),
        ("safe", 0),
        ("SAFE", 0),
        ("non-hate", 0),
        ("nonhate", 0),
        ("non_hate", 0),
        ("neutral", 0),
        ("0", 0),
        ("1", 1),
        (0, 0),
        (1, 1),
        (np.int64(0), 0),
        (np.int64(1), 1),
    ],
)
def test_normalize_label(value, expected):
    assert normalize_label(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "unknown",
        "hate-ish",
        "",
        2,
        -1,
        None,
    ],
)
def test_normalize_label_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        normalize_label(value)


def test_normalize_labels_series():
    labels = pd.Series(
        [
            "hate",
            "safe",
            "1",
            "0",
        ]
    )

    result = normalize_labels(labels)

    assert result.tolist() == [
        1,
        0,
        1,
        0,
    ]

    assert result.dtype.kind in {"i", "u"}


def test_sample_binary_prototypes():
    df = pd.DataFrame(
        {
            "text": [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
            ],
            "label": [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
            ],
        }
    )

    result = sample_binary_prototypes(
        df,
        n_per_class=2,
        seed=42,
    )

    assert len(result) == 4

    counts = result["label"].value_counts()

    assert counts[0] == 2
    assert counts[1] == 2


def test_sample_binary_prototypes_is_reproducible():
    df = pd.DataFrame(
        {
            "text": [f"sample-{i}" for i in range(20)],
            "label": ([0] * 10 + [1] * 10),
        }
    )

    first = sample_binary_prototypes(
        df,
        n_per_class=3,
        seed=123,
    )

    second = sample_binary_prototypes(
        df,
        n_per_class=3,
        seed=123,
    )

    assert first["text"].tolist() == second["text"].tolist()


def test_sample_binary_prototypes_uses_available_examples():
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "label": [0, 1, 1],
        }
    )

    result = sample_binary_prototypes(
        df,
        n_per_class=10,
        seed=42,
    )

    counts = result["label"].value_counts()

    assert counts[0] == 1
    assert counts[1] == 2


def test_sample_binary_prototypes_rejects_missing_class():
    df = pd.DataFrame(
        {
            "text": ["a", "b"],
            "label": [0, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="class 1",
    ):
        sample_binary_prototypes(
            df,
            n_per_class=1,
            seed=42,
        )
