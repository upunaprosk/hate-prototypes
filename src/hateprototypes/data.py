from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


LABEL_MAPPING = {
    "hate": 1,
    "unsafe": 1,
    "implicit": 1,
    "implicit_hate": 1,
    "implicit-hate": 1,
    "non-hate": 0,
    "nonhate": 0,
    "non_hate": 0,
    "neutral": 0,
    "safe": 0,
}


def normalize_label(value) -> int:
    """Convert supported binary hate-speech labels to 0 or 1."""
    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in LABEL_MAPPING:
            return LABEL_MAPPING[normalized]

        try:
            value = int(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Unrecognized label: {value!r}"
            ) from exc

    if isinstance(value, (int, np.integer)):
        value = int(value)

        if value in (0, 1):
            return value

    raise ValueError(
        f"Expected a binary label (0/1 or supported string), "
        f"got {value!r}."
    )


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series of labels to0/1"""
    return series.apply(normalize_label).astype(int)


class TextDataset(Dataset):
    """ tokenized text classification dataset"""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if len(self.texts) != len(self.labels):
            raise ValueError(
                "texts and labels must have the same length."
            )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }

        item["labels"] = torch.tensor(
            int(self.labels[index]),
            dtype=torch.long,
        )

        return item


def make_loader(
    texts: Sequence[str],
    labels: Sequence[int],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader for text classification."""
    dataset = TextDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def load_csv(
    train_pattern: str,
    test_pattern: str,
    dataset: str,
    text_col: str = "sentence",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test CSV files:
        "{ds}_train.csv"
        "{ds}_test.csv"
    """
    train_path = Path(train_pattern.format(ds=dataset))
    test_path = Path(test_pattern.format(ds=dataset))

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    for frame in (train, test):
        missing = {
            column
            for column in (text_col, label_col)
            if column not in frame.columns
        }

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        frame.dropna(
            subset=[text_col, label_col],
            inplace=True,
        )

        frame["text"] = frame[text_col].astype(str)
        frame["label"] = normalize_labels(frame[label_col])

    return train, test

def sample_binary_prototypes(
    df: pd.DataFrame,
    n_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Sample up to n_per_class examples from each binary class."""
    samples = []

    for class_id in (0, 1):
        class_df = df[df["label"] == class_id]

        if class_df.empty:
            raise ValueError(
                f"Cannot sample prototypes: class {class_id} is empty."
            )

        samples.append(
            class_df.sample(
                n=min(n_per_class, len(class_df)),
                random_state=seed,
            )
        )

    return pd.concat(samples, ignore_index=True)