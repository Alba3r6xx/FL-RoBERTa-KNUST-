from __future__ import annotations

import re
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_count(text: str) -> int:
    return len(text.split())


def preprocess_frame(frame: pd.DataFrame, text_col: str, label_col: str, min_tokens: int = 20, max_tokens: int = 512) -> pd.DataFrame:
    processed = frame[[text_col, label_col]].copy()
    processed[text_col] = processed[text_col].map(normalize_text)
    counts = processed[text_col].map(token_count)
    return processed[(counts >= min_tokens) & (counts <= max_tokens)].reset_index(drop=True)


def stratified_split(
    frame: pd.DataFrame,
    label_col: str,
    seed: int = 42,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_sum = train_size + val_size + test_size
    if abs(split_sum - 1.0) > 1e-9:
        raise ValueError(f"Split proportions must sum to 1.0 (got {split_sum}).")

    train, temp = train_test_split(frame, test_size=(1 - train_size), random_state=seed, stratify=frame[label_col])
    relative_test = test_size / (val_size + test_size)
    val, test = train_test_split(temp, test_size=relative_test, random_state=seed, stratify=temp[label_col])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
