#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from fl_roberta.config import ExperimentConfig
from fl_roberta.data import preprocess_frame, stratified_split


def train_logistic_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str, label_col: str, cfg: ExperimentConfig) -> None:
    vectorizer = TfidfVectorizer(max_features=cfg.tfidf_max_features)
    x_train = vectorizer.fit_transform(train_df[text_col])
    x_test = vectorizer.transform(test_df[text_col])

    clf = LogisticRegression(max_iter=cfg.logistic_max_iter)
    clf.fit(x_train, train_df[label_col])
    preds = clf.predict(x_test)

    print(f"Accuracy: {accuracy_score(test_df[label_col], preds):.4f}")
    print(f"Macro F1: {f1_score(test_df[label_col], preds, average='macro'):.4f}")
    print(classification_report(test_df[label_col], preds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train starter baseline models for FL-RoBERTa study.")
    parser.add_argument("--data", required=True, help="Path to CSV with text and label columns.")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    df = pd.read_csv(args.data)
    df = preprocess_frame(df, text_col=args.text_col, label_col=args.label_col, max_tokens=cfg.max_length)
    train_df, _, test_df = stratified_split(
        df,
        label_col=args.label_col,
        seed=cfg.seed,
        train_size=cfg.train_split,
        val_size=cfg.val_split,
        test_size=cfg.test_split,
    )

    train_logistic_regression(train_df, test_df, args.text_col, args.label_col, cfg)


if __name__ == "__main__":
    main()
