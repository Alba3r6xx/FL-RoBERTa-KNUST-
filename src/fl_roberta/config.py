from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    seed: int = 42
    max_length: int = 512
    train_split: float = 0.7
    val_split: float = 0.1
    test_split: float = 0.2

    roberta_model_name: str = "roberta-base"
    bert_model_name: str = "bert-base-uncased"

    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 10
    dropout: float = 0.1
    encoder_freeze_epochs: int = 3

    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    tfidf_max_features: int = 10_000
