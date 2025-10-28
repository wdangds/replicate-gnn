from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # data
    dataset: str = "Cora"
    root: str = "data"
    split: str = "auto" # "auto", "random", "ratio"
    train_per_class: int = 20  # used if split = "random"
    val_per_class: int = 30 
    train_ratio: float = 0.6 # used if split = "ratio"
    val_ratio: float = 0.2

    # model
    hidden: int = 64
    dropout: float = 0.5
    layers: int = 2
    residual: bool = True
    norm: str = "batch" # "batch", "layer", "none"

    # training 
    epochs: int = 500
    patience: int = 50
    lr: float = 0.01
    weight_decay: float = 5e-4
    seed: int = 42

    # misc
    device: str = "auto" # "auto", "cpu", "cuda"
    results_dir: str = "results"
    save_best: bool = True
    verbose: bool = True