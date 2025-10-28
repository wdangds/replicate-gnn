import torch
from sklearn.metrics import f1_score

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()

def macro_f1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1).cpu().numpy()
    return float(f1_score(y.cpu().numpy(), pred, average="macro"))
