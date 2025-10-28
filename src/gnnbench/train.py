from typing import Dict
import copy, os 
import torch
import torch.nn.functional as F
from .metrics import accuracy, macro_f1
from .utils import ensure_dir, save_json

@torch.no_grad()
def evaluate(model, data, split="val") -> Dict[str, float]:
    model.eval()
    out = model(data.x, data.edge_index)
    if split == "train":
        mask = data.train_mask
    elif split == "val":
        mask = data.val_mask
    elif split == "test":
        mask = data.test_mask
    else:
        raise ValueError("split must be train-val-test")
    loss = F.cross_entropy(out[mask], data.y[mask])
    return {
        "loss": float(loss.item()),
        "acc": accuracy(out[mask], data.y[mask]),
        "macro_f1": macro_f1(out[mask], data.y[mask])
    }

def train_gcn(model, data, *, epochs=500, patience=50, lr=0.01, weight_decay=5e-4,
              results_dir="results", save_best=True, verbose=True) -> Dict:
    device = next(model.parameters()).device
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {"val_macro_f1": -1.0, "state": None}
    bad = 0
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        val = evaluate(model, data, "val")
        history.append({"epoch": ep, **val})

        if val["macro_f1"] > best["val_macro_f1"]:
            best["val_macro_f1"] = val["macro_f1"]
            best["state"] = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"[{ep:03d}] train_loss={float(loss.item()):.4f} | val_f1={val['macro_f1']:.4f}")

        if bad >= patience:
            if verbose:
                print(f"Early stopping at epoch {ep} (patience={patience}).")
            break

    if save_best and best["state"] is not None:
        model.load_state_dict(best["state"])

    test = evaluate(model, data, "test")
    result = {"val_best_macro_f1": best["val_macro_f1"], "test": test, "history": history}

    # optional save
    ensure_dir(results_dir)
    save_json(result, os.path.join(results_dir, "last_run.json"))
    return result