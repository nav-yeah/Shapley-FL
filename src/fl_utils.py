"""
Shared utilities for the FL Shapley experiments: a small PyTorch MLP,
a multi-round local training routine, and FedAvg aggregation of model
STATE DICTS (not one-shot sklearn coefficient averaging).

This is the correct setting for GTG-Shapley's gradient-reconstruction
trick: each client's "update" is a small step from a shared starting
point, accumulated over multiple rounds, so FedAvg-ing those updates
approximates retraining well (unlike one-shot logistic regression fits).
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

N_FEATURES = 561
N_CLASSES = 6
HIDDEN_DIM = 64
LOCAL_EPOCHS = 3
LOCAL_LR = 0.01
DEVICE = "cpu"  # HAR is tiny, CPU is fine and keeps things simple/reproducible


class SimpleMLP(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden=HIDDEN_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_model(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return SimpleMLP().to(DEVICE)


def state_dict_to_vector(state_dict):
    """Flatten a state_dict into a single numpy vector (for diffing/scaling)."""
    return torch.cat([p.flatten() for p in state_dict.values()]).detach().numpy()


def clone_state_dict(state_dict):
    return copy.deepcopy(state_dict)


def local_train(model, X, y, epochs=LOCAL_EPOCHS, lr=LOCAL_LR, batch_size=32, seed=0):
    """
    Trains `model` IN PLACE for a few local epochs on (X, y).
    Returns the resulting state_dict (the client's local model after training).
    This stands in for one client's local FL training round.
    """
    torch.manual_seed(seed)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = len(y)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    return clone_state_dict(model.state_dict())


def fedavg(state_dicts, weights):
    """
    Weighted average of a list of state_dicts.
    weights: list of floats summing to 1 (or will be normalized here).
    """
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    avg_state = {}
    keys = state_dicts[0].keys()
    for key in keys:
        stacked = torch.stack([sd[key].float() * w
                                for sd, w in zip(state_dicts, weights)])
        avg_state[key] = stacked.sum(dim=0)
    return avg_state


def evaluate(state_dict, X_test, y_test):
    """
    Loads state_dict into a fresh model and evaluates accuracy on X_test/y_test.
    If state_dict is None, returns majority-class baseline accuracy
    (matches the "empty coalition" convention used in exact Shapley).
    """
    if state_dict is None:
        majority_class = np.bincount(y_test).argmax()
        preds = np.full_like(y_test, majority_class)
        return float((preds == y_test).mean())

    model = get_model()
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_t)
        preds = torch.argmax(logits, dim=1).numpy()
    return float((preds == y_test).mean())


def run_federated_rounds(client_ids, data, global_seed=42, n_rounds=5,
                          local_epochs=LOCAL_EPOCHS, lr=LOCAL_LR, verbose=True):
    """
    Runs standard multi-round FedAvg training across client_ids.
    Returns:
        global_state_history: list of global state_dicts, one per round
                               (index 0 = initial random init, before any training)
        client_state_history: list of dicts {client_id: state_dict}, one per round
                               (each client's local model state AFTER training
                                that round, starting from that round's global model)
        client_sizes: dict {client_id: n_train_samples}
    """
    global_model = get_model(seed=global_seed)
    global_state = clone_state_dict(global_model.state_dict())

    global_state_history = [global_state]
    client_state_history = []

    client_sizes = {cid: len(data[cid]["y_train"]) for cid in client_ids}

    for r in range(n_rounds):
        round_client_states = {}
        for cid in client_ids:
            local_model = get_model()
            local_model.load_state_dict(global_state_history[-1])
            X_i, y_i = data[cid]["X_train"], data[cid]["y_train"]
            new_state = local_train(local_model, X_i, y_i,
                                     epochs=local_epochs, lr=lr,
                                     seed=global_seed + r * 100 + cid)
            round_client_states[cid] = new_state

        client_state_history.append(round_client_states)

        # Server-side FedAvg over this round's client updates
        weights = [client_sizes[cid] for cid in client_ids]
        states = [round_client_states[cid] for cid in client_ids]
        new_global_state = fedavg(states, weights)
        global_state_history.append(new_global_state)

        if verbose:
            print(f"  Round {r+1}/{n_rounds} complete.")

    return global_state_history, client_state_history, client_sizes