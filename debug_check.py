"""
Diagnostic: directly compare V(full coalition) as computed by
exact_shapley.py's logic vs gtg_shapley_approx.py's logic.
If these don't match, we've found the bug.
"""
import pickle
import numpy as np
from fl_utils import get_model, local_train, fedavg, evaluate, clone_state_dict, LOCAL_EPOCHS, LOCAL_LR

PKL_PATH = "preprocessed/har_clients.pkl"
GLOBAL_SEED = 42
N_ROUNDS = 5

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

with open("exact_shapley_results.pkl", "rb") as f:
    exact_results = pickle.load(f)
client_ids = exact_results["client_subset"]
print(f"Client ids: {client_ids}")

X_test = np.concatenate([data[c]["X_test"] for c in client_ids])
y_test = np.concatenate([data[c]["y_test"] for c in client_ids])

def train_coalition(client_subset, n_rounds=N_ROUNDS, seed=GLOBAL_SEED):
    global_state = clone_state_dict(get_model(seed=seed).state_dict())
    sizes = {c: len(data[c]["y_train"]) for c in client_subset}
    for r in range(n_rounds):
        states, weights = [], []
        for cid in client_subset:
            m = get_model()
            m.load_state_dict(global_state)
            new_state = local_train(m, data[cid]["X_train"], data[cid]["y_train"],
                                     epochs=LOCAL_EPOCHS, lr=LOCAL_LR,
                                     seed=seed + r * 100 + cid)
            states.append(new_state)
            weights.append(sizes[cid])
        global_state = fedavg(states, weights)
    return global_state

print("\nRun 1 (full coalition):")
state1 = train_coalition(client_ids)
v1 = evaluate(state1, X_test, y_test)
print(f"  V(full) = {v1:.6f}")

print("\nRun 2 (full coalition, repeated for determinism check):")
state2 = train_coalition(client_ids)
v2 = evaluate(state2, X_test, y_test)
print(f"  V(full) = {v2:.6f}")

print(f"\nMatch: {abs(v1 - v2) < 1e-9}")
print(f"\nphi_exact sum = {sum(exact_results['phi_exact'].values()):.6f}")
print(f"V(full) - V(empty) should roughly relate to this sum (efficiency property)")

empty_acc = float((np.full_like(y_test, np.bincount(y_test).argmax()) == y_test).mean())
print(f"V(empty) [majority class baseline] = {empty_acc:.6f}")
print(f"V(full) - V(empty) = {v1 - empty_acc:.6f}")
