"""
Build the 5 GTG-Shapley paper distribution scenarios on HAR data, using a
subset of N_CLIENTS clients to keep runtime practical (paper uses n=10
participants on MNIST -- we mirror n=10 here).

Scenarios (mirroring Section 5.1.1 of Liu et al. 2022):
  1. same_dist_same_size   -- baseline IID-like: equal-ish sizes, natural classes
  2. diff_dist_same_size   -- label skew: 2 clients get 80% of one class each
  3. same_dist_diff_size   -- size skew: ratios 10/15/20/25/30% applied in pairs
  4. noisy_labels          -- label flip: 0/5/10/15/20% per client pair
  5. noisy_features        -- Gaussian noise added to features: 0/5/10/15/20%

Each scenario produces its own client_data dict with the SAME 10 base
client IDs but modified X/y, saved as separate pickle files so
gtg_shapley_approx.py (or a small wrapper) can be run on each independently.

Usage:
    python build_scenarios.py

Expects: preprocessed/har_clients.pkl
Produces: scenarios/scenario_1_same_dist_same_size.pkl
          scenarios/scenario_2_diff_dist_same_size.pkl
          scenarios/scenario_3_same_dist_diff_size.pkl
          scenarios/scenario_4_noisy_labels.pkl
          scenarios/scenario_5_noisy_features.pkl
"""

import os
import pickle
import numpy as np

PKL_PATH = "preprocessed/har_clients.pkl"
OUT_DIR = "scenarios"
N_CLIENTS = 10
RANDOM_SEED = 42
N_CLASSES = 6


def load_client_data(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def select_base_clients(data, n, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    all_ids = list(data.keys())
    chosen = rng.choice(all_ids, size=n, replace=False)
    return sorted(chosen.tolist())


def equalize_sizes(data, client_ids, target_size, seed=RANDOM_SEED):
    """
    Subsample each client's train set down to target_size samples
    (without replacement), keeping the original class proportions as
    close as possible by simple random subsampling.
    """
    rng = np.random.RandomState(seed)
    out = {}
    for cid in client_ids:
        X, y = data[cid]["X_train"], data[cid]["y_train"]
        n_avail = len(y)
        size = min(target_size, n_avail)
        idx = rng.choice(n_avail, size=size, replace=False)
        out[cid] = {
            "X_train": X[idx],
            "y_train": y[idx],
            "X_test": data[cid]["X_test"],
            "y_test": data[cid]["y_test"],
        }
    return out


# ----------------------------------------------------------------------
# Scenario 1: Same Distribution and Same Size
# ----------------------------------------------------------------------
def build_scenario_1(data, client_ids):
    """
    Equalize all clients to the same (smallest available) train size.
    Classes are whatever each subject naturally has -- this is the
    closest "IID-like" baseline achievable with subject-partitioned data.
    """
    min_size = min(len(data[cid]["y_train"]) for cid in client_ids)
    return equalize_sizes(data, client_ids, min_size)


# ----------------------------------------------------------------------
# Scenario 2: Different Distributions and Same Size (label skew)
# ----------------------------------------------------------------------
def build_scenario_2(data, client_ids, seed=RANDOM_SEED):
    """
    Mirrors the paper: pick 2 clients to have 80% of one class each,
    remaining clients keep roughly natural-ish proportions but same size.
    All clients still end up the same total size (paper's 1084/client
    equivalent -- here we use the min available size from scenario 1).
    """
    rng = np.random.RandomState(seed)
    min_size = min(len(data[cid]["y_train"]) for cid in client_ids)
    out = {}

    skewed_clients = client_ids[:2]   # first 2 clients get label skew
    skewed_classes = [0, 1]            # WALKING, WALKING_UPSTAIRS

    for i, cid in enumerate(client_ids):
        X, y = data[cid]["X_train"], data[cid]["y_train"]

        if cid in skewed_clients:
            target_class = skewed_classes[skewed_clients.index(cid)]
            n_majority = int(0.8 * min_size)
            n_minority = min_size - n_majority

            majority_idx = np.where(y == target_class)[0]
            minority_idx = np.where(y != target_class)[0]

            # sample with replacement if not enough majority-class samples
            maj_pick = rng.choice(majority_idx,
                                   size=min(n_majority, len(majority_idx)),
                                   replace=len(majority_idx) < n_majority)
            min_pick = rng.choice(minority_idx,
                                   size=min(n_minority, len(minority_idx)),
                                   replace=len(minority_idx) < n_minority)
            idx = np.concatenate([maj_pick, min_pick])
        else:
            n_avail = len(y)
            idx = rng.choice(n_avail, size=min(min_size, n_avail), replace=False)

        out[cid] = {
            "X_train": X[idx],
            "y_train": y[idx],
            "X_test": data[cid]["X_test"],
            "y_test": data[cid]["y_test"],
        }
    return out


# ----------------------------------------------------------------------
# Scenario 3: Same Distribution and Different Sizes (size skew)
# ----------------------------------------------------------------------
def build_scenario_3(data, client_ids, seed=RANDOM_SEED):
    """
    Paper's ratios (10/15/20/25/30%) applied in pairs across 10 clients.
    Ratios are relative to the largest available "full" pool size.
    """
    rng = np.random.RandomState(seed)
    ratios = [0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 0.25, 0.25, 0.30, 0.30]
    # base pool size: use the smallest client's available size as the
    # "100%" reference point, scaled down to keep things from exceeding
    # any single client's actual data
    base_pool = min(len(data[cid]["y_train"]) for cid in client_ids)
    max_target = base_pool  # 30% of "full" should still be <= what's available

    out = {}
    for cid, ratio in zip(client_ids, ratios):
        X, y = data[cid]["X_train"], data[cid]["y_train"]
        # scale ratio so that the LARGEST ratio (0.30) maps to max_target
        size = max(10, int((ratio / 0.30) * max_target))
        n_avail = len(y)
        idx = rng.choice(n_avail, size=min(size, n_avail), replace=False)
        out[cid] = {
            "X_train": X[idx],
            "y_train": y[idx],
            "X_test": data[cid]["X_test"],
            "y_test": data[cid]["y_test"],
        }
    return out


# ----------------------------------------------------------------------
# Scenario 4: Noisy Labels and Same Size
# ----------------------------------------------------------------------
def build_scenario_4(data, client_ids, seed=RANDOM_SEED):
    """
    Built on top of scenario 3's (same-distribution, diff-size) data,
    per the paper -- but we use scenario 1's equal sizes for simplicity
    and clarity (paper's choice of base doesn't materially change what
    label noise demonstrates). Flip percentages: 0/5/10/15/20% in pairs.

    NOTE: this directly mirrors what P3 (Byzantine detection) will need --
    flag this overlap to your team.
    """
    rng = np.random.RandomState(seed)
    base = build_scenario_1(data, client_ids)
    flip_rates = [0.0, 0.0, 0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.20, 0.20]

    out = {}
    for cid, flip_rate in zip(client_ids, flip_rates):
        X, y = base[cid]["X_train"], base[cid]["y_train"].copy()
        n = len(y)
        n_flip = int(flip_rate * n)
        if n_flip > 0:
            flip_idx = rng.choice(n, size=n_flip, replace=False)
            for idx in flip_idx:
                # flip to a random DIFFERENT class
                other_classes = [c for c in range(N_CLASSES) if c != y[idx]]
                y[idx] = rng.choice(other_classes)
        out[cid] = {
            "X_train": X,
            "y_train": y,
            "X_test": base[cid]["X_test"],
            "y_test": base[cid]["y_test"],
        }
    return out


# ----------------------------------------------------------------------
# Scenario 5: Noisy Features and Same Size
# ----------------------------------------------------------------------
def build_scenario_5(data, client_ids, seed=RANDOM_SEED):
    """
    Built on scenario 1 (equal sizes). Gaussian noise added to features,
    scaled relative to each feature's own std so noise is meaningful
    across HAR's 561 differently-scaled features.
    Noise levels: 0/5/10/15/20% in pairs (interpreted as noise std
    relative to each feature's per-client std).
    """
    rng = np.random.RandomState(seed)
    base = build_scenario_1(data, client_ids)
    noise_levels = [0.0, 0.0, 0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.20, 0.20]

    out = {}
    for cid, noise_level in zip(client_ids, noise_levels):
        X = base[cid]["X_train"].copy()
        if noise_level > 0:
            feature_std = X.std(axis=0, keepdims=True)
            noise = rng.normal(loc=0.0, scale=noise_level * feature_std,
                                size=X.shape)
            X = X + noise
        out[cid] = {
            "X_train": X,
            "y_train": base[cid]["y_train"],
            "X_test": base[cid]["X_test"],
            "y_test": base[cid]["y_test"],
        }
    return out


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading preprocessed HAR client data...")
    data = load_client_data(PKL_PATH)

    client_ids = select_base_clients(data, N_CLIENTS)
    print(f"Selected {N_CLIENTS} base clients: {client_ids}")

    scenarios = {
        "scenario_1_same_dist_same_size": build_scenario_1(data, client_ids),
        "scenario_2_diff_dist_same_size": build_scenario_2(data, client_ids),
        "scenario_3_same_dist_diff_size": build_scenario_3(data, client_ids),
        "scenario_4_noisy_labels": build_scenario_4(data, client_ids),
        "scenario_5_noisy_features": build_scenario_5(data, client_ids),
    }

    for name, scenario_data in scenarios.items():
        path = os.path.join(OUT_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "client_ids": client_ids,
                "data": scenario_data,
            }, f)
        sizes = [len(scenario_data[c]["y_train"]) for c in client_ids]
        print(f"  {name}: sizes={sizes}")
        print(f"    Saved to {path}")

    print("\nAll 5 scenarios built. Next: run GTG-Shapley on each.")
