"""
ShapleyFL - UCI HAR Preprocessing Script
Loads UCI HAR dataset, treats each subject as one federated client (natural non-IID split),
and saves per-client train/test splits ready for use in Flower.

Run from: C:\\Users\\Navya\\ShapleyFL
Usage: python preprocess_har.py
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---- CONFIG ----
DATA_DIR = os.path.join("data", "UCI HAR Dataset")
OUTPUT_DIR = "preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_split(split):
    """Load X, y, and subject IDs for a given split ('train' or 'test')."""
    X_path = os.path.join(DATA_DIR, split, f"X_{split}.txt")
    y_path = os.path.join(DATA_DIR, split, f"y_{split}.txt")
    subj_path = os.path.join(DATA_DIR, split, f"subject_{split}.txt")

    X = pd.read_csv(X_path, sep=r"\s+", header=None).values
    y = pd.read_csv(y_path, header=None).values.flatten()
    subjects = pd.read_csv(subj_path, header=None).values.flatten()

    return X, y, subjects


def main():
    print("Loading UCI HAR dataset...")
    X_train, y_train, subj_train = load_split("train")
    X_test, y_test, subj_test = load_split("test")

    # Combine train+test, we'll re-split per client ourselves
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    subj_all = np.concatenate([subj_train, subj_test])

    unique_subjects = np.unique(subj_all)
    print(f"Found {len(unique_subjects)} subjects (clients): {unique_subjects}")
    print(f"Total samples: {X_all.shape[0]}, Features: {X_all.shape[1]}")

    # ---- Partition by subject = natural non-IID client split ----
    client_data = {}
    for subj in unique_subjects:
        mask = subj_all == subj
        X_client = X_all[mask]
        y_client = y_all[mask] - 1  # labels are 1-indexed in UCI HAR, convert to 0-indexed

        # 80/20 train/test split per client
        n = len(X_client)
        idx = np.random.RandomState(42).permutation(n)
        split_point = int(0.8 * n)
        train_idx, test_idx = idx[:split_point], idx[split_point:]

        client_data[int(subj)] = {
            "X_train": X_client[train_idx].astype(np.float32),
            "y_train": y_client[train_idx].astype(np.int64),
            "X_test": X_client[test_idx].astype(np.float32),
            "y_test": y_client[test_idx].astype(np.int64),
        }
        print(f"  Client {subj}: {len(train_idx)} train, {len(test_idx)} test samples, "
              f"classes present: {sorted(set(y_client))}")

    # ---- Save to disk ----
    out_path = os.path.join(OUTPUT_DIR, "har_clients.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(client_data, f)
    print(f"\nSaved preprocessed client data to: {out_path}")

    # ---- Plot label distribution per client (visual non-IID confirmation) ----
    activity_labels = pd.read_csv(
        os.path.join(DATA_DIR, "activity_labels.txt"),
        sep=r"\s+", header=None, names=["id", "activity"]
    )
    activity_names = activity_labels["activity"].tolist()

    fig, ax = plt.subplots(figsize=(14, 7))
    client_ids = sorted(client_data.keys())
    n_classes = 6  # UCI HAR has 6 activities

    label_matrix = np.zeros((len(client_ids), n_classes))
    for i, cid in enumerate(client_ids):
        y_c = np.concatenate([client_data[cid]["y_train"], client_data[cid]["y_test"]])
        for c in range(n_classes):
            label_matrix[i, c] = np.sum(y_c == c)

    bottom = np.zeros(len(client_ids))
    for c in range(n_classes):
        ax.bar(client_ids, label_matrix[:, c], bottom=bottom, label=activity_names[c])
        bottom += label_matrix[:, c]

    ax.set_xlabel("Client (Subject) ID")
    ax.set_ylabel("Number of samples")
    ax.set_title("Label Distribution per Client - UCI HAR (Non-IID by Subject)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "label_distribution.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved label distribution plot to: {plot_path}")
    print("\nDone. Open the PNG to visually confirm non-IID skew across clients.")


if __name__ == "__main__":
    main()