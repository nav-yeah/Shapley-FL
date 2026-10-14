import argparse
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

import gtg_shapley_full_scale as seq   # reuse constants + helper functions

_DATA = None
_X_TEST = None
_Y_TEST = None


def _init_worker(data, x_test, y_test):
    global _DATA, _X_TEST, _Y_TEST
    _DATA, _X_TEST, _Y_TEST = data, x_test, y_test
    try:
        import torch
        torch.set_num_threads(1)      # one thread per process; parallelism is across processes
    except Exception:
        pass


def _eval_coalition(args):
    key, n_rounds, global_seed = args
    state = seq.train_subcoalition_fl(list(key), _DATA, n_rounds, global_seed)
    return key, seq.evaluate(state, _X_TEST, _Y_TEST)


def gtg_core(client_ids, eval_batch, n_permutations, m, eps_b, eps_i, seed,
             wave, conv_window, conv_threshold, verbose=True):
    """The GTG-Shapley loop. `eval_batch(list_of_keys) -> {key: utility}`."""
    rng = np.random.RandomState(seed)
    n = len(client_ids)
    cache = {}
    used = set()
    n_trained = 0

    def need(keys):
        nonlocal n_trained
        keys = [k for k in dict.fromkeys(keys) if k not in cache]
        if keys:
            cache.update(eval_batch(keys))
            n_trained += len(keys)

    k_empty, k_full = (), tuple(sorted(client_ids))
    need([k_empty, k_full])
    used.update([k_empty, k_full])
    v0, vN = cache[k_empty], cache[k_full]
    print(f"V(empty) = {v0:.4f}, V(full {n} clients) = {vN:.4f}", flush=True)

    phi = {cid: 0.0 for cid in client_ids}
    if abs(vN - v0) <= eps_b:
        print(f"Truncated entirely: |vN-v0|={abs(vN-v0):.4f} <= eps_b={eps_b}")
        return phi, len(used), n_trained

    phi_history = []
    k = 0
    for k in range(1, n_permutations + 1):
        pi_k = seq.guided_permutation(client_ids, m, k, rng)
        v_prev = v0
        for j in range(1, n + 1):
            if abs(vN - v_prev) >= eps_i:
                key = tuple(sorted(pi_k[:j]))
                if key not in cache:
                    ahead = [tuple(sorted(pi_k[:jj])) for jj in range(j, min(n, j + wave - 1) + 1)]
                    need(ahead)
                used.add(key)
                v_j = cache[key]
            else:
                v_j = v_prev
            cid_j = pi_k[j - 1]
            phi[cid_j] = ((k - 1) / k) * phi[cid_j] + (1 / k) * (v_j - v_prev)
            v_prev = v_j

        phi_history.append(dict(phi))

        if k >= max(conv_window + 1, 50):
            old_phi = phi_history[k - 1 - conv_window]
            diffs = []
            for cid in client_ids:
                denom = abs(phi[cid]) if abs(phi[cid]) > 1e-9 else 1e-9
                diffs.append(abs(phi[cid] - old_phi[cid]) / denom)
            if np.mean(diffs) < conv_threshold:
                if verbose:
                    print(f"Converged at permutation {k}", flush=True)
                break

        if verbose and k % 5 == 0:
            print(f"  ...permutation {k}/{n_permutations}, {len(used)} unique coalitions "
                  f"needed, {n_trained} trained", flush=True)

    if verbose:
        print(f"\nRan {k} permutations, {len(used)} unique coalitions needed "
              f"(theoretical max for exact Shapley: 2^{n} = {2**n}); "
              f"{n_trained} coalition trainings actually run", flush=True)
    return phi, len(used), n_trained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--rounds", type=int, default=seq.N_ROUNDS)
    args = ap.parse_args()

    print("Loading preprocessed HAR client data (all 30 clients)...", flush=True)
    data = seq.load_client_data(seq.PKL_PATH)
    client_ids = sorted(data.keys())
    X_test = np.concatenate([data[c]["X_test"] for c in client_ids])
    y_test = np.concatenate([data[c]["y_test"] for c in client_ids])
    print(f"Clients: {client_ids}")
    print(f"Shared test set size: {len(y_test)} samples")
    print(f"Rounds: {args.rounds}, worker processes: {args.workers}\n", flush=True)

    start = time.time()
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(data, X_test, y_test)) as pool:
        def eval_batch(keys):
            jobs = [(k, args.rounds, seq.GLOBAL_SEED) for k in keys]
            return dict(pool.map(_eval_coalition, jobs, chunksize=1))

        phi, n_unique, n_trained = gtg_core(
            client_ids, eval_batch, seq.N_PERMUTATIONS, seq.M_GUIDED,
            seq.EPSILON_B, seq.EPSILON_I, seq.RANDOM_SEED,
            wave=args.workers, conv_window=seq.CONVERGENCE_WINDOW,
            conv_threshold=seq.CONVERGENCE_THRESHOLD)
    elapsed = time.time() - start

    print("\n=== SHAPLEY VALUES (all 30 clients) ===")
    for cid in client_ids:
        print(f"  Client {cid}: phi = {phi[cid]:.6f}")
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Unique coalitions evaluated: {n_unique} (vs theoretical 2^30 = {2**30:,} for exact Shapley)")
    print(f"Efficiency ratio: {n_unique}/{2**30} = {n_unique/2**30:.2e}")
    print(f"Sum of phi = {sum(phi.values()):.4f}")

    with open("gtg_shapley_full_scale_results.pkl", "wb") as f:
        pickle.dump({"client_ids": client_ids, "phi": phi, "elapsed_time": elapsed,
                     "n_unique_coalitions": n_unique, "n_trainings_run": n_trained,
                     "n_rounds": args.rounds, "workers": args.workers}, f)
    print("\nSaved to gtg_shapley_full_scale_results.pkl")


if __name__ == "__main__":
    mp.freeze_support()
    main()