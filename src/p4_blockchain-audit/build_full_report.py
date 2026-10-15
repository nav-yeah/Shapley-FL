#!/usr/bin/env python3
"""
Build the full 1,500-row report (50 rounds x 30 clients).

Merges:
  - byzantine_detection_results.csv
  - shapley_scores.csv
  - gas JSON for the detection contract (0xA769...78Fe, ~668 entries, no tx hashes)
  - gas JSON for the shapley contract   (0x45E4...CC27, ~413 entries, with tx hashes)

Optional --recover pulls the missing entries from Sepolia via the Etherscan V2 API.
Recovery assumes the nth successful tx to a contract is row n of the CSV
(row index = (round-1)*30 + (client_id-1)). That assumption is VERIFIED against
every entry you already have; if any known entry disagrees, nothing is filled.

Usage:
  pip install pandas requests
  python build_full_report.py \
      --detection-csv byzantine_detection_results.csv \
      --shapley-csv shapley_scores.csv \
      --detection-json detection_gas_report.json \
      --shapley-json shapley_gas_report.json \
      [--recover --api-key YOUR_ETHERSCAN_KEY]
"""

import argparse
import json
import sys
import time

import pandas as pd
import requests

CLIENTS = 30
ROUNDS = 50
TOTAL = CLIENTS * ROUNDS


def pos(rnd, cid):
    return (int(rnd) - 1) * CLIENTS + (int(cid) - 1)


def load_report(path):
    with open(path, encoding="utf-8-sig") as f:
        j = json.load(f)
    df = pd.DataFrame(j["perEntry"])
    df["round"] = df["round"].astype(int)
    df["client_id"] = df["client_id"].astype(int)
    df["gas_used"] = df["gasUsed"].astype("int64")
    df["tx_hash"] = df["txHash"] if "txHash" in df else pd.NA
    return j["contractAddress"], df[["round", "client_id", "gas_used", "tx_hash"]]


def fetch_txs(contract, api_key):
    rows, page = [], 1
    while True:
        r = requests.get(
            "https://api.etherscan.io/v2/api",
            params=dict(
                chainid=11155111,
                module="account",
                action="txlist",
                address=contract,
                startblock=0,
                endblock=99999999,
                page=page,
                offset=1000,
                sort="asc",
                apikey=api_key,
            ),
            timeout=60,
        ).json()
        res = r.get("result")
        if not isinstance(res, list):
            if r.get("message") == "No transactions found":
                break
            sys.exit(f"Etherscan error for {contract}: {r}")
        rows += res
        if len(res) < 1000:
            break
        page += 1
        time.sleep(0.25)
    txs = [
        t
        for t in rows
        if t.get("isError") == "0" and (t.get("to") or "").lower() == contract.lower()
    ]
    txs.sort(key=lambda t: (int(t["blockNumber"]), int(t["transactionIndex"])))
    return txs


def attach(base, contract, rep, label, recover, api_key):
    """Return base with <label>_gas_used / <label>_tx_hash / <label>_status columns."""
    d = base[["round", "client_id"]].merge(rep, on=["round", "client_id"], how="left")
    d["pos"] = [pos(r, c) for r, c in zip(d["round"], d["client_id"])]
    d["status"] = d["gas_used"].notna().map({True: "measured", False: "needs_recovery"})
    d["gas_used"] = d["gas_used"].astype("Int64")

    if recover:
        txs = fetch_txs(contract, api_key)
        print(f"[{label}] {len(txs)} successful txs to {contract} (expected {TOTAL})")
        known = d[d["status"] == "measured"]
        bad = []
        for _, r in known.iterrows():
            p = int(r["pos"])
            if p >= len(txs):
                bad.append(
                    (int(r["round"]), int(r["client_id"]), "no tx at this position")
                )
                continue
            t = txs[p]
            if int(t["gasUsed"]) != int(r["gas_used"]):
                bad.append((int(r["round"]), int(r["client_id"]), "gas differs"))
            elif (
                pd.notna(r["tx_hash"])
                and t["hash"].lower() != str(r["tx_hash"]).lower()
            ):
                bad.append((int(r["round"]), int(r["client_id"]), "hash differs"))
        if bad:
            print(
                f"[{label}] ORDER CHECK FAILED on {len(bad)}/{len(known)} known entries "
                f"(first few: {bad[:5]}). Not filling anything for this contract."
            )
        elif len(txs) < TOTAL:
            print(
                f"[{label}] order check passed on {len(known)} known entries, but only "
                f"{len(txs)} txs exist on-chain; filling what exists."
            )
        if not bad:
            print(f"[{label}] order check PASSED on {len(known)} known entries.")
            for i, r in d.iterrows():
                p = int(r["pos"])
                if p < len(txs):
                    t = txs[p]
                    if r["status"] == "needs_recovery":
                        d.at[i, "gas_used"] = int(t["gasUsed"])
                        d.at[i, "status"] = "recovered_from_chain"
                    if pd.isna(r["tx_hash"]):
                        d.at[i, "tx_hash"] = t["hash"]

    d = d.rename(
        columns={
            "gas_used": f"{label}_gas_used",
            "tx_hash": f"{label}_tx_hash",
            "status": f"{label}_status",
        }
    )
    return d[
        [
            "round",
            "client_id",
            f"{label}_gas_used",
            f"{label}_tx_hash",
            f"{label}_status",
        ]
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detection-csv", required=True)
    ap.add_argument("--shapley-csv", required=True)
    ap.add_argument("--detection-json", required=True)
    ap.add_argument("--shapley-json", required=True)
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--out", default="full_report_1500")
    a = ap.parse_args()
    if a.recover and not a.api_key:
        sys.exit("--recover needs --api-key (free Etherscan key)")

    det = pd.read_csv(a.detection_csv)
    sh = pd.read_csv(a.shapley_csv)
    base = det.merge(sh, on=["round", "client_id"], how="outer", validate="1:1")
    base = base.sort_values(["round", "client_id"]).reset_index(drop=True)
    if len(base) != TOTAL:
        sys.exit(f"Expected {TOTAL} merged rows, got {len(base)}")

    c_det, r_det = load_report(a.detection_json)
    c_sh, r_sh = load_report(a.shapley_json)

    out = base.merge(
        attach(base, c_det, r_det, "detection", a.recover, a.api_key),
        on=["round", "client_id"],
        how="left",
        validate="1:1",
    )
    out = out.merge(
        attach(base, c_sh, r_sh, "shapley", a.recover, a.api_key),
        on=["round", "client_id"],
        how="left",
        validate="1:1",
    )

    out.to_csv(f"{a.out}.csv", index=False)
    out.to_json(f"{a.out}.json", orient="records", indent=2)
    missing = out[
        (out["detection_status"] == "needs_recovery")
        | (out["shapley_status"] == "needs_recovery")
    ]
    missing[["round", "client_id", "detection_status", "shapley_status"]].to_csv(
        f"{a.out}_missing.csv", index=False
    )

    print(f"\nRows: {len(out)}")
    for label in ("detection", "shapley"):
        print(
            f"{label}: {out[f'{label}_status'].value_counts().to_dict()}  "
            f"total gas = {int(out[f'{label}_gas_used'].fillna(0).sum()):,}"
        )
    print(f"Wrote {a.out}.csv, {a.out}.json, {a.out}_missing.csv")


if __name__ == "__main__":
    main()
