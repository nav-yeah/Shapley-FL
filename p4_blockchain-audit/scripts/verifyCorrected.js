const fs = require("fs");
const hre = require("hardhat");

function parseCSV(file) {
  const lines = fs.readFileSync(file, "utf8")
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .filter(line => line.trim());

  const headers = lines[0].split(",").map(x => x.trim());

  return lines.slice(1).map(line => {
    const cells = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = (cells[i] ?? "").trim();
    });
    return row;
  });
}

async function main() {
  if (hre.network.name !== "sepolia") {
    throw new Error("Wrong network: expected sepolia");
  }

  const deployment = JSON.parse(
    fs.readFileSync("./deployed-address-corrected.json", "utf8")
  );

  const address = deployment.address;

  if (
    address.toLowerCase() !==
    "0x45e4a0f7933a56d8a30288039c883d36f768cc27"
  ) {
    throw new Error("Unexpected contract address; stopping verification.");
  }

  const shapleyRows = parseCSV("./data/shapley_scores.csv");
  const byzRows = parseCSV("./data/byzantine_detection_results.csv");

  if (shapleyRows.length !== 1500 || byzRows.length !== 1500) {
    throw new Error(
      `Expected 1500 rows in each CSV. Shapley=${shapleyRows.length}, detection=${byzRows.length}`
    );
  }

  const shapleyMap = new Map();

  for (const r of shapleyRows) {
    const key = `${r.round}-${r.client_id}`;

    if (shapleyMap.has(key)) {
      throw new Error(`Duplicate Shapley key: ${key}`);
    }

    if (!Number.isFinite(Number(r.shapley_value))) {
      throw new Error(`Invalid Shapley value: ${key}`);
    }

    shapleyMap.set(key, r);
  }

  const seen = new Set();

  const rows = byzRows.map(b => {
    const key = `${b.round}-${b.client_id}`;

    if (seen.has(key)) {
      throw new Error(`Duplicate detection key: ${key}`);
    }
    seen.add(key);

    const s = shapleyMap.get(key);

    if (!s) {
      throw new Error(`No matching Shapley row: ${key}`);
    }

    if (!["0", "1"].includes(b.flagged_status)) {
      throw new Error(`Invalid flagged_status: ${key}`);
    }

    if (!/^\d+$/.test(b.round) || !/^\d+$/.test(b.client_id)) {
      throw new Error(`Invalid round/client ID: ${key}`);
    }

    return {
      round: b.round,
      client_id: b.client_id,
      shapley_value: s.shapley_value,
      flagged: b.flagged_status === "1"
    };
  });

  const contract = await hre.ethers.getContractAt(
    "ShapleyAudit",
    address
  );

  const total = Number(await contract.totalEntries());

  console.log("Contract:", address);
  console.log("CSV rows:", rows.length);
  console.log("On-chain entries:", total);

  const mismatches = [];

  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const key = `${r.round}-${r.client_id}`;

    try {
      const scaled = BigInt(
        Math.round(Number(r.shapley_value) * 1e18)
      );

      const dataString =
        `${r.round}|${r.client_id}|${r.shapley_value}|${r.flagged}`;

      const expectedHash = hre.ethers.keccak256(
        hre.ethers.toUtf8Bytes(dataString)
      );

      const e = await contract.getEntry(r.round, r.client_id);
      const errors = [];

      if (e.round !== BigInt(r.round)) {
        errors.push("round");
      }

      if (e.clientId !== BigInt(r.client_id)) {
        errors.push("client_id");
      }

      if (e.shapleyValueScaled !== scaled) {
        errors.push("Shapley value");
      }

      if (e.flagged !== r.flagged) {
        errors.push("flagged status");
      }

      if (
        e.dataHash.toLowerCase() !== expectedHash.toLowerCase()
      ) {
        errors.push("data hash");
      }

      if (errors.length) {
        mismatches.push({ key, errors });
      }
    } catch (err) {
      mismatches.push({
        key,
        errors: [err.shortMessage || err.message || String(err)]
      });
    }

    if ((i + 1) % 100 === 0) {
      console.log(`Checked ${i + 1}/${rows.length}`);
    }
  }

  const output = {
    contractAddress: address,
    network: hre.network.name,
    sourceShapley: "./data/shapley_scores.csv",
    sourceDetection: "./data/byzantine_detection_results.csv",
    csvRows: rows.length,
    onChainEntries: total,
    verifiedRows: rows.length - mismatches.length,
    mismatchCount: mismatches.length,
    mismatches: mismatches.slice(0, 100)
  };

  fs.writeFileSync(
    "./verification_summary_corrected.json",
    JSON.stringify(output, null, 2) + "\n"
  );

  console.log("\n--- Verification summary ---");
  console.log("Verified rows:", output.verifiedRows);
  console.log("Mismatches:", output.mismatchCount);
  console.log("Report: ./verification_summary_corrected.json");

  if (mismatches.length === 0 && total === 1500) {
    console.log("SUCCESS: all 1500 records match.");
  } else {
    console.log("VERIFICATION FAILED. Review the report.");
  }
}

main().catch(err => {
  console.error("Verification error:", err);
  process.exitCode = 1;
});