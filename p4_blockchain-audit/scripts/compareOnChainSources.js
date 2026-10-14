const fs = require("fs");
const hre = require("hardhat");

function parseCSV(file) {
  const lines = fs.readFileSync(file, "utf8")
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .filter(line => line.trim());

  const headers = lines[0].split(",").map(h => h.trim());

  return lines.slice(1).map(line => {
    const cells = line.split(",");
    return Object.fromEntries(
      headers.map((h, i) => [h, (cells[i] ?? "").trim()])
    );
  });
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function getEntryWithRetry(contract, round, clientId) {
  let lastError;

  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      return await contract.getEntry(round, clientId);
    } catch (err) {
      lastError = err;
      const message = String(err.shortMessage || err.message || err);

      // Do not retry a genuine missing-entry or contract revert indefinitely.
      if (
        /execution reverted|missing revert data|could not decode result/i.test(message)
      ) {
        throw err;
      }

      if (attempt < 5) {
        const delay = Math.min(1000 * (2 ** attempt), 15000);
        console.log(
          `RPC retry ${attempt + 1}/5 for ${round}-${clientId}; waiting ${delay} ms`
        );
        await sleep(delay);
      }
    }
  }

  throw lastError;
}

async function main() {
  if (hre.network.name !== "sepolia") {
    throw new Error("Wrong network; expected Sepolia.");
  }

  const byz = parseCSV("./data/byzantine_detection_results.csv");
  const shapleyRows = parseCSV("./data/shapley_scores.csv");
  const merged = parseCSV(require("path").join(__dirname, "..", "..", "byzantine_detection_results_merged.csv"));

  const shapleyIndex = new Map(
    shapleyRows.map(r => [`${r.round}-${r.client_id}`, r])
  );
  const mergedIndex = new Map(
    merged.map(r => [`${r.round}-${r.client_id}`, r])
  );

  const deployed = JSON.parse(
    fs.readFileSync("./deployed-address.json", "utf8")
  );
  const contract = await hre.ethers.getContractAt(
    "ShapleyAudit",
    deployed.address
  );

  const total = Number(await contract.totalEntries());

  console.log("Network:", hre.network.name);
  console.log("Contract:", deployed.address);
  console.log("On-chain total entries:", total);
  console.log("Original detection CSV rows:", byz.length);
  console.log("Original Shapley CSV rows:", shapleyRows.length);
  console.log("Merged CSV rows:", merged.length);

  const results = [];
  let rpcErrors = 0;

  for (let i = 0; i < byz.length; i++) {
    const b = byz[i];
    const key = `${b.round}-${b.client_id}`;
    const s = shapleyIndex.get(key);
    const m = mergedIndex.get(key);

    if (!s || !s.shapley_value) {
      results.push({
        key,
        status: "MISSING_SOURCE_SHAPLEY",
        originalShapley: s?.shapley_value ?? null,
        mergedShapley: m?.shapley_value ?? null
      });
      continue;
    }

    try {
      const entry = await getEntryWithRetry(
        contract,
        b.round,
        b.client_id
      );

      const onChainScaled = entry.shapleyValueScaled.toString();
      const originalScaled = BigInt(
        Math.round(Number(s.shapley_value) * 1e18)
      ).toString();

      const mergedScaled = m?.shapley_value
        ? BigInt(
            Math.round(Number(m.shapley_value) * 1e18)
          ).toString()
        : null;

      const originalFlag = b.flagged_status === "1";
      const mergedFlag = m
        ? m.flagged_status === "1"
        : null;

      let status = "MATCH_ORIGINAL";

      if (onChainScaled !== originalScaled) {
        status = onChainScaled === mergedScaled
          ? "MATCHES_MERGED_ONLY"
          : "MATCHES_NEITHER_SHAPLEY_VALUE";
      }

      if (entry.flagged !== originalFlag) {
        status += "|FLAG_MISMATCH_ORIGINAL";
      }

      if (m && entry.flagged !== mergedFlag) {
        status += "|FLAG_MISMATCH_MERGED";
      }

      results.push({
        key,
        status,
        onChainScaled,
        originalScaled,
        mergedScaled,
        onChainFlag: entry.flagged,
        originalFlag,
        mergedFlag,
        onChainHash: entry.dataHash
      });
    } catch (err) {
      rpcErrors++;
      results.push({
        key,
        status: "RPC_OR_ENTRY_ERROR",
        error: String(err.shortMessage || err.message || err).slice(0, 250)
      });
      console.error(
        `Failed ${key}:`,
        String(err.shortMessage || err.message || err).slice(0, 150)
      );
    }

    // One request at a time, with a pause to reduce rate limiting.
    if ((i + 1) % 25 === 0 || i + 1 === byz.length) {
      console.log(`Processed ${i + 1}/${byz.length}`);
    }
    await sleep(150);
  }

  const counts = {};
  for (const r of results) {
    counts[r.status] = (counts[r.status] || 0) + 1;
  }

  const report = {
    network: hre.network.name,
    contractAddress: deployed.address,
    onChainTotalEntries: total,
    detectionRows: byz.length,
    shapleyRows: shapleyRows.length,
    mergedRows: merged.length,
    rpcOrEntryErrors: rpcErrors,
    statusCounts: counts,
    results
  };

  fs.writeFileSync(
    "./verification_comparison_report.json",
    JSON.stringify(report, null, 2)
  );

  console.log("\nStatus counts:");
  console.log(JSON.stringify(counts, null, 2));
  console.log("\nFull report saved to verification_comparison_report.json");
  console.log("No transactions were sent.");
}

main().catch(err => {
  console.error("Verification failed:", err);
  process.exitCode = 1;
});
