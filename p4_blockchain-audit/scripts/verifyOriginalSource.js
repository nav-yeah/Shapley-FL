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

async function main() {
  if (hre.network.name !== "sepolia") {
    throw new Error("Wrong network; expected Sepolia");
  }

  const byzPath = "./data/byzantine_detection_results.csv";
  const shapleyPath = "./data/shapley_scores.csv";
  const byzRows = parseCSV(byzPath);
  const shapleyRows = parseCSV(shapleyPath);

  const shapleyIndex = new Map(
    shapleyRows.map(r => [`${r.round}-${r.client_id}`, r])
  );

  const rows = byzRows.map(r => {
    const fallback = shapleyIndex.get(`${r.round}-${r.client_id}`);
    const shapley = r.shapley_value || fallback?.shapley_value;
    if (!shapley) {
      throw new Error(`Missing Shapley value: ${r.round}-${r.client_id}`);
    }
    return {
      round: r.round,
      client_id: r.client_id,
      shapley_value: shapley,
      flagged: r.flagged_status === "1"
    };
  });

  const { address } = JSON.parse(
    fs.readFileSync("./deployed-address.json", "utf8")
  );
  const contract = await hre.ethers.getContractAt("ShapleyAudit", address);
  const onChainEntries = Number(await contract.totalEntries());

  console.log("Contract:", address);
  console.log("Detection CSV rows:", byzRows.length);
  console.log("Shapley CSV rows:", shapleyRows.length);
  console.log("On-chain entries:", onChainEntries);

  const seen = new Set();
  const mismatches = [];
  let verifiedRows = 0;

  for (let start = 0; start < rows.length; start += 20) {
    const batch = rows.slice(start, start + 20);

    const results = await Promise.all(batch.map(async (r, j) => {
      const key = `${r.round}-${r.client_id}`;
      const line = start + j + 2;

      if (seen.has(key)) {
        return { line, key, errors: ["Duplicate CSV key"] };
      }
      seen.add(key);

      try {
        const scaled = BigInt(
          Math.round(parseFloat(r.shapley_value) * 1e18)
        );
        const dataString =
          `${r.round}|${r.client_id}|${r.shapley_value}|${r.flagged}`;
        const hash = hre.ethers.keccak256(
          hre.ethers.toUtf8Bytes(dataString)
        );

        const e = await contract.getEntry(r.round, r.client_id);
        const errors = [];

        if (e.round !== BigInt(r.round)) errors.push("round");
        if (e.clientId !== BigInt(r.client_id)) errors.push("client_id");
        if (e.shapleyValueScaled !== scaled) errors.push("Shapley value");
        if (e.flagged !== r.flagged) errors.push("flagged status");
        if (e.dataHash.toLowerCase() !== hash.toLowerCase()) {
          errors.push("data hash");
        }

        return errors.length ? { line, key, errors } : null;
      } catch (err) {
        return {
          line,
          key,
          errors: [err.shortMessage || err.message || String(err)]
        };
      }
    }));

    for (const result of results) {
      if (result) mismatches.push(result);
      else verifiedRows++;
    }

    console.log(`Checked ${Math.min(start + 20, rows.length)}/${rows.length}`);
  }

  const summary = {
    contractAddress: address,
    network: hre.network.name,
    detectionSource: byzPath,
    shapleySource: shapleyPath,
    csvRows: rows.length,
    onChainEntries,
    verifiedRows,
    mismatchCount: mismatches.length,
    mismatches: mismatches.slice(0, 100)
  };

  fs.writeFileSync(
    "./verification_original_source_summary.json",
    JSON.stringify(summary, null, 2)
  );

  console.log("\nVerification summary:");
  console.log(JSON.stringify({
    verifiedRows,
    mismatchCount: mismatches.length,
    report: "./verification_original_source_summary.json"
  }, null, 2));
}

main().catch(err => {
  console.error("Verification failed:", err);
  process.exitCode = 1;
});
