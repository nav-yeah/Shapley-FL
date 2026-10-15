const fs = require("fs");
const hre = require("hardhat");

// NOTE: this script compares on-chain entries against the MERGED source CSV.
// Its output is NOT the corrected verification result. That is recorded in
// verification_summary_corrected.json. Output goes to a separate file so
// this script never overwrites the corrected summary.
const OUTPUT_FILE = "./verification_merged_source_summary.json";

function parseCSV(file) {
  const lines = fs.readFileSync(file, "utf8")
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .filter(line => line.trim());

  const headers = lines[0].split(",").map(x => x.trim());

  return lines.slice(1).map(line => {
    const cells = line.split(",");
    const row = {};
    headers.forEach((h, i) => row[h] = (cells[i] ?? "").trim());
    return row;
  });
}

async function main() {
  const source = require("path").join(__dirname, "..", "..", "byzantine_detection_results_merged.csv");
  const rows = parseCSV(source);
  const { address } = JSON.parse(
    fs.readFileSync("./deployed-address.json", "utf8")
  );
  const contract = await hre.ethers.getContractAt("ShapleyAudit", address);

  if (hre.network.name !== "sepolia") throw Error("Wrong network");
  if (rows.length !== 1500) throw Error(`Expected 1500 rows; got ${rows.length}`);

  console.log("Contract:", address);
  console.log("CSV rows:", rows.length);
  console.log("On-chain entries:", (await contract.totalEntries()).toString());
  console.log("Checking records directly; no blockchain-wide scan.");

  const mismatches = [];
  const verified = [];
  const seen = new Set();

  for (let start = 0; start < rows.length; start += 25) {
    const batch = rows.slice(start, start + 25);

    const results = await Promise.all(batch.map(async (r, j) => {
      const line = start + j + 2;
      const key = `${r.round}-${r.client_id}`;

      try {
        if (!r.round || !r.client_id || r.shapley_value === "" ||
            !["0", "1"].includes(r.flagged_status)) {
          throw Error("Invalid required CSV field");
        }

        if (seen.has(key)) throw Error("Duplicate CSV round/client pair");
        seen.add(key);

        const n = Number(r.shapley_value);
        if (!Number.isFinite(n)) throw Error("Invalid Shapley value");

        const scaled = BigInt(Math.round(n * 1e18));
        const flagged = r.flagged_status === "1";
        const dataString =
          `${r.round}|${r.client_id}|${r.shapley_value}|${flagged}`;
        const expectedHash = hre.ethers.keccak256(
          hre.ethers.toUtf8Bytes(dataString)
        );

        const e = await contract.getEntry(r.round, r.client_id);
        const errors = [];

        if (e.round !== BigInt(r.round)) errors.push("round");
        if (e.clientId !== BigInt(r.client_id)) errors.push("client_id");
        if (e.shapleyValueScaled !== scaled) errors.push("Shapley value");
        if (e.flagged !== flagged) errors.push("flagged status");
        if (e.dataHash.toLowerCase() !== expectedHash.toLowerCase())
          errors.push("data hash");

        if (errors.length) return { mismatch: { line, key, errors } };

        return {
          entry: {
            round: r.round,
            client_id: r.client_id,
            flagged
          }
        };
      } catch (e) {
        return {
          mismatch: {
            line, key,
            errors: [e.shortMessage || e.message || String(e)]
          }
        };
      }
    }));

    for (const result of results) {
      if (result.mismatch) mismatches.push(result.mismatch);
      else verified.push(result.entry);
    }

    console.log(`Checked ${Math.min(start + 25, rows.length)}/${rows.length}`);
  }

  const output = {
    contractAddress: address,
    network: hre.network.name,
    source,
    csvRows: rows.length,
    onChainEntries: (await contract.totalEntries()).toString(),
    verifiedRows: verified.length,
    mismatchCount: mismatches.length,
    mismatches: mismatches.slice(0, 100)
  };

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(output, null, 2));

  if (mismatches.length || verified.length !== 1500) {
    console.log(`VERIFICATION FAILED. See ${OUTPUT_FILE}`);
    return;
  }

  console.log("SUCCESS: all 1500 rows match the on-chain records.");
  console.log(`Summary saved to ${OUTPUT_FILE}`);
  console.log("This script does not submit transactions.");
}

main().catch(e => {
  console.error("Verification error:", e);
  process.exitCode = 1;
});
