const fs = require("fs");
const hre = require("hardhat");

function parseCSV(filePath) {
  const lines = fs.readFileSync(filePath, "utf8")
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .filter(line => line.trim().length > 0);

  if (lines.length < 2) throw new Error(`CSV is empty: ${filePath}`);

  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).map((line, i) => {
    const cells = line.split(",");
    const row = {};
    headers.forEach((h, j) => row[h] = (cells[j] ?? "").trim());
    if (cells.length !== headers.length) {
      throw new Error(`Malformed CSV row ${i + 2} in ${filePath}`);
    }
    return row;
  });
}

function key(row) {
  return `${row.round}-${row.client_id}`;
}

async function main() {
  const addressFile = "./deployed-address-corrected.json";
  const shapleyPath = process.env.SHAPLEY_CSV || "./data/shapley_scores.csv";
  const byzPath = process.env.BYZ_CSV || "./data/byzantine_detection_results.csv";
  const reportPath = "./gas_report_corrected_sepolia.json";
  const maxRows = process.env.MAX_ROWS
    ? Number.parseInt(process.env.MAX_ROWS, 10)
    : 1500;

  if (!Number.isInteger(maxRows) || maxRows < 1) {
    throw new Error("MAX_ROWS must be a positive integer.");
  }
  if (!fs.existsSync(addressFile)) {
    throw new Error(`Missing ${addressFile}; refusing to use the old deployment.`);
  }

  const { address, network } = JSON.parse(fs.readFileSync(addressFile, "utf8"));
  if (!address || network !== hre.network.name) {
    throw new Error(`Deployment network mismatch. File says ${network}; active network is ${hre.network.name}.`);
  }

  const shapleyRows = parseCSV(shapleyPath);
  const byzRows = parseCSV(byzPath);

  if (shapleyRows.length !== 1500 || byzRows.length !== 1500) {
    throw new Error(`Expected 1,500 rows in each CSV. Found Shapley=${shapleyRows.length}, detection=${byzRows.length}.`);
  }

  const shapleyIndex = new Map();
  for (const row of shapleyRows) {
    const k = key(row);
    if (shapleyIndex.has(k)) throw new Error(`Duplicate Shapley key: ${k}`);
    if (!Number.isFinite(Number(row.shapley_value))) {
      throw new Error(`Invalid Shapley value for ${k}`);
    }
    shapleyIndex.set(k, row);
  }

  const seen = new Set();
  const rows = byzRows.map(byz => {
    const k = key(byz);
    if (seen.has(k)) throw new Error(`Duplicate detection key: ${k}`);
    seen.add(k);

    const shapley = shapleyIndex.get(k);
    if (!shapley) throw new Error(`No matching Shapley row for ${k}`);
    if (!["0", "1"].includes(byz.flagged_status)) {
      throw new Error(`Invalid flagged_status for ${k}: ${byz.flagged_status}`);
    }
    if (!/^\d+$/.test(byz.round) || !/^\d+$/.test(byz.client_id)) {
      throw new Error(`Invalid round/client ID for ${k}`);
    }

    return {
      round: byz.round,
      client_id: byz.client_id,
      shapley_value: shapley.shapley_value,
      flagged: byz.flagged_status === "1"
    };
  });

  if (seen.size !== 1500 || shapleyIndex.size !== 1500) {
    throw new Error("Source keys do not form exactly 1,500 unique entries.");
  }

  let report = { perEntry: [] };
  if (fs.existsSync(reportPath)) {
    report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
    if (report.contractAddress !== address) {
      throw new Error("Gas report belongs to a different contract; refusing to mix reports.");
    }
  }

  const [signer] = await hre.ethers.getSigners();
  const contract = await hre.ethers.getContractAt("ShapleyAudit", address, signer);
  const initialCount = Number(await contract.totalEntries());

  if (initialCount > 1500) {
    throw new Error(`New contract unexpectedly has ${initialCount} entries.`);
  }

  console.log(`Network: ${hre.network.name}`);
  console.log(`Corrected contract: ${address}`);
  console.log(`Source validation: ${rows.length} unique matched rows`);
  console.log(`On-chain entries before run: ${initialCount}`);
  console.log(`Maximum new entries this run: ${maxRows}`);

  let logged = 0;
  let skipped = 0;
  let gasUsed = 0n;
  const gasLog = [];

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    let exists = false;

    try {
      await contract.getEntry(row.round, row.client_id);
      exists = true;
    } catch (err) {
      const message = `${err.shortMessage || ""} ${err.reason || ""} ${err.message || ""}`;
      const expectedMissing = /not found|does not exist|no entry|entry not found|invalid entry/i.test(message);
      if (!expectedMissing) {
        throw new Error(`Could not safely check ${key(row)}. Stopping rather than risking a duplicate: ${message}`);
      }
    }

    if (exists) {
      skipped++;
      continue;
    }

    if (logged >= maxRows) {
      console.log(`Run limit reached. ${logged} logged, ${skipped} skipped.`);
      break;
    }

    const scaledValue = BigInt(Math.round(Number(row.shapley_value) * 1e18));
    const dataString = `${row.round}|${row.client_id}|${row.shapley_value}|${row.flagged}`;
    const dataHash = hre.ethers.keccak256(hre.ethers.toUtf8Bytes(dataString));

    const tx = await contract.logScore(
      row.round, row.client_id, scaledValue, row.flagged, dataHash
    );
    const receipt = await tx.wait();

    logged++;
    gasUsed += receipt.gasUsed;
    gasLog.push({
      round: row.round,
      client_id: row.client_id,
      gasUsed: receipt.gasUsed.toString(),
      txHash: receipt.hash
    });

    if ((logged + skipped) % 25 === 0 || logged === maxRows) {
      console.log(`Progress: ${logged + skipped}/1500 processed; ${logged} newly logged.`);
    }
  }

  const prior = report.perEntry || [];
  const combined = [...prior, ...gasLog];
  report = {
    contractAddress: address,
    network: hre.network.name,
    sourceShapley: shapleyPath,
    sourceDetection: byzPath,
    totalRowsInSource: rows.length,
    measuredGasEntries: combined.length,
    totalGasUsedMeasured: combined.reduce((sum, e) => sum + BigInt(e.gasUsed), 0n).toString(),
    perEntry: combined
  };
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

  const finalCount = Number(await contract.totalEntries());
  console.log("\n--- Logger summary ---");
  console.log(`New entries logged this run: ${logged}`);
  console.log(`Existing entries skipped: ${skipped}`);
  console.log(`Gas used this run: ${gasUsed}`);
  console.log(`Contract totalEntries(): ${finalCount}`);
  console.log(`Gas report: ${reportPath}`);

  if (finalCount !== 1500) {
    console.log("Logging is not complete yet. Resolve any issue and rerun this script to resume.");
  } else {
    console.log("All 1,500 entries are present. Run the verifier before declaring success.");
  }
}

main().catch(err => {
  console.error("\nLOGGER STOPPED:", err.shortMessage || err.message || err);
  process.exitCode = 1;
});
