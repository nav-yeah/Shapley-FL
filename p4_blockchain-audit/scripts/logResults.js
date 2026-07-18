const fs = require("fs");
const hre = require("hardhat");

// --- tiny CSV reader (files have simple comma-separated numeric columns,
//     no quoted/escaped fields, so a full CSV library isn't needed) ---
function parseCSV(filePath) {
  const raw = fs.readFileSync(filePath, "utf-8");
  const lines = raw.split(/\r?\n/).filter((l) => l.trim().length > 0);
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row = {};
    headers.forEach((h, i) => (row[h] = (cells[i] ?? "").trim()));
    return row;
  });
}

async function main() {
  const shapleyPath = process.env.SHAPLEY_CSV || "./data/shapley_scores.csv";
  const byzPath = process.env.BYZ_CSV || "./data/byzantine_detection_results.csv";

  const shapleyRows = parseCSV(shapleyPath);
  const byzRows = parseCSV(byzPath);
  const byzHasShapleyValue = byzRows.length > 0 && "shapley_value" in byzRows[0];

  // Index the baseline shapley file by (round, client_id) as a fallback source
  // for shapley_value when the BYZ file doesn't carry it itself.
  const shapleyIndex = {};
  for (const r of shapleyRows) {
    shapleyIndex[`${r.round}-${r.client_id}`] = r;
  }

  // The BYZ file is the driver: it defines exactly which (round, client_id)
  // pairs get logged. This works whether it's the full 30-client / 150-row
  // deliverable or a smaller scenario export (e.g. 10-client / 50-row).
  const rowsToLog = byzRows.map((byz) => {
    let shapleyValueStr = byzHasShapleyValue ? byz.shapley_value : undefined;
    if (shapleyValueStr === undefined || shapleyValueStr === "") {
      const fallback = shapleyIndex[`${byz.round}-${byz.client_id}`];
      shapleyValueStr = fallback ? fallback.shapley_value : "0";
    }
    return {
      round: byz.round,
      client_id: byz.client_id,
      shapley_value: shapleyValueStr,
      flagged: byz.flagged_status === "1",
    };
  });

  const missingShapley = rowsToLog.filter((r) => r.shapley_value === "0");
  if (missingShapley.length > 0) {
    console.warn(
      `Warning: ${missingShapley.length} row(s) had no shapley_value in either ` +
        `${byzPath} or ${shapleyPath} and were logged as 0. Check the CSVs cover ` +
        `the same (round, client_id) pairs.`
    );
  }

  const flaggedRowCount = rowsToLog.filter((r) => r.flagged).length;
  if (flaggedRowCount === 0) {
    console.warn(
      `Warning: every row in ${byzPath} has flagged_status=0. No AnomalyFlagged ` +
        `events will fire and every client will be reward-eligible. If you expected ` +
        `real flags, confirm you're pointing BYZ_CSV at the right file.`
    );
  } else {
    console.log(`${flaggedRowCount} of ${rowsToLog.length} rows are flagged in this run.`);
  }

  const { address } = JSON.parse(fs.readFileSync("./deployed-address.json", "utf-8"));
  const contract = await hre.ethers.getContractAt("ShapleyAudit", address);

  let totalGasUsed = 0n;
  let txCount = 0;
  const gasLog = [];

  for (const row of rowsToLog) {
    // Scale the float shapley_value into a fixed-point int256 (x1e18) since
    // Solidity has no native floating point.
    const scaledValue = BigInt(Math.round(parseFloat(row.shapley_value) * 1e18));

    const dataString = `${row.round}|${row.client_id}|${row.shapley_value}|${row.flagged}`;
    const dataHash = hre.ethers.keccak256(hre.ethers.toUtf8Bytes(dataString));

    const tx = await contract.logScore(row.round, row.client_id, scaledValue, row.flagged, dataHash);
    const receipt = await tx.wait();

    totalGasUsed += receipt.gasUsed;
    txCount += 1;
    gasLog.push({
      round: row.round,
      client_id: row.client_id,
      flagged: row.flagged,
      gasUsed: receipt.gasUsed.toString(),
    });

    if (row.flagged) {
      console.log(`AnomalyFlagged fired: round=${row.round} client=${row.client_id}`);
    }
  }

  const avgGas = txCount > 0 ? totalGasUsed / BigInt(txCount) : 0n;

  console.log(`\nLogged ${txCount} entries to ${address}`);
  console.log(`Total gas used: ${totalGasUsed.toString()}`);
  console.log(`Average gas per entry: ${avgGas.toString()}`);

  fs.writeFileSync(
    "./gas_report.json",
    JSON.stringify(
      {
        contractAddress: address,
        source: byzPath,
        txCount,
        totalGasUsed: totalGasUsed.toString(),
        avgGasPerEntry: avgGas.toString(),
        perEntry: gasLog,
      },
      null,
      2
    )
  );
  console.log("Gas report written to ./gas_report.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
