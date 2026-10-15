// Same as logResults.js, but SAFE TO RE-RUN: before logging each row, it
// checks whether that (round, client_id) pair is already on-chain via
// getEntry(), and skips it if so. This lets a run that stopped partway
// (e.g. ran out of Sepolia test ETH) be resumed by just running this
// script again -- it will skip everything already logged and continue
// from the first un-logged row, instead of crashing on the duplicate-entry
// revert the way logResults.js would.

const fs = require("fs");
const hre = require("hardhat");

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

async function isAlreadyLogged(contract, round, clientId) {
  try {
    await contract.getEntry(round, clientId);
    return true; // no revert -- entry exists
  } catch (err) {
    return false; // reverted with "not found" -- not logged yet
  }
}

async function main() {
  const shapleyPath = process.env.SHAPLEY_CSV || "./data/shapley_scores.csv";
  const byzPath = process.env.BYZ_CSV || "./data/byzantine_detection_results.csv";
  // Optional cap on how many NEW rows to log this run -- useful on a real
  // network (Sepolia) where each row costs real (free, but faucet-limited)
  // test ETH, and you want to log a representative subset now and the rest
  // later once more funds are available, rather than running until funds
  // run out mid-transaction.
  const maxRows = process.env.MAX_ROWS ? parseInt(process.env.MAX_ROWS, 10) : Infinity;

  const shapleyRows = parseCSV(shapleyPath);
  const byzRows = parseCSV(byzPath);
  const byzHasShapleyValue = byzRows.length > 0 && "shapley_value" in byzRows[0];

  const shapleyIndex = {};
  for (const r of shapleyRows) {
    shapleyIndex[`${r.round}-${r.client_id}`] = r;
  }

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

  const { address } = JSON.parse(fs.readFileSync("./deployed-address.json", "utf-8"));
  const contract = await hre.ethers.getContractAt("ShapleyAudit", address);

  console.log(`Connected to ${address}. Checking which of ${rowsToLog.length} rows are already logged...`);
  if (maxRows !== Infinity) {
    console.log(`MAX_ROWS set: will log at most ${maxRows} NEW rows this run, then stop cleanly.\n`);
  } else {
    console.log("");
  }

  let totalGasUsed = 0n;
  let txCount = 0;
  let skippedCount = 0;
  const gasLog = [];

  for (let i = 0; i < rowsToLog.length; i++) {
    const row = rowsToLog[i];

    const alreadyLogged = await isAlreadyLogged(contract, row.round, row.client_id);
    if (alreadyLogged) {
      skippedCount += 1;
      continue;
    }

    if (txCount >= maxRows) {
      console.log(
        `\nMAX_ROWS limit (${maxRows}) reached -- stopping cleanly. ` +
          `${txCount} new rows logged this run, ${skippedCount} already logged, ` +
          `${rowsToLog.length - txCount - skippedCount} still remaining.`
      );
      console.log("Get more test ETH and re-run this same script (with or without a new MAX_ROWS) to continue.");
      break;
    }

    const scaledValue = BigInt(Math.round(parseFloat(row.shapley_value) * 1e18));
    const dataString = `${row.round}|${row.client_id}|${row.shapley_value}|${row.flagged}`;
    const dataHash = hre.ethers.keccak256(hre.ethers.toUtf8Bytes(dataString));

    try {
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

      // Periodic progress update so a long Sepolia run isn't silent.
      if ((txCount + skippedCount) % 25 === 0) {
        console.log(
          `Progress: ${txCount + skippedCount}/${rowsToLog.length} processed ` +
            `(${txCount} logged this run, ${skippedCount} already logged, skipped)`
        );
      }
    } catch (err) {
      const msg = err.shortMessage || err.message || String(err);
      if (msg.includes("insufficient funds")) {
        console.error(
          `\nRan out of funds after logging ${txCount} new rows this run ` +
            `(${skippedCount} were already logged from a previous run).`
        );
        console.error("Top up the wallet from a faucet, then re-run this same script -- it will resume safely.");
        break;
      } else {
        console.error(`\nUnexpected error on round=${row.round} client=${row.client_id}:`, msg);
        break;
      }
    }
  }

  const avgGas = txCount > 0 ? totalGasUsed / BigInt(txCount) : 0n;

  console.log(`\n--- This run ---`);
  console.log(`Newly logged: ${txCount}`);
  console.log(`Already logged (skipped): ${skippedCount}`);
  console.log(`Remaining un-logged: ${rowsToLog.length - txCount - skippedCount}`);
  console.log(`Gas used this run: ${totalGasUsed.toString()}`);
  if (txCount > 0) console.log(`Average gas per new entry: ${avgGas.toString()}`);

  const reportPath = "./gas_report_sepolia.json";
  let existingReport = { perEntry: [] };
  if (fs.existsSync(reportPath)) {
    try {
      existingReport = JSON.parse(fs.readFileSync(reportPath, "utf-8"));
    } catch (e) {
      // ignore, start fresh
    }
  }
  const mergedPerEntry = [...(existingReport.perEntry || []), ...gasLog];
  const mergedTotalGas = mergedPerEntry.reduce((sum, e) => sum + BigInt(e.gasUsed), 0n);

  fs.writeFileSync(
    reportPath,
    JSON.stringify(
      {
        contractAddress: address,
        network: hre.network.name,
        source: byzPath,
        totalRowsInSource: rowsToLog.length,
        totalLoggedSoFar: mergedPerEntry.length,
        totalGasUsedAllRuns: mergedTotalGas.toString(),
        perEntry: mergedPerEntry,
      },
      null,
      2
    )
  );
  console.log(`\nCumulative gas report written to ${reportPath}`);
  console.log(`Total logged across all runs so far: ${mergedPerEntry.length} / ${rowsToLog.length}`);

  if (txCount + skippedCount < rowsToLog.length && txCount > 0) {
    console.log("\nRun this script again to continue logging the remaining rows.");
  } else if (mergedPerEntry.length === rowsToLog.length) {
    console.log("\nAll rows logged. Done.");
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});