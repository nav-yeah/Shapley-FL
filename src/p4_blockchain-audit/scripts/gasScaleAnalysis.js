// Deploys a FRESH ShapleyAudit contract for each dataset size (150 / 600 / 1500
// rows), logs every row, and records total + average gas per size. Fresh
// deployments per size keep measurements independent -- avoids any risk of
// duplicate (round, client_id) rejections between differently-sized runs and
// keeps each measurement a clean, isolated benchmark.
//
// Run AFTER generateSyntheticData.js. Requires a running local Hardhat node
// (npx hardhat node) same as the normal deploy/log/demo flow.

const fs = require("fs");
const path = require("path");
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

async function logDataset(csvPath, label) {
  const rows = parseCSV(csvPath);

  const ShapleyAudit = await hre.ethers.getContractFactory("ShapleyAudit");
  const contract = await ShapleyAudit.deploy();
  await contract.waitForDeployment();
  const address = await contract.getAddress();

  let totalGasUsed = 0n;
  let flaggedCount = 0;

  const deployTx = contract.deploymentTransaction();
  const deployReceipt = await deployTx.wait();
  const deploymentGas = deployReceipt.gasUsed;

  for (const row of rows) {
    const flagged = row.flagged_status === "1";
    if (flagged) flaggedCount += 1;

    const scaledValue = BigInt(Math.round(parseFloat(row.shapley_value) * 1e18));
    const dataString = `${row.round}|${row.client_id}|${row.shapley_value}|${flagged}`;
    const dataHash = hre.ethers.keccak256(hre.ethers.toUtf8Bytes(dataString));

    const tx = await contract.logScore(row.round, row.client_id, scaledValue, flagged, dataHash);
    const receipt = await tx.wait();
    totalGasUsed += receipt.gasUsed;
  }

  const avgGas = rows.length > 0 ? totalGasUsed / BigInt(rows.length) : 0n;

  return {
    label,
    csvPath,
    rowCount: rows.length,
    flaggedCount,
    contractAddress: address,
    deploymentGas: deploymentGas.toString(),
    totalGasUsed: totalGasUsed.toString(),
    avgGasPerEntry: avgGas.toString(),
    totalGasIncludingDeployment: (totalGasUsed + deploymentGas).toString(),
  };
}

function printBarChart(results) {
  console.log("\nGas cost vs. dataset size (terminal bar chart, logging gas only):\n");
  const maxGas = Math.max(...results.map((r) => Number(r.totalGasUsed)));
  const barWidth = 50;
  for (const r of results) {
    const barLen = Math.round((Number(r.totalGasUsed) / maxGas) * barWidth);
    const bar = "#".repeat(barLen).padEnd(barWidth, " ");
    console.log(
      `  ${String(r.rowCount).padStart(5)} rows | ${bar} | ${Number(r.totalGasUsed).toLocaleString()} gas`
    );
  }
}

function printScalingCheck(results) {
  console.log("\nLinearity check (does gas scale linearly with row count, as expected?):\n");
  const base = results[0];
  for (const r of results) {
    const rowRatio = r.rowCount / base.rowCount;
    const gasRatio = Number(r.totalGasUsed) / Number(base.totalGasUsed);
    const deviation = (((gasRatio - rowRatio) / rowRatio) * 100).toFixed(2);
    console.log(
      `  ${r.rowCount} rows: ${rowRatio.toFixed(2)}x the rows of the ${base.rowCount}-row run, ` +
        `${gasRatio.toFixed(2)}x the gas (${deviation}% deviation from perfectly linear)`
    );
  }
}

async function main() {
  const dataDir = process.env.SYNTHETIC_DATA_DIR || path.join(__dirname, "..", "data");
  const datasets = [
    { file: "synthetic_150.csv", label: "150 rows (30 clients x 5 rounds)" },
    { file: "synthetic_600.csv", label: "600 rows (30 clients x 20 rounds)" },
    { file: "synthetic_1500.csv", label: "1500 rows (30 clients x 50 rounds)" },
  ];

  const results = [];
  for (const ds of datasets) {
    const csvPath = path.join(dataDir, ds.file);
    if (!fs.existsSync(csvPath)) {
      console.error(`Missing ${csvPath}. Run generateSyntheticData.js first.`);
      process.exitCode = 1;
      return;
    }
    console.log(`Logging ${ds.label}...`);
    const result = await logDataset(csvPath, ds.label);
    results.push(result);
    console.log(
      `  -> deployed at ${result.contractAddress}, ` +
        `total gas (logging only): ${Number(result.totalGasUsed).toLocaleString()}, ` +
        `avg/entry: ${result.avgGasPerEntry}\n`
    );
  }

  printBarChart(results);
  printScalingCheck(results);

  const outPath = path.join(__dirname, "..", "gas_scale_report.json");
  fs.writeFileSync(outPath, JSON.stringify({ generatedAt: new Date().toISOString(), results }, null, 2));

  // Also write a plain CSV summary, easy to paste into Excel/Sheets for a real chart.
  const csvOutPath = path.join(__dirname, "..", "gas_scale_report.csv");
  const csvLines = [
    "row_count,flagged_count,deployment_gas,total_logging_gas,avg_gas_per_entry,total_gas_including_deployment",
  ];
  for (const r of results) {
    csvLines.push(
      `${r.rowCount},${r.flaggedCount},${r.deploymentGas},${r.totalGasUsed},${r.avgGasPerEntry},${r.totalGasIncludingDeployment}`
    );
  }
  fs.writeFileSync(csvOutPath, csvLines.join("\n") + "\n");

  console.log(`\nFull results written to ${outPath}`);
  console.log(`Spreadsheet-ready summary written to ${csvOutPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
