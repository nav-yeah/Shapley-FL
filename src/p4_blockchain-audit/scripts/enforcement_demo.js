const fs = require("fs");
const hre = require("hardhat");

// Same tiny CSV reader as logResults.js
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
  const { address } = JSON.parse(fs.readFileSync("./deployed-address.json", "utf-8"));
  const contract = await hre.ethers.getContractAt("ShapleyAudit", address);

  const total = await contract.totalEntries();
  console.log(`Contract has ${total.toString()} logged entries.\n`);

  // Derive the client ID list from whatever was actually logged, rather than
  // assuming a fixed 30 — this way the demo works correctly whether the run
  // used the full 30-client deliverable or a smaller scenario export.
  const byzPath = process.env.BYZ_CSV || "./data/byzantine_detection_results.csv";
  const byzRows = parseCSV(byzPath);
  const clientIds = [...new Set(byzRows.map((r) => Number(r.client_id)))].sort((a, b) => a - b);

  const roundsPerClient = {};
  for (const r of byzRows) {
    roundsPerClient[r.client_id] = (roundsPerClient[r.client_id] || 0) + 1;
  }
  const maxRounds = Math.max(...Object.values(roundsPerClient));
  // Require full clean history (all logged rounds clean) rather than a fixed
  // constant, so this scales with whatever dataset was actually used.
  const MIN_CLEAN_ROUNDS = maxRounds;

  console.log(
    `Reward eligibility (never flagged AND >= ${MIN_CLEAN_ROUNDS} clean rounds logged, ` +
      `derived from ${clientIds.length} clients found in ${byzPath}):\n`
  );
  console.log("client_id | flaggedCount | cleanCount | eligible");
  console.log("----------|--------------|------------|---------");

  const results = [];
  for (const clientId of clientIds) {
    const flaggedCount = await contract.flaggedCount(clientId);
    const cleanCount = await contract.cleanCount(clientId);
    const eligible = await contract.isRewardEligible(clientId, MIN_CLEAN_ROUNDS);

    results.push({
      client_id: clientId,
      flaggedCount: flaggedCount.toString(),
      cleanCount: cleanCount.toString(),
      eligible,
    });

    console.log(
      `${String(clientId).padEnd(9)} | ${flaggedCount.toString().padEnd(12)} | ` +
        `${cleanCount.toString().padEnd(10)} | ${eligible}`
    );
  }

  const eligibleCount = results.filter((r) => r.eligible).length;
  const flaggedTotal = results.filter((r) => r.flaggedCount !== "0").length;

  console.log(`\n${eligibleCount} / ${clientIds.length} clients are currently reward-eligible.`);
  console.log(`${flaggedTotal} / ${clientIds.length} clients have at least one flagged round.`);

  if (flaggedTotal === 0) {
    console.log(
      "\nNote: no client has any flagged rounds in this run, so this check is not yet " +
        "demonstrating the rejection path. Point BYZ_CSV at a dataset with real flags " +
        "(e.g. byzantine_scenario4_flagged.csv) to see ineligible clients."
    );
  }

  fs.writeFileSync("./enforcement_report.json", JSON.stringify(results, null, 2));
  console.log("\nFull report written to ./enforcement_report.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
