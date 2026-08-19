// Generates synthetic (round, client_id, shapley_value, flagged_status, ...) CSVs
// at three sizes, purely for gas-cost-vs-scale benchmarking. Values are randomized
// but shaped exactly like the real byzantine_detection_results_merged.csv schema,
// so logResults.js can consume these files with zero code changes.
//
// Row counts chosen to match the real future dataset shape (30 clients):
//   150  rows = 30 clients x 5 rounds   (matches the current N_ROUNDS=5 baseline)
//   600  rows = 30 clients x 20 rounds  (midpoint scaling step)
//   1500 rows = 30 clients x 50 rounds  (matches the planned N_ROUNDS=50 rerun)
//
// This is synthetic data ONLY for measuring gas cost vs. row count -- it does not
// represent real Shapley values or real detection results. Once P3 delivers the
// real 1500-row byzantine_detection_results_merged.csv, re-run the pipeline against
// that instead (see p4.md).

const fs = require("fs");
const path = require("path");

const N_CLIENTS = 30;
const CONFIGS = [
  { rounds: 5, rows: 150, outFile: "synthetic_150.csv" },
  { rounds: 20, rows: 600, outFile: "synthetic_600.csv" },
  { rounds: 50, rows: 1500, outFile: "synthetic_1500.csv" },
];

const OUT_DIR = process.env.SYNTHETIC_DATA_DIR || path.join(__dirname, "..", "data");

function seededRandom(seed) {
  // simple deterministic PRNG so re-runs are reproducible
  let s = seed;
  return function () {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

function generateCSV(rounds, expectedRows, outPath) {
  const rand = seededRandom(42);
  const header = "round,client_id,shapley_value,flagged_status,rolling_variance,trend_slope,z_score";
  const lines = [header];

  // Flag roughly 10% of clients as "attackers" for the whole run, consistent
  // per-client (not per-row), same shape as a real sustained-detection result.
  const flaggedClients = new Set();
  for (let c = 1; c <= N_CLIENTS; c++) {
    if (rand() < 0.10) flaggedClients.add(c);
  }

  for (let round = 1; round <= rounds; round++) {
    for (let client = 1; client <= N_CLIENTS; client++) {
      const shapleyValue = (rand() * 0.06 - 0.02).toFixed(17); // roughly matches real range
      const isFlagged = flaggedClients.has(client) && round >= 2 ? 1 : 0;
      const rollingVariance = round < 3 ? "" : (rand() * 0.001).toFixed(10);
      const trendSlope = round < 3 ? "" : (rand() * 2 - 1).toFixed(10);
      const zScore = (rand() * 4 - 2).toFixed(10);
      lines.push(`${round},${client},${shapleyValue},${isFlagged},${rollingVariance},${trendSlope},${zScore}`);
    }
  }

  fs.writeFileSync(outPath, lines.join("\n") + "\n");
  const actualRows = lines.length - 1;
  if (actualRows !== expectedRows) {
    console.warn(`Warning: ${outPath} has ${actualRows} rows, expected ${expectedRows}`);
  }
  return actualRows;
}

function main() {
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

  console.log("Generating synthetic gas-benchmark datasets...\n");
  for (const cfg of CONFIGS) {
    const outPath = path.join(OUT_DIR, cfg.outFile);
    const rows = generateCSV(cfg.rounds, cfg.rows, outPath);
    console.log(`  ${cfg.outFile}: ${rows} rows (${N_CLIENTS} clients x ${cfg.rounds} rounds) -> ${outPath}`);
  }
  console.log("\nDone. These are synthetic values for gas benchmarking only --");
  console.log("not real Shapley scores or real detection results.");
}

main();
