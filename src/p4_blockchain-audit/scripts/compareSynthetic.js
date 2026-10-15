const fs = require("fs");
const path = require("path");
const hre = require("hardhat");

const ROOT = path.resolve(__dirname, "..");

function csv(file) {
  const text = fs.readFileSync(path.join(ROOT, file), "utf8")
    .replace(/^\uFEFF/, "").trim();
  const [header, ...lines] = text.split(/\r?\n/);
  const keys = header.split(",");
  return lines.filter(Boolean).map(line =>
    Object.fromEntries(keys.map((k, i) => [k.trim(), (line.split(",")[i] || "").trim()]))
  );
}

async function main() {
  const report = JSON.parse(fs.readFileSync(
    path.join(ROOT, "verification_original_source_summary.json"), "utf8"
  ));
  const mismatches = report.mismatches || [];
  const source = new Map(
    csv("data/synthetic_1500.csv").map(r => [`${r.round}-${r.client_id}`, r])
  );

  const contract = await hre.ethers.getContractAt(
    "ShapleyAudit",
    "0xA7694F641937687Af7D09a314Ee0fb828c1F78Fe"
  );

  let valueMatches = 0;
  let hashMatches = 0;
  let checked = 0;

  for (const m of mismatches) {
    const [round, clientId] = m.key.split("-");
    const r = source.get(m.key);
    if (!r) continue;

    const e = await contract.getEntry(round, clientId);
    const scaled = BigInt(Math.round(Number(r.shapley_value) * 1e18));
    const flagged = r.flagged_status === "1";
    const input = `${round}|${clientId}|${r.shapley_value}|${flagged}`;
    const hash = hre.ethers.keccak256(hre.ethers.toUtf8Bytes(input));

    if (scaled.toString() === e.shapleyValueScaled.toString()) valueMatches++;
    if (hash.toLowerCase() === e.dataHash.toLowerCase()) hashMatches++;
    checked++;
  }

  console.log(JSON.stringify({
    mismatchesFromOriginalCSV: mismatches.length,
    syntheticRowsFoundAndChecked: checked,
    syntheticValueMatches: valueMatches,
    syntheticHashMatches: hashMatches
  }, null, 2));
}

main().catch(err => {
  console.error(err);
  process.exitCode = 1;
});
