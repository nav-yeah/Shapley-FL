const fs = require("fs");
const hre = require("hardhat");

async function main() {
  const readCSV = (p) => {
    const lines = fs.readFileSync(p, "utf8").replace(/^\uFEFF/, "").trim().split(/\r?\n/);
    const headers = lines[0].split(",").map(x => x.trim());
    return lines.slice(1).map(line =>
      Object.fromEntries(headers.map((h, i) => [h, (line.split(",")[i] || "").trim()]))
    );
  };

  const s = readCSV("./data/shapley_scores.csv").find(r => r.round === "1" && r.client_id === "1");
  const b = readCSV("./data/byzantine_detection_results.csv").find(r => r.round === "1" && r.client_id === "1");

  const c = await hre.ethers.getContractAt(
    "ShapleyAudit",
    "0xA7694F641937687Af7D09a314Ee0fb828c1F78Fe"
  );
  const e = await c.getEntry(1, 1);

  const flagged = b.flagged_status === "1";
  const scaled = BigInt(Math.round(Number(s.shapley_value) * 1e18));
  const input = `1|1|${s.shapley_value}|${flagged}`;
  const hash = hre.ethers.keccak256(hre.ethers.toUtf8Bytes(input));

  console.log(JSON.stringify({
    sourceShapley: s.shapley_value,
    expectedScaled: scaled.toString(),
    onChainScaled: e.shapleyValueScaled.toString(),
    expectedFlagged: flagged,
    onChainFlagged: e.flagged,
    hashInput: input,
    expectedHash: hash,
    onChainHash: e.dataHash,
    scaledMatches: scaled.toString() === e.shapleyValueScaled.toString(),
    flaggedMatches: flagged === e.flagged,
    hashMatches: hash.toLowerCase() === e.dataHash.toLowerCase()
  }, null, 2));
}

main().catch(console.error);
