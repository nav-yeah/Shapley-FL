// Demonstrates dispute resolution using what's ALREADY in ShapleyAudit.sol --
// no new contract code needed. The mechanism: every logged entry stores a
// keccak256 hash of (round, client_id, shapley_value, flagged). To resolve a
// dispute, we take the client's CLAIMED value, recompute the hash the same
// way the contract did at logging time, and compare it against the immutable
// on-chain hash via getEntry(). Match = the on-chain record is proven to equal
// the claim. Mismatch = proof the record differs from the claim (the record
// itself, being immutable, cannot be forged after the fact -- this is what
// makes the "resolution" trustworthy instead of just one party's word).
//
// Run AFTER deploy.js + logResults.js have already populated the contract
// (same deployed-address.json used elsewhere).

const fs = require("fs");
const hre = require("hardhat");

function computeHash(round, clientId, shapleyValueStr, flagged) {
  const dataString = `${round}|${clientId}|${shapleyValueStr}|${flagged}`;
  return hre.ethers.keccak256(hre.ethers.toUtf8Bytes(dataString));
}

async function resolveDispute(contract, round, clientId, claimedShapleyValueStr, claimedFlagged, description) {
  console.log(`\n--- Dispute: ${description} ---`);
  console.log(`Client ${clientId} disputes round ${round}, claiming:`);
  console.log(`  shapley_value = ${claimedShapleyValueStr}, flagged = ${claimedFlagged}`);

  let onChainEntry;
  try {
    onChainEntry = await contract.getEntry(round, clientId);
  } catch (err) {
    console.log(`  RESULT: No entry found on-chain for round ${round}, client ${clientId}. Cannot resolve.`);
    return { round, clientId, resolved: false, reason: "no_entry" };
  }

  const onChainHash = onChainEntry.dataHash;
  const claimedHash = computeHash(round, clientId, claimedShapleyValueStr, claimedFlagged);

  console.log(`  On-chain hash:    ${onChainHash}`);
  console.log(`  Recomputed hash:  ${claimedHash} (from client's claimed value)`);

  const matches = onChainHash.toLowerCase() === claimedHash.toLowerCase();

  if (matches) {
    console.log(`  RESULT: MATCH. The on-chain record confirms the client's claim exactly.`);
    console.log(`  Dispute resolved: no discrepancy -- the logged entry is what the client says it is.`);
  } else {
    console.log(`  RESULT: MISMATCH. The claimed value does NOT match the immutable on-chain record.`);
    console.log(
      `  Dispute resolved: the record logged at block ${onChainEntry.blockNumber.toString()} ` +
        `(timestamp ${onChainEntry.timestamp.toString()}) stands as authoritative. ` +
        `The claim is rejected -- or, if the client believes the ORIGINAL logging pipeline made an ` +
        `error (not that the chain was tampered with), that is a separate off-chain investigation ` +
        `into the input CSV, since the on-chain hash can only prove what was actually submitted, ` +
        `not whether the submission itself was correct.`
    );
  }

  return { round, clientId, resolved: true, matches, onChainHash, claimedHash };
}

async function main() {
  const { address } = JSON.parse(fs.readFileSync("./deployed-address.json", "utf-8"));
  const contract = await hre.ethers.getContractAt("ShapleyAudit", address);
  console.log(`Connected to ShapleyAudit at ${address}\n`);
  console.log("Running two dispute scenarios against already-logged data:");

  const results = [];

  // Scenario A: client's claim matches what's actually on-chain.
  // (Adjust round/clientId/value below to match a real row from whatever
  // CSV was last logged via logResults.js, if you want this to reflect a
  // true positive-match case rather than a designed example.)
  const entryA = await contract.getEntry(1, 1);
  const shapleyValueAStr = (Number(entryA.shapleyValueScaled) / 1e18).toString();
  results.push(
    await resolveDispute(
      contract,
      1,
      1,
      shapleyValueAStr,
      entryA.flagged,
      "client claims the value that was actually logged"
    )
  );

  // Scenario B: client disputes with a DIFFERENT value than what's on-chain
  // (simulating either an honest disagreement or an attempted false claim).
  results.push(
    await resolveDispute(
      contract,
      1,
      1,
      "0.999", // deliberately wrong, to demonstrate the mismatch path
      false,
      "client claims a different value than what was logged"
    )
  );

  fs.writeFileSync("./dispute_resolution_report.json", JSON.stringify(results, null, 2));
  console.log("\nFull dispute log written to ./dispute_resolution_report.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
