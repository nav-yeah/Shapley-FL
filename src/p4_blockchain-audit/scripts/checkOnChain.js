const hre = require("hardhat");

async function main() {
  const { address } = JSON.parse(
    require("fs").readFileSync("./deployed-address.json", "utf8")
  );

  const contract = await hre.ethers.getContractAt("ShapleyAudit", address);
  console.log("Network:", hre.network.name);
  console.log("Contract:", address);
  console.log("Owner:", await contract.owner());
  console.log("On-chain total entries:", (await contract.totalEntries()).toString());
}

main().catch(err => {
  console.error(err);
  process.exitCode = 1;
});
