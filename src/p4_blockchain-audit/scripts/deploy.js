const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const ShapleyAudit = await hre.ethers.getContractFactory("ShapleyAudit");
  const contract = await ShapleyAudit.deploy();
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  console.log("ShapleyAudit deployed to:", address);

  fs.writeFileSync(
    "./deployed-address.json",
    JSON.stringify({ address, network: hre.network.name }, null, 2)
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
