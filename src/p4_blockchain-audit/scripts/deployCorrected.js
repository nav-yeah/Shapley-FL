const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const outputPath = path.resolve("./deployed-address-corrected.json");

  if (fs.existsSync(outputPath)) {
    throw new Error(
      "deployed-address-corrected.json already exists. Refusing to overwrite it."
    );
  }

  const [deployer] = await hre.ethers.getSigners();
  console.log("Network:", hre.network.name);
  console.log("Deployer:", deployer.address);

  const Factory = await hre.ethers.getContractFactory("ShapleyAudit");
  const contract = await Factory.deploy();
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  const deploymentTx = contract.deploymentTransaction();
  const receipt = deploymentTx ? await deploymentTx.wait() : null;

  const record = {
    address,
    network: hre.network.name,
    deployer: deployer.address,
    deploymentTx: deploymentTx ? deploymentTx.hash : null,
    deploymentBlock: receipt ? receipt.blockNumber : null,
    deployedAt: new Date().toISOString()
  };

  fs.writeFileSync(outputPath, JSON.stringify(record, null, 2) + "\n");

  console.log("Corrected contract deployed to:", address);
  console.log("Deployment record saved to:", outputPath);
  console.log("Original deployed-address.json was not modified.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
