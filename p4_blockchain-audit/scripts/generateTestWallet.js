const { ethers } = require("ethers");

const wallet = ethers.Wallet.createRandom();

console.log("=================================");
console.log("TEST WALLET GENERATED");
console.log("=================================");
console.log("Address:");
console.log(wallet.address);

console.log("\nPrivate Key:");
console.log(wallet.privateKey);

console.log("\nDO NOT SHARE THE PRIVATE KEY.");
console.log("=================================");