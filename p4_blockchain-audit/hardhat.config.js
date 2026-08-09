require("@nomicfoundation/hardhat-toolbox");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: { enabled: true, runs: 200 },
    },
  },
  networks: {
    hardhat: {
      // local, in-process chain — fast, deterministic, no faucet needed
    },
    // Uncomment and fill in to deploy to the real Sepolia testnet later.
    // Requires an RPC URL (Infura/Alchemy) and a funded test account.
    //
    // sepolia: {
    //   url: process.env.SEPOLIA_RPC_URL || "",
    //   accounts: process.env.SEPOLIA_PRIVATE_KEY ? [process.env.SEPOLIA_PRIVATE_KEY] : [],
    // },
  },
};
