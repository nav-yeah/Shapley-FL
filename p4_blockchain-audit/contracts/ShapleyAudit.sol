// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title ShapleyAudit
/// @notice Immutable on-chain audit trail for per-round Shapley contribution
///         scores in a federated learning run, plus Byzantine-flag enforcement.
/// @dev Values are logged by an authorized writer (the pipeline / oracle),
///      not computed on-chain. The contract's job is tamper-evident storage
///      and the AnomalyFlagged event + reward-eligibility check.
contract ShapleyAudit {
    struct ScoreEntry {
        uint256 round;
        uint256 clientId;
        int256 shapleyValueScaled; // shapley_value * 1e18, fixed-point (int256 since values can be negative)
        bool flagged;              // mirrors P3's flagged_status
        bytes32 dataHash;          // keccak256 of the raw row, for tamper-evidence
        uint256 timestamp;
        uint256 blockNumber;
    }

    ScoreEntry[] public entries;

    // round => clientId => 1-based index into entries (0 = not logged)
    mapping(uint256 => mapping(uint256 => uint256)) private entryIndex;

    // Running per-client tallies, used for reward-eligibility enforcement
    mapping(uint256 => uint256) public flaggedCount;
    mapping(uint256 => uint256) public cleanCount;

    address public owner;

    event ScoreLogged(
        uint256 indexed round,
        uint256 indexed clientId,
        int256 shapleyValueScaled,
        bool flagged,
        bytes32 dataHash
    );

    event AnomalyFlagged(
        uint256 indexed round,
        uint256 indexed clientId,
        int256 shapleyValueScaled,
        bytes32 dataHash
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "ShapleyAudit: not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    /// @notice Log one client's score for one round. Reverts on duplicate
    ///         (round, clientId) pairs so history can't be silently overwritten.
    function logScore(
        uint256 round,
        uint256 clientId,
        int256 shapleyValueScaled,
        bool flagged,
        bytes32 dataHash
    ) external onlyOwner {
        require(entryIndex[round][clientId] == 0, "ShapleyAudit: already logged");

        entries.push(
            ScoreEntry({
                round: round,
                clientId: clientId,
                shapleyValueScaled: shapleyValueScaled,
                flagged: flagged,
                dataHash: dataHash,
                timestamp: block.timestamp,
                blockNumber: block.number
            })
        );

        entryIndex[round][clientId] = entries.length; // 1-based

        if (flagged) {
            flaggedCount[clientId] += 1;
            emit AnomalyFlagged(round, clientId, shapleyValueScaled, dataHash);
        } else {
            cleanCount[clientId] += 1;
        }

        emit ScoreLogged(round, clientId, shapleyValueScaled, flagged, dataHash);
    }

    function getEntry(uint256 round, uint256 clientId) external view returns (ScoreEntry memory) {
        uint256 idx = entryIndex[round][clientId];
        require(idx != 0, "ShapleyAudit: not found");
        return entries[idx - 1];
    }

    function totalEntries() external view returns (uint256) {
        return entries.length;
    }

    /// @notice Reward eligibility gate: never flagged, and at least
    ///         `minCleanRounds` of clean (non-flagged) logged history.
    function isRewardEligible(uint256 clientId, uint256 minCleanRounds) external view returns (bool) {
        return flaggedCount[clientId] == 0 && cleanCount[clientId] >= minCleanRounds;
    }
}
