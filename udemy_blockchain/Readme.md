## Summary 
- Title: Blockchain A-Z: Learn How To Build Your First Blockchain
- Instructors: Hadelin de Ponteves, Kirill Eremenko

## Section 1: Introduction

2. Get the Datasets here
- https://www.superdatascience.com/blockchain/

## Section 2: Part 1 - Blockchain

## Section 3: Blockchain Intuition

6. What is a Blockchain
- Genesis block
  - Prev. Hash: 00000000
  - Hash      : 034DFA35
- 2nd block
  - Prev. Hash: 034DFA35
  - Hash      : 4DF175F2
- Continues with linkage ...
- Terms to know
  - Mining
  - Consensus protocol
  - Hash cryptography
  - Distributed P2P Network
  - Immutable Ledger

7. Understawnding SHA256 - Hash
- Finger prints of electronic documents
- 5 requirements for hash algorithms
  - One way
    - Cannot recover documents using hash values
  - Deterministic
    - Must yield same hash values for the same documents
  - Fast computation
  - Avalanche effect
    - Slight changes of the documents yield different hash values
  - Must withstand collisions
    - Must avoid forged documents yielding the same hash values of real documents

8. Immutable Ledger
- Forgering ledger requires the change of the entire sequential blocks

9. Distributed P2P network

10. How mining works: the nonce
- A block stores multiple transactions
- Nonce: Number used once

11. How mining works: the cryptographic puzzle
- Target: expressed with leading zeroes
  - Miners search hash values lower than target by changing Nonce
  - Avalanche effect must be guaranteed to avoid any forecasting

12. Byzantine fault tolerance
- https://en.wikipedia.org/wiki/Byzantine_fault

13. Consensus protocol: defense against attackers

14. Consensus protocol: Proof-of-work (PoW)

15. Blockchain Demo
- https://tools.superdatascience.com/blockchain/hash/

## Section 4: Create a Blockchain

## Section 5: Part 2 - Cryptocurrency

## Section 6: Cryptocurrency

34. What is Bitcoin?
- Cryptocurrency overview
  - Technology: Blockchain
  - Protocol /coin/: Waves, Ethereum, Bitcoin, Neo, Ripple
  - Token: 
- Bitcoin ecosystem
  - Nodes
  - Miners
  - Large mines
  - Mining pools

35. Bitcoin's monetary policy
- The having: a block reward goes down every 4 years

36. Understanding Mining Difficulty?
- What is the current target and how does that feel?
- How is mining difficulty calculated?
- For 18 leading zeros in 64digit hexadecimal number, a randomly picked hash is valid by 2E-22
- Difficulty = current target/max target
  - Difficulty is adjusted every 2016 blocks (2weeks)

38. Mining pools
- Cluster of small miners
  - Can avoid any redundancy 

39. Nonce range
- Unsigned 32bit number: [0, 4Billion]
- Timestamp: Unix time in sec
  - Nonce must be found within a sec. Unless, it repeats with new block data

40. How Miners pick transactions (part 1)
- From mempool

41. How Miners pick transactions (part 2)
- Mostly block size is 1MB

42. CPU vs GPU vs ASIC
- CPU: < 10 Mega hash /sec
- GPU: < 1 Giga hash /sec
- ASIC: > 1000 Giga hash /sec
  - Etherium miners may not use ASIC due to high memory requirement

43. How do Mempools work?
- A mempool is a storage area where transactions are stored before they are added to a block. Every participant of the P2P distributed network has their own mempool on their computer.

44. Orphaned blocks
- When multiple miners hash simultaneously, longer blocks survive
  - Some transactions in the orphaned blocks will be stored in mempools
- Wait for 6 confirmation
  - Ref: https://originstamp.com/blog/what-are-blockchain-confirmations-and-why-do-we-need-them/

45. The 51% attack
- Not 51% of nodes
- 51% of hashes
  - Longest chain wins
  - Nothing illegal (?)

46. Extra: Bits to Target conversion

## Section 7: Cryptocurrency Transactions Intuition

48. Transactions and UTXO's
- UTXOs
  - Unspent Transaction Output
  - They are removes from UTXO list when they are consumed in later transactions

49. Where do transactions fees come from?

50. How wallets work
- Wallet calculates how much UTXOs are available

51. Signature: Private & Public Keys
- Privacy of transactions
  - Private & Public key

## Section 8: Create a Cryptocurrency

## Section 9: Part 3 - Smart Contract

## Section 10: Smart Contract

## Section 11: Create a Smart Contract

## Section 12: Alt Coins

## Section 13: Special Offer
