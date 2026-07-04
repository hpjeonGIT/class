# InfiniBand Deep Dive: Networking for AI focused Datacentres
- Instructor: Ashish Prajapati

## Section 1: Introduction

### 1. Getting Started

## Section 2: Infiniband - The Network for AI Workloads

### 2. 01 - Why do we need faster network
- Data Growth
  - Massive increase in data from AI, analytics, and applications
- Distributed computing needs
  - Workloads run across multiple machine
- AI/ML training demands
  - Frequent synchronization b/w GPUs/nodes
- Real-Time & Low latency requirements
  - Application need instant responses (microseconds matter)
- Maximizing compute utilization

### 3. 02 - Analogy Road vs. Bullet Train

### 4. 03 - Evolution of InfiniBand
- 2002: SDR (Single Data Rate, 10Gbps)
- 2008: QDR (Quad Data Rate, 40Gbps)
- 2011: FDR (Fourteen Data Rate, 56Gbps)
- 2015: EDR (Enahnced Data Rate, 100Gbps)
- 2018: HDR (High Data Rate, 200Gbps)
- 2021: NDR (Next Data Rate, 400Gpbs)
- 2023: XDR (eXtream Data Rate, 800Gbps)
- 2026: GDR (Gigabit Data Rate, 1600Gbps)
  - Not standardized yet

### 5. 04 - InfiniBand Connections
- CPU nodes
- GPU nodes
- IB switches
- Storage

## Section 3: Infiniband Architecture Layers

### 6. 05 - Analogy - Package Delivery

### 7. 06 - InfiniBand Architcture Layers
- Upper layer
  - IPoIB (IP over infiniband)
  - NVMe over fabrics
  - GPU Direct RDMA
  - Interfaces directly with applications
  - Enables High performance distributed computing
- Transport layer
  - Send/Receive queues
  - Transport Types: RC, UC, UD
  - RDMA operations
  - Handles end-to-end communication
  - Zero-copy data transfer (no CPU involved)
  - Guarantees delivery (RC) or best-effort (UD)
- Network layer
  - LID (Local Indentifier addressing)
  - Subnet Manager (SM)
  - Routing tables
  - Determines path inside IB subnet
  - Centralized control via subnet manager
  - Optimized for low-latency deterministic routing
- Link layer
  - Packet framing
  - CRC (error detection) control
  - Virtual Lanes (VLs)
  - Ensures reliable node-to-node communication
  - Prevents packet using credit system
  - Supports QoS via multiple virtual lanes
- Physical Layer
  - Copper/Fiber cables
  - Connectors (QSFP, OSFP)
  - Electrical/optical signaling
  - Transmits raw bits over physical medium
  - Defines speed (HDR, NDR, XDR, etc)
  - Responsible for signal integrity

### 8. 07 - Software and Hardware
<img src="./sec08_HW.png" height="500">

## Section 4: Connecting through Infiniband

### 9. 08 - InfiniBand Physical Connectivity
<img src="./sec09_connection.png" height="500">

### 10. 09 - Host Channel Adapters
- Equivalent to an ethernet adaptor on an ethernet network
- Provides CPU offload which increases performance and decreases latency
- Single port vs multiple port HCA
- ConnectX
  - A unified HCA which supports both IB and Ethernet
  - Modern models like ConnectX-8 is programmable, offload-heavy accelerator

### 11. 10 - Cables
- Direct Attached Copper (DAC)
  - Electrical signals over copper
  - Short range (5-7 meters)
  - Very low power consumption
  - Cheaper
  - Thicker, heavier, less flexible
  - Susceptible to EMI
- Active Optical Cable (AOC)
  - Optical (light) signals over fiber
  - Long range (up to 100m+)
  - Higher power consumption
  - More expensive
  - Thinner, lighter, more flexible
  - Immune to EMI

### 12. 11 - Connecting Two Nodes
<img src="./sec12_2nodes.png" height="500">

### 13. 12 - Physical Connection
- Check Nvidia Docs Hub for HW installation
  - https://networking-docs.nvidia.com/
  - Download Nvidia OFED
- Demo:
```bash
$ hostnamectl # check network status
...
$ lspci | grep Mellanox # find HCA
...
$ lsmod | grep mlx # check if the driver is installed such as ib_core, ib_uverbs, mlx4/5_ib, mlx4/5_core
...
```

### 14. 13 - IB Modules
<img src="./sec14_ib_mod.png" height="500">

### 15. 14 - Verifying Device
- Demo:
```bash
$ ibv_devinfo # if not work,  sudo apt install ibverbs-utils
... # port info, state, guid, port_lid
$ 
```

### 16. 15 - Nodes Communication
- When cables are connected, port state becomes from `PORT_DOWN` to `PORT_INIT`
- To make the port active, subnet manager is required
  - Let's run opensm on one node

## Section 5: Subnet Manager (SM)

### 17. 16 - Why we need Subnet Manager
- Subnet Manager is the "brain" of the IB fabric that discovers, configures, and maintains the entire network
  - Assigns LIDs
  - Discovers topology
  - Programs routing tables
  - Ensures path consistency
  - Manages partitions (P_Key)
  - Handles link state changes
  - Optimizes performance
- Multiple SM can run for fail-over
- Where can SM run in IB network?
  - Must be "inside the fabric" to control it
  - Switch
    - Most common in production fabrics
    - Runs inside managed switches 
    - High reliability, always-on with the fabric
  - Any node
    - Runs on a server with an HCA (e.g., openSM)
    - Useful for labs, small clusters, or cost-sensitive setups
  - Dedicated management node
    - A separate server purely for fabric management
    - Preferred in large-scale HPC/AI clusters

### 18. 17 - OpenSM Installation
- Demo:
```bash
$ opensm --help
$ sudo apt install opensm rdma-core ibverbs-utils infiniband-diags
$ sudo modprobe ib_umad
# moving to 2nd node
$ ibv_devinfo
# Check PORT_INIT
$ sudo systemctl start opensm
$ ibv_devinfo 
# Check PORT_ACTIVE now. Now sm_lid and port_lid are given (local ID)
```

### 19. 18 - Connecting through Switch
- Demo using IS5022 

## Section 6: Infiniband Addressing

### 20. 19 - Getting Details of IB Adapter
### 21. 20 - Global Unique Identifier
### 22. 21 - Local Identifier

## Section 7: Infiniband Utilities

### 23. 22 - Checking Connectivity
### 24. 23 - Topology Discovery
### 25. 24 - InfiniBand Utilities - Part 1
### 26. 25 - InfiniBand Utilities - Part 2
4min

### 27. 26 - Direct Memory Access
### 28. 27 - Remote DMA
### 29. 28 - Traditional Data Transfer
### 30. 29 - RDMA - Analogy
### 31. 30 - Zero Copy Transfer
### 32. 31 - RDMA Verbs
2min

### 33. 32 - Key InfiniBand Features
### 34. 33 - Queue Pairs
### 35. 34 - Memory Registration
### 36. 35 - Hardware Offload
### 37. 36 - Lossless, Credit Based Flow
### 38. 37 - RDMA Demo
9min

### 39. 38 - GPUDirect RDMA
### 40. 39 - GPUDirect Storage
### 41. 40 - GPUDirect RDMA vs GPUDirect Storage
3min

### 42. 41 - Ethernet vs InfiniBand
### 43. 42 - TCP-IP vs InfiniBand
3min

### 44. 43 - Going Beyond an InfiniBand Subnet
### 45. 44 - InfiniBand Router
### 46. 45 - InfiniBand Packet
### 47. 46 - Global ID (GID)
### 48. 47 - GUID vs. LID vs. GID
3min

### 49. 48 - NVIDIA InfiniBand Stack
### 50. 49 - NVIDIA InfiniBand Hardware Stack
### 51. 50 - NVIDIA InfiniBand Software Stack
### 52. 51 - OpenFabrics Enterprise Distribution - Part 1
### 53. 52 - OpenFabrics Enterprise Distribution (OFED) - Part 2
3min

### 54. 53 - Traffic Isolation in InfiniBand
### 55. 54 - Power of Partition Key (PKey)
### 56. 55 - Configuring Partitions
### 57. 56 - 3 Node Partitioning
12min

### 58. 57 - Network Topology and Routing
### 59. 58 - Control Plane and Data Plane
### 60. 59 - One Switch Setup
### 61. 60 - Two Switch Setup
### 62. 61 - Direct or Through Switch
### 63. 62 - Routing Algorithms
### 64. 63 - MINHOP Routing
### 65. 64 - UPDN Routing
### 66. 65 - Fat Tree Routing
### 67. 66 - Adaptive Routing
### 68. 67 - Chagning Routing Algorithm
### 69. 68 - Why Adaptive Routing?
### 70. 69 - Configuring Adaptive Routing
### 71. 70 - Adaptive Routing Demo - Part 1
### 72. 71 - Adaptive Routing Demo - Part 2
### 73. 72 - LID Mask Control - LMC
### 74. 73 - Network Congestion
### 75. 74 - Credit Loops
### 76. 75 - How to avoid Credit Loops?
### 77. 76 - Algorithm Comparision
6min

### 78. 77 - Traffic Prioritization
### 79. 78 - Quality of Service (QoS)
### 80. 79 - QoS vs SL vs VL
### 81. 80 - Configuring QoS - Part 1
### 82. 81 - Configuring QoS - Part 2
### 83. 82 - Configuring QoS - Part 3
### 84. 83 - Verify QoS
### 85. 84 - QoS in Action
9min

### 86. 85 - Unified Fabric Manager
### 87. 86 - UFM Architecture
### 88. 87 - UFM Platform Capabilities
### 89. 88 - Getting Started with UFM
### 90. 89 - UFM Traffic Map
### 91. 90 - UFM Quick Tour
### 92. 91 - Topology File
### 93. 92 - Important UFM Messages
### 94. 93 - Role Based Access Control
### 95. 94 - UFM Cyber-AI
3min

### 96. 95 - Device Level Management
### 97. 96 - Accessing Switch Console Port
### 98. 97 - Initial Configuration
### 99. 98 - Some Useful Commands
### 100. 99 - Mellanox OS - Web Interface
### 101. 100 - Upgrade Procedure
5min

### 102. 102 - Troubleshooting Issues
### 103. 103 - Performance Issues
### 104. 104 - Connectivity Issues
### 105. 105 - Connectivity Issues
### 106. 106 - Physical Layer Issues
7min

### 107. Next Steps
1min

