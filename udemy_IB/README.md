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
```bash
$ ibstat
...
```
- CA: Channel Adapter
- ibp1s0: PCI location of IB device
- MT4099: HW model
- Rate: Link speed. 40 means 40Gb/sec
- Base lid: Local Identifier assigned by Subnet Manager
- LMC: 0 means only 1 lid assigned
- SM lid: LID of the Subnet Manager (SM)
- Capability mask: Bitmask - mostly used for low-level diagnostics

### 21. 20 - Global Unique Identifier (GUID)
- 64bit value burned into HW (by manufacture)
- Used by SM to assign LIDs

### 22. 21 - Local Identifier (LID)
- A 16-bit address assigned to each IB port (Range: 0x0001->0xFFFE)
- Unique within a subnet, used for routing packets within a subnet
- Assigned dynamically by the SM
- Why LID is needed?
  - Enables efficient HW-level routing
  - Allows switches to forward packets using lookup tables (LFTs)
  - Decopules communication from GUID

| Feature | LID (16bit) | GUID(64bit)|
|---------|-------------|------------|
| Scope   | Local(subnet) | Global   |
| Assigned by| Subnet Manager | Manufacturer |
| Changes | Yes          | No         |
| Used for routing | Yes | No         |

## Section 7: Infiniband Utilities

### 23. 22 - Checking Connectivity
- ibping
  - A tool test connectivity b/w IB nodes
  - Works at IB layer (LID/GUID)
  - Similar to ping but not IP-based
  - When one node runs as server: `ibping -S`
  - Another node sends ping requests: `ibping -L <LID>`

### 24. 23 - Topology Discovery
- ibnetdiscover
  - A tool to discover and map the entire IB fabric
  - Shows
    - Nodes (HCAs)
    - Switches
    - Ports and connections
<img src="./sec24_topo.png" height="300">
<img src="./sec24_found.png" height="500">

### 25. 24 - InfiniBand Utilities - Part 1

| Tool          |  Purpose   | When to use  |
| ------------- |------------| -------------|
| iblinkinfo | Shows link-level | Quick topology check |
| ibtracert  | Traces path b/w nodes | Identify hops/switches |
| ibhosts    | Lists all IB hosts | Fabric discovery |
| ibswitches | Lists switches | Switch inventory |
| ibnodes    | List all nodes | Full fabric view (nodes + switches) |
| ibv_devinfo | Detailed HCA capabilities | HW validation

### 26. 25 - InfiniBand Utilities - Part 2
<img src="./sec25_tools.png" height="200">

## Section 8: Remote Direct Memory Access (RDMA)

### 27. 26 - Direct Memory Access
- DMA (Precursor to RDMA)
  - A HW feature that lets supported devices transfer data to/from memory without the CPU actigin as middle-man
- Without DMA:
  - CPU must copy every byte from device to memory
  - CPU utilization up to 90-100% for data transfer
- With DMA:
  - Device controller handles transfers directly
  - CPU utilization: 5-10%
  - 10-100x faster data movement

<img src="./sec27_dma.png" height="300">

### 28. 27 - Remote DMA (RDMA)
- DMA over the network
  - Allows one computer to directly read or write memory on another computer without involving the CPU or OS
  - IB or RoCE or iWARP

<img src="./sec28_rdma.png" height="200">

### 29. 28 - Traditional Data Transfer
<img src="./sec29_data.png" height="300">

- High latency due to multiple SW layers
- CPU bottlenecks
- Inefficient data handling
  - Repeated memory copies add overhead and reduce efficiency

### 30. 29 - RDMA - Analogy
- RDMA removes layers, offloads the CPU, and eliminates copies

<img src="./sec30_rdma.png" height="300">

### 31. 30 - Zero Copy Transfer
- Traditional data transfer needs to copy data from application to kernel then to NIC
  - Heavy CPU utilization
  - Destination host repeats the steps inversely to get data in application

<img src="./sec31_zerocopy.png" height="400">

### 32. 31 - RDMA Verbs
- Basic commands (APIs) that applications use to talk directly to the network HW
  - Low-level instructions to perform RDMA operations
- Why "verbs"?
  - Because they are literally actions like create, register, send, write, read, poll  
- Application <--> Verbs <--> RDMA supporting HW like IB, iWARP, RoCE

## Section 9: Key Infiniband features

### 33. 32 - Key InfiniBand Features
- IB is a natural fit for RDMA because it provides:
  - Queue Pair (QP) built into HW
  - Memory Registration + Key mechanism
  - HW Offload capability
  - Lossless, credit-based flow control

### 34. 33 - Queue Pairs
<img src="./sec34_qp.png" height="300">

- Application: 
  - Creates a QP using RDMA verbs (ibv_create_qp)
  - Posts work requests -> Send Queue (SQ)
- Once work is posted
  - NIC reads the request from SQ
  - Fetches data using DMA
  - Sends packets over fabric (ibv_post_send)
  - Remote NIC processes and places data in memory
  - Completion is written to Completion Queue (CQ)
- Application:
  - Polls for completion -> CQ(ibv_poll_cq)

### 35. 34 - Memory Registration (MR)
- A block of memory registered with the RDMA NIC to enable direct memory access through RDMA operations
- Each MR is associated with two keys used to authorize RDMA access
  - Local key (lkey): used by the local RDMA NIC for internal access to the memory
  - Remote key (rkey): shared with remote peers to grant them access, based on the permissions set during registration

### 36. 35 - Hardware Offload
- IB HCAs move transport logic, data movement, and flow control from the CPU-driven SW stack into dedicated HW
- Top 5 things HCA does in HW

| Function | IB HCA (HW) |
|----------|-------------|
| Transport processing | Handles sequencing, ACK/NACK, retransmissions, in-order delivery in HW |
| Data movement | Direct memory-to-memory transfer via DMA (zero-copy) |
| Request execution model| Processes RDMA operations via Queue Pairs (QP) in HW |
| Packetization & reassembly | Segmentation and reassembly done in NIC HW |
| Memory protection & access control | Validates lkey/rkey, enforces memory region boundaries in HW|

### 37. 36 - Lossless, Credit Based Flow
- IB doesn't deal with packet loss - it prevents it from happening in the first place
- Sender only sends data when the receiver is ready - so nothing gets dropped
- Credit-based flow control 
  - Credits are pre-approved permission to send data - just like a credit limit of your credit card is pre-approved permission to send

<img src="./sec37_credit.png" height="250">

### 38. 37 - RDMA Demo
```bash
$ ibstat
...
$ ibping -S # in the first node
$ ibping 4 # in the second node
```
- Bandwidth and latency test
```bash
$ ib_write_bw # in the first node
$ ib_write_bw 192.168.0.xx # in the second node
...
$ ib_write_lat 192.168.0.yy
```
<img src="./sec38_test.png" height="350">

## Section 10: How Nvidia is using RDMA?

### 39. 38 - GPUDirect RDMA
<img src="./sec39_gpuDMA.png" height="400">

### 40. 39 - GPUDirect Storage
- Direct GPU-to-storage transfers bypassing CPU and system memory
- Reduces IO bottlenecks in training

<img src="./sec40_gpuDirectStorage.png" height="400">

### 41. 40 - GPUDirect RDMA vs GPUDirect Storage
<img src="./sec41_comparison.png" height="400">

## Section 11: Comparing IB

### 42. 41 - Ethernet vs InfiniBand
<img src="./sec42_ethervsIB.png" height="400">

### 43. 42 - TCP-IP vs InfiniBand
<img src="./sec43_TCPvsIB.png" height="400">

## Section 12: Going beyond a subnet

### 44. 43 - Going Beyond an InfiniBand Subnet
- Spine vs leaf switches
  - Spine: upper hierarchy switches
  - Leaf: lower hierarcy switches
- Is One Subnet sufficient?
  - A subnet is optimized for speed and simplicity, not infinite scale
    - Each node in a subnet gets a LID
    - LIDs are assigned by the subnet manager (SM)
    - Routing is LID-based within the subnet
  - So the subnet size is fundamentally tied to LID space

<img src="./sec44_limit.png" height="200">

- Challenges with single subnet design
  - Scalability limit
  - Subnet manager bottleneck
  - Large fault domain
  - Shared congestion domain
- Different topologies
  - To optimize performance, scalability, and cost based on workload patterns, traffic flow, and cluster size

<img src="./sec44_topo.png" height="200">

### 45. 44 - InfiniBand Router
- An IB router connects multiple IB subnets, enabling communication across separate subnet while maintaining isolation and scalability
- Not commonly used
- But might be asked in certificate exams

<img src="./sec45_fabric.png" height="500">

### 46. 45 - InfiniBand Packet
<img src="./sec46_packet.png" height="400">

- ibdump: capturing IB packet
```bash
$ ibping -S #  in the first node
$ ibping 2 # in the 2nd node
$ ibdump -d mlx4_0 -i 1 -w ib_capture.pcap # in the first node
```
- Wireshark can open *.pcap files

### 47. 46 - Global ID (GID)
- A globally unique address used to identify a device across multiple IB subnets
- LID works only within a single subnet that's why GID is needed when communication goes across subnets (via IB router)
- Used for global routing across subnets
- GID follows IPv6 format (128-bit addressing)
  - Subnet prefix (64bits, identifies the IB subnet) + Interface ID (64bits, unique identifier of the device (HCA port))
  - `ibv_devinfo -v` or `rdma link show` or `ib_addr` to view GIDs

### 48. 47 - GUID vs. LID vs. GID
<img src="./sec48_address.png" height="400">

## Section 13: Nvidia Infiniband Stack

### 49. 48 - NVIDIA InfiniBand Stack
<img src="./sec49_history.png" height="400">

### 50. 49 - NVIDIA InfiniBand Hardware Stack
<img src="./sec50_hw1.png" height="400">
<img src="./sec50_hw2.png" height="400">

### 51. 50 - NVIDIA InfiniBand Software Stack
<img src="./sec51_sw.png" height="400">

### 52. 51 - OpenFabrics Enterprise Distribution - Part 1
- OFED Drivers
  - Drivers shipped with OS may not be optimized for ConnectX adapters
  - For the best performance, replace it with optimized drivers
  - MLNX_OFED: ConnectX-3..5
  - DOCA_OFED: ConnectX-5..8
    - BlueField-2/3 DPU
```bash
$ lsmod |grep mlx
...
$ modinfo mlx5_core | grep filename
...
$ ofed_info
...
```

### 53. 52 - OpenFabrics Enterprise Distribution (OFED) - Part 2
- Mellanox driver: https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
- sudo ./mlnxofedinstall --add-kernel-support --fw-update

## Section 14: Traffic Isolation in InfiniBand

### 54. 53 - Traffic Isolation in InfiniBand
- Why traffic isolation is critical?
  - In a shared IB fabric, uncontrolled communication can lead to security risks, performance interference, and operational instability
- Real-World Scenarios
  - Multi-tenant AI clusters: prevent data leakage and interference b/w teams sharing GPU infrastructure
  - Dev/Test/Production: avoid accidental cross-communication and protect production stability
  - Healthcare & research workloads: enforce strict data prviacy and regulatory compliance
  - Mixed workloads: Prevent performance degradation caused by competing traffic patterns
- Traffic isolation ensures security, predictable performance, and efficient resource sharing on a single IB fabric
- How to implement isolation?
  - PKey (Partition Key) is a 16bit field in IB used to logically isolate traffic within the same physical fabric
  - The subnet Manager
    - Defines partitions
    - Assigns PKeys to nodes (HCAs)
    - Distribute PKey tables across the fabric
  - The node (HCA/host)
    - Stores assigned PKeys in its PKey table
    - Tags outgoing packets with a selected PKey
    - Validates incoming packets against its PKey table
    - Accepts or drops traffic based on PKey membership
    - Enforces partition isolation(data plane enforcement)

### 55. 54 - Power of Partition Key (PKey)
- Configuring PKey
  - Define partitions (partitions.conf)
  - Create membership
    - Full: full member can talk to everyone in the same partition
    - Limited: Limited member can only talk to Full members - never to other Limited members
      - Control plane traffic (SM) is always allowed
      - PKey only influences Data Plane Traffic

### 56. 55 - Configuring Partitions
- Steps
  - Point OpenSM to configuration file: `opensm -P /etc/opensm/partitions.conf`
  - Add required entries: `vi /etc/opensm/partitions.conf`
  - Restart OpenSM service: `sudo systemctl restart opensm`
- Multiple partitions can be defined for complex partition setup
  - Different partitions will not talk to each other

### 57. 56 - 3 Node Partitioning
- Demo using 3 nodes
```bash
$ smpquery PKeys <LID> <PORT>
...
$ cat /etc/opensm/partitions.conf
...
$ ibdiagnet
...
```

## Section 15: Network Topology and Routing

### 58. 57 - Network Topology and Routing
- Network topologies
  - Physical connectivity of hosts and switches
  - Fat Tree
  - Full Mesh
  - Dragon Fly+
  - Torus 3D

### 59. 58 - Control Plane and Data Plane
- Subnet Manager (Control Plane) workflow
  - Discover topology
  - Assigns LIDs (to all nodes)
  - Choose routing algorithm
  - Compute paths
  - Program linear forwarding table (LFT) in switches
- Switch (Data Plane) workflow  
  - Receive packet
  - Lookup LFT (using destination LID)
  - Select output port (based on routing type)
  - Forward packet to next hop

### 60. 59 - One Switch Setup
- Linear Forwarding Table
  - Inside of every IB switch
  - Maps destination LID to an output port
  - Enables fast, HW-based forwarding
  - Pre-programmed by Subnet Manager
  - Stored in swtich ASIC (Application-Specific Integrated Circuit)
- When packet arrives
  - Switch reads destination LID
  - Looks up LFT
  - Sends packet out correct port
- To view LFT: `ibroute <switch_lid>`
  - Other useful commands
    - `ibnetdiscover`
    - `ibtracert <src_lid> <dst_lid>`
- Demo using 3 nodes
```bash
$ ibhosts
...
$ ibswitches
...
$ ibnodes # hosts + switches
...
$ ibroute 4 # 4 is the LID of the switch here. Shows LFT
```

<img src="./sec61_demo.png" height="400">

### 61. 60 - Two Switch Setup
- Using LID 4 and 5 for two switches
```bash
$ ibnetdiscover
... # copy/paste the content into chatGPT then ask a diagram of the topology
$ ibtracert 1 2 # LID 1 -> LID 4 (switch) -> LID 2
```

### 62. 61 - Direct or Through Switch
```bash
$ ibroute 4 # shows different ports when direct or through another switch
...
$ ibroute 5 # compare the results with the above
```

### 63. 62 - Routing Algorithms
- MINHOP: default algorithms
- UPDN: 
- Fat-tree
- Adaptive Routing: in AI/HPC

<img src="./sec63_alg.png" height="100">

### 64. 63 - MINHOP Routing
- Minimum hops: Always picks least number of hops, first available shortest path
  - No congestion awareness: certain switchs might be saturated easily
  - Other paths are ignored (even if better)

### 65. 64 - UPDN Routing
- Traffic must go UP (towards spine) then DOWN (towards destination)
  - No zig-zag
  - Never go up again to prevent loops

### 66. 65 - Fat Tree Routing
- Uses ALL equal-cost paths simultaneously
  - Traffic is split across both spines
  - Achieves load balancing and high throughput
  - No congestion awareness

### 67. 66 - Adaptive Routing
- Chooses path based on congestion in real time
- If congestion changes, path can change mid-flow
  - Needs modern-smart switches: Quantum series

### 68. 67 - Changing Routing Algorithm
- View current routing algorithm
  - `cat /etc/opensm/opensm.conf |grep routing_engine`
- Change routing algorithm
  - `routing_engine`: minhop/updn/ftree
- Restart Opensm: `systemctl restart opensm`

### 69. 68 - Why Adaptive Routing?
- Why adaptive routing is well suited for AI workload?
  - Spreads elephant flows across multiple paths dynamically
    - Large, long-lived traffic that consume significant BW
      - Distributed AI training
      - Checkpointing
      - Large dataset transfers
    - Elephant flows = few but dominant flows
      - Heavy load
      - Fill switch buffers quickly
      - Occupy links for long duration
      - Traditional load balancing breaks as hashing doesn't distribute them well
  - Avoids static flow-based load balancing that creates congestion hotspots
    - In static flow-based load balancing, each flow is assigned a fixed path based on a hash and does not change, regardless of network conditions
    - How it works?
      - Uses hashing (source/destination IP, ports)
      - Selects one path from multiple equal-cost paths
      - Flow stays on that path for its entire lifetime
    - Static load balancing can place multiple elephant flows on the same path
  - Balances low-entropy traffic using real-time path selection
    - Entropy: a measure of the amount of disorder in a system
    - Low-entropy traffic means there is very little disorder in communication patterns - traffic is highly predictable, repetitive, and concentrated b/w the same set of nodes
    - In AI workloads:
      - Same GPUs talk to the same GPUs again and again
      - Large flows follow similar paths repeatedly
      - Very little variation (low randomness) in traffic

### 70. 69 - Configuring Adaptive Routing
- Confirm your switches support it
  - In Switch: `show system capabilities`
- Install supported openSM
  - opensm binary form MLNX_OFED, not OS distribution
- /etc/opensm/ib-osm-roots.cfg:
```bash
ar_mode 1
enable_ar_by_device_cap TRUE
ar_transport_mask 0x000A
ar_sl_mask 0x01
guid 0x0
```
- root_guid_file
  - Identify your core/spine switches and create a root_guid_file
  - Root = switch that connects to the most other switches
  - Secondary roots = switches that connect to root AND leaves
  - /etc/opensm/ib-som-roots.cfg:
```bash
# Core/spine switch
0x0002c90200425380
# Secondary roots
0x0002c90300759bc0
0x506b4b0300599750
```  

### 71. 70 - Adaptive Routing Demo - Part 1
- Why a single HCA has multiple LIDs?
  - LMC (Lid Mask Control): an IB feature that allows a single port (HCA or switch port) to be assigned multiple LIDs
- Demo of multiple-paths:
```bash
$ ibstat
...
$ ibaddr
...
$ ibtracert 8 10 # from LID 8 to LID 10
...
```
<img src="./sec71_demo.png" height="300">

### 72. 71 - Adaptive Routing Demo - Part 2
- Demo of BW shift
- Server: `ib_send_bw -d lmx5_0 -i 1 -q 8 -s 65536 --run_infinitely`
- Client: `ib_send_bw -d mlx5_0 -i 1 -q 8 -s 65536 --run_infinitely 192.168.0.48`
- Observation: `watch -n 1 'echo "SX6036:" perfquery 5 4 | grep XmitData; echo "SwitchX:"; perfquery 7 4 | grep XmitData'`

### 73. 72 - LID Mask Control - LMC
- An IB feature that allows a single port (HCA or switch port) to be assigned multiple LIDs
- Each LID maps to a different path in the forwarding tables. So same destination node gets different routes
- Why LMC?
  - Path diversity (multipathing)
  - Improved performance
  - Works with Adaptive Routing
- In /etc/opensm/opensm.conf:
```
lmc 0 # default 0 -> 1LID, 1 -> 2LIDs, 2-> 4LIDs
```
- `ibaddr` shows the status of LMC 

### 74. 73 - Network Congestion
- A condition where network traffic exceeds available bandwidth, causing packet delays, queue buildup, and performance degradation
- Why it happens:
  - Multi-send
  - Hotspot
  - Oversubscribed
  - Buffer limit
- Symptoms:
  - Increased latency
  - Packet drops/retransmissions
  - Reduced throughput
  - GPU/CPU underutilization (waiting for data)
- Congestion in AI/HPC context
  - Slows down distributed training
  - Causes straggler nodes
  - Impacts synchronization across GPUs
  - Reduces overall cluster efficiency
  - How it is handled
    - Congestion control
    - QoS priority
    - Adaptive Routing
    - Fabric Design
- How nvidia (Mellanox) switches eliminate congestion?
  - IB prevents congestion instead of reacting to it
    - Adaptive routing: spreads traffic across alternate paths to avoid hot links and balance load under changing conditions
    - Congestion control: uses explicit notifications and rate reduction so sources slow down before congestion gets worse
    - QoS and Virtual lanes: QoS assigns priority to traffic, and Virtual Lanes isolate it, ensuring critical flows are not impacted during congestion
    - Shared-buffer: absorb short traffic spikes by dynamically allocating memory across ports, preventing packet loss during microbursts

### 75. 74 - Credit Loops
- A credit loop occurs when switches in a cyclic path wait on each other for buffer credits, creating a deadlock across the traffic
- Demo of creating deadlock:

<img src="./sec75_demo.png" height="300">

- Key take-aways
  - Credit loops may occur: if there are 2 circular flows
  - Credit loop is NOT per host: it happens across switches (fabric-level problem)
  - Exact devices in loop: only the switches forming the cycle are stuck - not entire network

### 76. 75 - How to avoid Credit Loops?
- Use deadlock-free routing such as UPDN or Fat-Tree (avoid MINHOP)
- Use adaptive routing to balance congestion, but do not rely on it alone for deadlock prevention
- Configur Virtual Lanes properly to isolate traffic and reduce blocking
- Reduce hotspots by avoiding many-to-one patterns
- Use Unified Fabirc Manager (UFM) to monitor congestion and fabric behavior

### 77. 76 - Algorithm Comparison
<img src="./sec77_comparison.png" height="300">

## Section 16: Traffic Prioritization

### 78. 77 - Traffic Prioritization
- Policy -> QoS
- Tagging -> Service Level
- Path -> Virtual Lane

### 79. 78 - Quality of Service (QoS)
- QoS controls how different traffic types share the network to ensure predictable performance
- OoS ensures:
  - Priority handling
  - Fair BW allocation
  - Predicatable performance
- QoS is defined by the Subnet Manager, applied by the HCA, and enforced by the switches
- Service Level (SL)
  - 4-bit QoS marking assigned to IB packets
  - SLs are packet labels used to classify and prioritize traffic in IB
  - Used to determine:
    - Which VL the packet uses
    - How congestion & arbitration are handled
  - SL is used for setting priority and mapping to virtual lanes (VLs)
- Virtual Lane
  - Creates multiple logical channels withing a single physical IB link
  - Enables traffic isolation (e.g., storage vs AI vs control traffic)
  - Avoids network deadlocks
  - Improves performance and predicatability under congestion
  - VL0 -> VL14 (total 15 usable VLs)
    - VL15 is reserved for Subnet Manager Traffic

### 80. 79 - QoS vs SL vs VL

### 81. 80 - Configuring QoS - Part 1
- A multi-tenant cluster

<img src="./sec81_demo.png" height="300">

- Configuring QoS
  - opensm.conf
    - Activates QoS and defines the rules
    - Defines VL arbitration weights and SL2VL table which is pushed to every switch port
  - qos-polic.conf
    - Assigns those rules to tenants - tenant awareness
    - Define port groups by GUID and it assigns a Service Levels (SL) to each group
    
### 82. 81 - Configuring QoS - Part 2
- opensm.conf
  - qos TRUE # Turn QoS on
  - qos_max_vls 4 # use 4 virtual laens
  - qos_policy_file /etc/opensm/qos-policy.conf # details for tenant rules
  - qos_sl2vl 0,1,2,3,3,3,3,3,3,3,3,3,3,3,3,15
  - qos_vlarb_high 0:100,1:50,2:20,3:10 # VL 0 has the highest weight as 100, VL1 has 50, VL2 has 20, VL3 has 10
    - Weights are relative

<img src="./sec82_mapping.png" height="300">

<img src="./sec82_weights.png" height="300">

### 83. 82 - Configuring QoS - Part 3
- qos-policy.conf
  - Who 
  - Which
  - Matching Engine

### 84. 83 - Verify QoS
- The subnet management packet query tool (smpquery) sends MADs (Management Datagram) directly to fabric devices to read their configuration
- Query service level to virtual lane mapping table
  - `smpquery sl2vl <LID>`
- Query the VL Arbitration table
  - `smpquery vlarb <LID>`
- Management Datagram (MAD)
  - The native management message format in IB
  - It is how the Subne Manager (OpenSM), management tools (smpquery, perfquery, ibstat), and fabric devices (HCAs, swithces) all talk to each other for management and configuration purposes - completely separate from your data traffic

### 85. 84 - QoS in Action
- Bandwidth test - ib_write_bw for 2 node system

<img src="./sec85_demo.png" height="300">

<img src="./sec85_result.png" height="300">

## Section 17: Unified Fabric Manager

### 86. 85 - Unified Fabric Manager
- Why UFM?
  - Without UFM you rely on CLI tools, manual configs, reactive troubleshooting
    - This will work for small setups (2-10 nodes) in labs/demos but breaks down quickly when:
      - You scale to 100+ nodes
      - AI workloads start stressing the network
      - You need visibility or automation
- UFM is Nvidia's centralized platform to monitor, manage, and optimize high performance network fabrics (IB & Ethernet for AI/HPC)
  - Full fabric visibility
  - Faster troubleshooting
  - Telemetry & performance monitoring
  - Predictive analysis (Cyber-AI)
  - Multi-tenancy & isolation
  - Automation & control
  - Scale management

### 87. 86 - UFM Architecture
<img src="./sec87_arch.png" height="400">

### 88. 87 - UFM Platform Capabilities
- UFM Telemetry: real-time monitoring
- UFM Enterprise: fabric visibility and control
- UFM Cyber-AI: cyber intelligence and analytics
- https://www.nvidia.com/en-gb/networking/infiniband/ufm/

### 89. 88 - Getting Started with UFM

### 90. 89 - UFM Traffic Map

### 91. 90 - UFM Quick Tour
<img src="./sec91_graph.png" height="300">

### 92. 91 - Topology File
- A text-based representaion of your fabric layout - it describes all nodes, switches, ports, and how they are connected
  - Generated using `ibdiagnet` tool

### 93. 92 - Important UFM Messages
<img src="./sec93_messg.png" height="300">

### 94. 93 - Role Based Access Control
- Role-Based Access Control (RBAC) is a security model where:
  - You don't give permissions directly to users
  - Instead, you:
    - Define roles (Admin, Operator, Viewer, etc)
    - Assign permissions to roles
    - Then assign users to roles
- Why RBAC is important in IB/UFM
  - Security - prevent unauthorized configuration changes
  - Stability - avoid accidental misconfiguration of: routing, Pkeys, QoS
  - Multi-tenancy - different teams/tenants get controlled access

### 95. 94 - UFM Cyber-AI
- Turns telemetry into intelligence - detecting, predicting, and preventing network issues
  - AI-powered anomaly detection
  - Security threat detection
  - Predictive failure analysis
  - Performance bottleneck identification
  - Behavioral baseline modeling

## Section 18: Mellanox OS (MLNX OS)

### 96. 95 - Device Level Management
- UFM: controller/orchestrator
- MLNX-OS: the switch's OS
  - Similar in concept to Cisco's IOS or Juniper's Junos
- UFM sends commands to switches
- MLNX-OS executes those commands on the device

### 97. 96 - Accessing Switch Console Port
- Console port with RJ45 connector
- USB port is for loading instruction from usb memory stick or installing OS image

### 98. 97 - Initial Configuration
- In putty
  - Connection -> Serial
    - Speed (baud): 9600
  - Session: serial
    - Then click connect
- Initial configuration
```bash
Switch> enable
Switch# configure terminal
Switch (config)# interface mgmt0
Switch (config interface mgmt0)# dhcp
Switch (config interface mgmt0)# exit
```

### 99. 98 - Some Useful Commands
- MLNX-OS will be deprecatd while Onyx and Cumulus will replace it

<img src="./sec99_history.png" height="300">

- Useful commands
  - show version
  - show inventory
  - show username
  - show system capabilities
  - show users
  - show interfaces
- MLNX-OS defines two primary roles
  - admin: full access (configuration + management)
  - monitor: read-only access (view configs, status)
- This separation ensures operational safety and role-based access control
- XML users -> tools/automation (UFM, scripts, APIs)
  - xmladmin: same power as admin but for automation tools
  - xmluser: read-only access via APIs

### 100. 99 - Mellanox OS - Web Interface
- Virtual Protocol Interconnect
  - IB
  - Ethernet

<img src="./sec100_gui.png" height="300">  

### 101. 100 - Upgrade Procedure
- CLI upgrade
  - Log in to the switch via SSH
  - Verify current version: `show version`
  - Fetch the new image: `image fetch scp://username:passwd@server/path/to/image.img`
  - Install the image: `image install <image_name>`
  - Set the next boot partition: `image boot next`
  - Verify the update is scheduled: `show images`
  - Save configuration: `configuration write`
  - Reload the switch: `reload`

## Section 19: Troubleshooting InfiniBand

### 102. 102 - Troubleshooting Issues
<img src="./sec102_tools.png" height="300">  

### 103. 103 - Performance Issues
- Categorization
  - Performance issues
  - Connectivity issues
  - Physical layer issues

<img src="./sec103_perf.png" height="200">  

### 104. 104 - Connectivity Issues
<img src="./sec104_command.png" height="300">  

### 105. 105 - Connectivity Issues
<img src="./sec105_command.png" height="300">  

### 106. 106 - Physical Layer Issues
- CVT: Cable Validation Tool

<img src="./sec106_hw.png" height="300">  

## Section 20: Next Steps

### 107. Next Steps