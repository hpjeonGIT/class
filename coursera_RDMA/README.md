## The Fundamentals of RDMA Programming
- Coursera by NVIDIA

### Lesson 1: RDMA - Bypassing the OS
- RDMA programming relies on the following three concepts
  1. Transport offload
    - Overhead from OS to the dedicated HW
  2. Kernel pass
    - When data is passed, not through OS
  3. RDMA operations and atomics
- How socket applications work?
  - When the application wants to send, it passes the buffer to the socket library -> OS -> TCP stack to the device driver
  - In RDMA, application passes the buffer to RDMA library, not socket library, overriding OS

### Lesson 2: What is RDMA?
- Communication model
  - Two sided communication
    - A sender and a receiver
  - One sided communication
    - A sender but the receiver will not participate in the operation
- RDMA
  - When a message is sent, it already has a destination address. Hence the HW of the receiver side directs the message to the specific address
  - Usually one side communication model

### Lesson 3: Memory zero copy
- Two techniques for sending data from point to point
  1. Buffer copy
    - Sender/receiver creates temporary buffers, then copy sender buffer into receiver buffer. Buffered data goes to memory after then
    - Two sided
  2. Zero copy
    - No buffer created. Addresses are directed
    - Sender notifi
    - One sided/RDMA

### Lesson 4: Transport offloads
- Transport layer from 7 OSI layer model
  - Reliability
    - Every message sent is checked and may send again
    - Timeout
  - Connectivity
  - Message/stream
- RC: Reliable
- UD: Unreliable but low latency

### Lesson 5: What are RDMA Verbs?
- Verbs
  1. Control path: resource allocation, modification, and destruction
  2. Data path: send/receive data
- libibverbs

### Lesson 6: RDMA Verbs Objects
- Requests: requests from the application to the HW (not receiver) to execute an operation
- Completions: indicates the completion of a previous work requests and typically contains both pointed the work request and the status of how it was completed
- Queues: posts work requests to be completed
  - Send and receive queues

### Lesson 7: RDMA data-path flow

### Lesson 8: RDMA Memory Management
- Memory registration has 3 properties
  1. Protection and permission
  2. Memory pinning (physical locking)
  3. Handle for accessing that memory

### Lesson 9: Knowledge Recap
- 3 most important verbs objects
  1. Queue pairs: transport endpoint like a socket for TCP
  2. Completion queues: a method for notifying us
  3. Memory regions: reference to registered memory
