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

### Lesson 10: RDMA Send & Receive Operations
- Send operation
  - Receiver must be ready
  - SW sends Work Request (WR) to HW then HW creates Send Queue (SQ). When HW sees the entry, it will access the data and send it on wire. HW will generate a Completion Queue and SW gets Work Completion
- Glossary
  - SR - Send Request
  - RR - Receive Request
  - CQ - Completion Queue

### Lesson 11: RDMA Write Operation
- Write operation
  - One side operation

### Lesson 12: RDMA Read Operation
- Read operation
  - One side operation

### Lesson 13: RDMA Atomic Operations
- Atomic operation
  - Fetch and Add
  - Compare and Swap
  - Done by receiver side
  - Avoid interruption

### Lesson 14: Memory Registration
```c
// Memory Reg Example
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd,
					    void *addr, size_t length,
					    enum ibv_access_flags access); 
/* Notice the following fields in struct ibv_mr:
rkey  - The remote key of this MR
lkey  - The local key of this MR
addr â€“ The start address of the memory buffer that this MR registered
length â€“ The size of the memory buffer that was registered
*/
//Deregister a Memory Region
int ibv_dereg_mr(struct ibv_mr *mr); 
/*
This verb should be called if there is no outstanding 
Send Request or Receive Request that points to it
*/
```

### Lesson 15: RDMA Send Request
```c
#define size_t int 
#define uint64_t int
#define uint32_t int
struct ibv_pd *pd;
struct ibv_mr *mr;
//Scatter Gather Entry
struct ibv_sge {
	uint64_t addr; // Start address of the memory buffer (registered memory)
	uint32_t length; // Size (in bytes) of the memory buffer
	uint32_t lkey;   // lkey of Memory Region associated with this memory buffer
};
//Post Send
int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                  struct ibv_send_wr **bad_wr);
struct ibv_send_wr {
	uint64_t wr_id;            // Private context that will be available in the corresponding Work Completion
	struct ibv_send_wr *next;  // Address of the next Send Request. NULL in the last Send Request
	struct ibv_sge *sg_list;   // Array of scatter/gather elements
	int num_sge;               // Number of elements in sg_list
	enum ibv_wr_opcode opcode; // The opcode to be used
	int send_flags;            // Send flags. Or of the following flags:

    /* IBV_SEND_FENCE â€“ Prevent process this Send Request until the processing of previous RDMA 
    //                   Read and Atomic operations were completed.
    //IBV_SEND_SIGNALED â€“ Generate a Work Completion after processing of this Send Request ends
    //IBV_SEND_SOLICITED â€“ Generate Solicited event for this message in remote side
    IBV_SEND_INLINE  - allow the low-level driver to read the gather buffers*/
	uint32_t imm_data;  // Send message with immediate data (for supported opcodes)
union {
	 struct {                           // Attributes for RDMA Read and write opcodes
		uint64_t remote_addr;      // Remote start address (the message size is according to the S/G entries)
		uint32_t rkey;             // rkey of Memory Region that is associated with remote memory buffer
	 } rdma;
	 struct {                           // Attributes for Atomic opcodes 
		uint64_t remote_addr;      // Remote start address (the message size is according to the S/G entries)
		uint64_t compare_add;      // Value to compare/add (depends on opcode)
		uint64_t swap;             // Value to swap if the comparison passed
		uint32_t rkey;             // rkey of Memory Region that is associated with remote memory buffer
	} atomic;
//â€¦
}
};
```

### Lesson 16: RDMA Receive Request
```c
//Post Receive Request
int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                  struct ibv_recv_wr **bad_wr);
//Post Send Request
int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                  struct ibv_send_wr **bad_wr);                  
/* Warning: bad_wr is mandatory; It will be assigned with the address of 
the Receive Request that its posting failed */
struct ibv_recv_wr {
	uint64_t wr_id;           // Private context, available in the corresponding Work Completion
	struct ibv_recv_wr *next; // Address of the next Receive Request. NULL in the last Request
	struct ibv_sge *sg_list;  // Array of scatter elements
	int num_sge;              // Number of elements in sg_list
};
```

### Lesson 17: RDMA Request Completion
```c
// Polling for work completion
int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc);
//Work Completion for each entry
struct ibv_wc {
	uint64_t wr_id;                // Private context that was posted in the corresponding Work Request
	enum ibv_wc_status status;     // The status of the Work Completion
	enum ibv_wc_opcode opcode;     // The opcode of the Work Completion
	uint32_t vendor_err;           // Vendor specific error syndrome
	uint32_t byte_len;             // Number of bytes that were received
	uint32_t imm_data;             // Immediate data, in network order, if the flags indicate that such exists
	uint32_t qp_num;               // The local QP number that this Work Completion ended in
	uint32_t src_qp;               // The remote QP number
	int wc_flags;                  // Work Completion flags. Or of the following flags:
     /* IBV_WC_GRH â€“ Indicator that the first 40 bytes of the receive buffer(s) contain a valid GRH
      IBV_WC_WITH_IMM â€“ Indicator that the received message contains immediate data */
	uint16_t pkey_index;
	uint16_t slid;                                // For UD QP: the source LID
	uint8_t sl;                                     // For UD QP: the source Service Level
	uint8_t dlid_path_bits;                      // For UD QP: the destination LID path bits
};
// typical completion statuses
/*
IBV_WC_SUCCESS â€“ Operation completed successfully
IBV_WC_LOC_LEN_ERR â€“ Local length error when processing SR or RR
IBV_WC_LOC_PROT_ERR â€“ Local Protection error; S/G entries doesnâ€™t point to a valid MR
IBV_WC_WR_FLUSH_ERR â€“ Work Request flush error; it was processed when the QP was in Error state

IBV_WC_RETRY_EXC_ERR â€“ Retry exceeded; the remote QP didnâ€™t send any ACK/NACK, even after
            message retransmission                                                                            
IBV_WC_RNR_RETRY_EXC_ERR â€“ Receiver Not Ready; a message that requires a Receive Request
           was sent, but isnâ€™t any RR in the remote QP, even after message retransmission
*/
```

### Lesson 18: Connection Establishment
- Connection-oriented transports require a connection establishment between the two hosts before the data could be sent
  - RC-ping-pong
  - Communication Manager

### Lesson 19: RDMA CM
- Connection manager for RDMA
- Glossary
  - rdma_connect() - connect to the remote server
  - rdma_accept() - accept the connection request
  - rdma_bind_addr() - set the local port number to listen on
  - rdma_listen() - begin listening for connection requests  
  - rdma_create_qp() - allocate a QP for the communication on the new rdma_cm_id
  - rdma_post_send() - post a buffer to send a message

### Lesson 20: Introduction to RCpingpong
- RC Ping-pong is a server-client application

### Lesson 21: RCpingpong context 1 (INIT)

### Lesson 22: RCpingpong context 2 (CLOSE)

### Lesson 23: RCpingpong Data-Path

### Lesson 24: RC Connection
