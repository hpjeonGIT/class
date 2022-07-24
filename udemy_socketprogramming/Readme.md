## Networking Concepts with Socket Programming - Acadmic Level
- Instructor: Abhishek

## Section 1: Introduction

1. ToC

2. Join Telegram Group

3. OSI Model Instruction
- 7 layers of theoretical OSI model
  - Application layer
  - Presentation layer
  - Session layer
  - Transport layer
  - Network layer
  - Data link layer
  - Physical layer
- OSI model doesn't define actual implementation
- Description of networking subsystem (or network stack)  
- A guide line
- Layer: logically complete functionality of a networking component
- Each layer has a specific function
- Functions of layers do not overlap each other
- Data/packets move across the layers of bi-directionally
- Actual TCP/IP stack
  - Application layer
  - Transport layer
  - Network layer
  - Data link layer
  - Physical layer
- Q:
  - Regarding protocols such as ARP,ICMP,HTTP,DNS, which OSI layer is correspondent?
  - What changes happen in the packet content when it traversess the layers of TCP/IP stack?

4. OSI model layers Functions
- Application layer: Ping, http, whatsapp, ...
- Transport layer: UDP, TCP, ...
- Network layer: IP, IPv6
- Data link layer: Ethernet
- Physical layer

5. TCP IP stack real world analogy

6. Data Encapsulation and Decapsulation - introduction
- Data encapsulation (sending machine)
  - Application layer: generates msg
  - Transport layer: attaches TP (transport) hdr (sending/receiving process info)
  - Network layer: attaches IP hdr (IP add of src/dest)
  - Datalink layer: attaches MAC hdr (current and next node MAC address)
  - Physical layer: places the data into wire
- Data decapsulation (receiving machine)
  - Physical layer: data arrives from wire
  - Datalink layer: detaches mac hdr and passes the rest
  - Network layer: detaches IP hdr and passes the rest
  - Transport layer: detaches TP hdr and passes the rest
  - Application layer: consumes the data

7. Data Encapsulation
- Each layer doesn't check the header from other layers

8. Data Decapsulation

9. Data Encapsulation and Decapsulation on Forwarding nodes
- TCP/IP stack, intermediate machines are involved in data link layer
  - Hop-by-Hop: 
- Terminology
  - Applicatin Layer: Data
  - Transport layer: segment
  - Network layer: Packet
  - Data linke layer: Frame
  - Physical layer: bits
  
10. A Big picture

## Section 2: Networking Labs

11. What in this section?
- Mininet is a network emulator
  - Deploy a network topology
- Learn how to: L2/L3/Vlan based/Inter Vlan routing

12. Objective and Goals

13. Mininet Installation Procedure
- VMWare workstation
  - Let's use VBox
- Mininet VM image
  - https://github.com/mininet/mininet/releases/
  - Needs VBox version 6.1 at Ubuntu18.04

14. Extra Tools Installation
- sudo apt-get install screen
- sudo apt-get install bridge-utils
- sudo apt-get install vlan
- ~/.screenrc
```
autodetach on
startup_message off
hardstatus alwayslastline
shelltitle 'bash'
hardstatus string '%{gk}[%{wk}%?%-Lw%?%{=b kR}(%{W}%n*%f %t%?(%u)%?%{=b kR})%{= w}%?%+Lw%?%? %{g}][%{d}%l%{g}][ %{= w}%Y/%m/%d %0C:%s%a%{g} ]%{W}'
```
15. Launching Mininet
- sudo mn
  - Adding 2 hosts of h1 and h2
  - Adding a switch of s1
- nodes
- net
- h1 ifconfig -a
- h1 ping h2
- h1 ping all

16. Mininet TreeTopologies
- sudo nm -topo=tree,depth=2,fanout=2
```
c0 ─ s1 ─ s2 ─ h1
             └─ h2
        └─ s3 ─ h3
             └─ h4
```
- h1 ping h2
- May use python code for more complex structures

## Section 3: IP Subnet

17. Subnetting Part 1
- How a subnet is formed
- L3 Routing: routing done by layer 3, the network layer
- L2 Routing: routing done by layer 2, the datalink layer
- Subnetting: dividing a network into two or more networks
- Subnet: layer 2 network. Data link layer takes care of data transfer
  - In the same subnet, **datalink layer** is responsible for data delivery
  - When communicating pair is located in different subnets, **network layer** is responsible for data communication. Source subnet vs destiny subnet.
- If 192.168.4.2 connects to 192.168.1.1, it will use L3 routing
- If 192.168.4.2 connects to 192.168.4.1, it will use L2 routing
- Note
  - Machines do not have IP/MAC addresses. Only Interfaces have
  - We send data to a particular interface of a remote machine, not a subnet

18. Subnetting Part 2
- Each interface of L3 router is a subnet

19. Data Delivery
- If a packet has matching IP with one of local interface, the receiving machine will consume the packet
- If not matching, the machine forwards the packet to the corresponding subnet using L2 routing
- Broacast domain: a single wire connecting multiple machines below a router
  - P2P domain for b/w two machines
  - All devices connected with the same wire (broadcast domain) forms one subnet
  - A packet sent by one machine (in the same broadcast domain) is heard by all other machines in the same subnet
    - Bit signals sent are heard through physical layer to all the machines in the broadcast domain
- L3 routing works only when a sending packet reaches a machine outside of the subnet
- For communication within a single subnet, L3 router does nothing

20. Mac and IP Address
- Within a subnet, data transmission b/w machines is done using MAC addresses
  - IP addesses do NOTHING
  - Datalink layer operates with MAC addresses only
- MAC addresses are called as Layer 2 addresses
- IP addresses are called Layer 3 addresses
- MAC address is a 6byte number or 6 octets
- IP address is 4bytes or 4 octets

21. Network ID
- Every IP address is associated with a mask, which is a vlue [0,32] in ipv4. In IPv6, [0,128]
- Mask is used to for a subnetwork
- Every subnet has:
  - Network ID
  - Broadcast Address
  - Max no. of available machines in the subnet
- Example
  - Machine B has IP as 11.1.1.3 and mask=24 (11.1.1.3/24)
  - Subnet/network ID = 11.1.1.0/24
  - Broadcast address = 11.1.1.255
  - Max N = 2^(32-24) - 2 = 254
- Steps to calculate subnet ID from IP/MASK
  - Represent IP address in binary format
    - 11.1.1.3=> 00001011 00000001 00000001 00000011
    - For 24 MASK, create a number whose 1st 24bits are 1 and others are zero
      - 11111111 11111111 11111111 00000000
    - Perform AND operation 
      - 00001011 000000001 00000001 00000000 => 11.1.1.0
      - This is the subnet ID (11.1.1.0/24)

22. Broadcast Addresses
- Steps to calculate Broadcast address
  - Get the network ID
  - Find M' = complement of mask M=24
    - 00000000 00000000 00000000 11111111
  - Perform OR operation of M' with Network ID
    - 00001011 00000001 00000001 11111111 => 11.1.1.255
    - This is network Broadcast address (11.1.1.255/24)

23. Max Value and Control Bits
- Given Mask = 24
- Max = 2^(32-24) - 2 = 254
- What limits the number?
  - DOF of control bits
- Control bits
  - The left over of bits after Mask (24) bits
  - In 11111111 11111111 11111111 00000000, 00000000 are control bits
- Generating assignable IP addresses for subnet devices
  - Perform logical OR with control bits only with network ID. This is the assignable IP address for a subnet
  - When all of control bits are zero, this is Network ID
  - When all of control bits are 1's, this is broadcast address
  - This is why we subtract 2 from the formula

24. IP Address Configuration
- Scenario
  - A router X is connected to a subnet of 11.1.1.0/24
    - Machine A is connected as 11.1.1.2/24
    - Machine B is connected as 11.1.1.3/24
    - Machine C is connected as 11.1.1.4/24
  - What happens if C's ip is changed as 11.1.2.4/24?
    - Separate subnets must be connected to a L3 router
    - No router b/w C and other machines
    - This is illegal and C cannot communicate with others
  - To enable C:
    - Add another router Y to the router X (11.1.1.6/24), as a neighbor of A & B
    - Then add C to the router Y

25. Point to Point Links Mask
- P2P ends are always in the same subnet
- This is a straight wire b/w two ends
- Max no. of devices with mask=24 is 254
- When **mask=30**, only 2 interfaces are allowed
  - 11.1.1.1/30 and 11.1.1.2/30: available IP addresses
  - 11.1.1.0/30: Network ID
  - 11.1.1.3/30: broadcast address

26. Broadcast Addresses in Detail
- All receivers in the domain should receive the packet through broadcast addresses
- Two types of broadcast addresses
  - MAC layer broadcast addresses: ff:ff:ff:ff:ff:ff
  - Networklayer broadcast address: from the previous lecture (ex: 11.1.1.255)
- Datalink layer uses MAC layer rule
  - If dest_MAC == MAC of receiving address or Broacast address, then accepts the frame. If not, rejects
- Network layer uses IP layer rule
  - If dest_ip == IP of receiving interface or broadcast address, it accepts the packet. If not, rejects

27. IP Maths Coding Assignments
- http://iodies.de/ipcalc
- http://www.silisoftware.com/tools/ipconverter.php

## Section 4: Layer 2 Routing

28. L2 Routing Introduction
- Routing done at datalink layer
- Only MAC address is used to deliver the frame to destination machine
- L2 routine:
  - routing within a subnet
  - From one machine to another machine within a subnet
  - Or from L3 router to machine present in a directly connected subnet

29. Local and Remote subnets
- Same subnet: when a set of interfaces of devices interconnected with one another and has IP addresses configured having same Network ID
- A subnet can be local or remote relative to a given router

30. L2 Routing - Basics
- Router is a gateway to a subnet
- Machines in different subnets cannot communicate without L3 routing (why L3?, not L2?)
- Protocol identifier field
  - Datalink layer -> Ethernet hdr (type)
  - Network layer -> IPV4/IPV6 hdr (protocol)
  - Transport layer -> UDP/TCP hdr (port no)
  - Application layer -> Application hdr

31. Ethernet Header Format
- Ethernet frame
  - Preamble (8 bytes) + Dest Address (6bytes) + Src Address (6bytes) + type (2bytes) + info [46-1500] + FCS (4bytes)
  - Ref: https://en.wikipedia.org/wiki/Ethernet_frame

32. How layer 2 Routing is done?

33. ARP Goals

34. ARP Standard Message Format

35. Address Resolution Protocol Part 1

36. Address Resolution Protocol Part 2

37. Address Resolution Protocol Demonstration

38. Layer 2 Switch Concept

39. L2 Switch Functioning

40. Layer 2 Switch Example

41. Lab Session

42. Test Topology Description

43. L2 Topology Demo

44. L2 Topology Assignment
