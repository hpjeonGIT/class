## Juniper JNCIA-Junos JN0-105
- Instructor: S2 Academy

## Section 1: Networking Basics

### 1. Introduction
- For JNCIA-Junos JNO-105
  - JNCIS - specialist level
  - JNCIP - professional level
  - JNCIE - expert level
- Certification is valid for three years
- Hands-on lab
  - Buy a physical juniper appliance
  - Or cloud instance  
    - SRX image
  - https://www.juniper.net/us/en/dm/vjunos-labs.html

### 2. OSI Model
- Open systems Interconnection
- 7 Layers
- Purely logical concept: layers do not directly represent physical elements
  - Application layer
    - Interface b/w applications and underlying network
    - Protocols of DNS, HTTP, FTP
  - Presentation layer
    - Formatting the data from the lower layers into well-known formats like jpeg, mpeg
  - Session layer
    - Responsible for establishing and controlling sessions b/w the sender and receiver
    - Responsible for synchronizing the session using sequence numbers
    - Protocols include NetBIOS, SOCKS, and NFS
  - Transport layer
    - Packets are transformed into segments
    - Responsible for providing a transport protocol like TCP or UDP
    - Responsible for process separation
  - Network layer
    - Frames are transformed into packets
    - Responsible for logical addressing of devices - IP addresses
    - Responsible for routing packets
    - Protocols of IPv4, IPv6, ICMP, IPsec
  - Data Link layer
    - Bits are transformed into frames
    - Responsible for communication over the LAN or the same network
    - Responosible for MAC address
    - Flow control and error control
  - Physical layer
    - Form of bits: zeros and ones
    - 3 types of communication
      - Simplex: one way like broadcasting
      - Half duplex: two ways but only send or receive at a time
      - Full duplex: two ways of send/receive

### 3. TCP/IP vs OSI model
- Divides network communication into four layers
  - by DoD
  - Application layer
  - Transport layer
  - Internet layer
  - Network access layer

### 4. Collision Domains, Broadcast Domains, and VLANs
- A collision domain refers to how many devices can send data at the same time
  - Need to minimize collision domain
  - A hub is a single collision domain: only one device can send data
  - A switch is a multiple collisions domain: all connected devices can send data at the same time
    - Each port is its own collision domain
  - A router: all connected devices can transmit at the same time
    - Each port is its own collision domain
- Broadcast domain is a logical division of network, in which all nodes can reach each other by broadcast
  - Broadcast never crosses LAN
  - On a hub, a broadcast reaches all devices
  - On a switch, a broadcast reaches all devices
  - On a router, a broad cast is confied to the connected port
    - Each port of a router might be a different LAN
- Broadcast may abuse the switch resources
  - How to manage efficiently? VLAN
- VLAN
  - A logical separation of devices on the same LAN
  - It allows you to divide a LAN segment into multiple logical LANs (Virtual LANs)
  - Each VLAN is a different broadcast domain
  - A single switch may be dividied into multiple VLANs, providing different broadcast domains
  - By default, VLANs do not talk to each other
  - A layer 3 device (router) is required route traffic b/w VLANs
  - Each VLAN is idenitified by a unique IEEE 802.1Q ID (aka tag)

### 5. Routers, switches, and other network devices
- Repeater
  - Layer 1 device, repeating signals
  - Receives a signal and trasmits it
- Hub
  - Layer 1 device that operates in half duplex mode
  - No intelligence 
  - Not good for security
- Bridge
  - Layer 2 device that learns MAC addresses
  - Uses a CAM table to store port and MAC address information
    - When a frame is received for the first time, the source port and MAC address is added to the CAM table
  - Frame forwarding is software-based
- Switch
  - Layer 2 device similar to bridge - learns MAC addresses
  - Specialized chips are used for frame forwarding, resulting in better performance
  - Supports VLANs
- Router
  - Layer 3 device that routes packets b/w different networks
  - Uses routing tables to make routing decisions

### 6. Layer 2 Addressing
- Device addressing
  - MAC address - layer 2 address
  - IP address - layer 3 address
- Devices need both addresses to reach a destination
  - MAC address identifies a device on the local network
  - IP address identifies a device outside the local network
- MAC Address
  - A 48-bit address that is burned on the network interface
  - Represented as 6 grups of 2 hexadecimal digits, separated by colons
    - Each hexadecimal character is represented by four bits, yielding a 4*6*2 = 48-bit address
  - The first three groups are together known as the Organization Unit Identifier (OUI)
    - Identifies the manufacturer
- Broadcast MAC address
  - FF:FF:FF:FF:FF:FF
  - Will reach all hoss on the same network

### 7. Introduction to IPv4
- 32-bit logical address assigned to a network device
- Four of 8 binary bits (octet)
- Class A: 0.0.0.0 to 127.255.255.255
- Class B: 128.0.0.0 to 191.255.255.255
- Class C: 192.0.0.0 to 223.255.255.255
- Class D: 224.0.0.0 to 239.255.255.255
  - Reserved for multicast purposes
- Class E: 240.0.0.0 to 255.255.255.255
  - Reserved for experimental purposes
- RFC 1918 Addresses
  - Private IP addresses
  - A block of IP addresses in each class A, B, and C
    - Class A: 10.0.0.0 to 255.255.255 (10.0.0.0/8)
    - Class B: 172.16.0.0 to 173.31.255.255 (172.16.0.0/12)
    - Class C: 192.168.0.0 to 192.168.255.255 (192.168.0.0./16)
  - Nobody owns these addresses
  - Allows organizations to use these private addresses for internal addressing
- Loopback address
  - The entire 127.0.0.0/8 address block is reserved
  - Common address is 127.0.0.1
    - Also known as localhost (itself)

### 8. Decimal to Binary Conversion

### 9. Subnet Mask and its importance
- What is subnet mask?
  - Consider 192.168.1.0/24
  - Every IP address has two parts - network portion and host portion
  - Every IP address is accompanied by a subnet mask
  - Two ways to denote a subnet mask
    - 10.0.0.0/8
    - 10.0.0.0/255.0.0.0
  - When the subnet mask is converted to binary, the 1's denote the network portion while 0's denote the host portion
    - 10.0.0.0/255.0.0.0
    - Decimal: 255.0.0.0
    - Binary: 11111111.00000000.0000000.000000
- Default subnet mask
  - Class A has a default subnet mask of /8
    - Ex: 122.0.0.0/8
  - Class B has a default subnet mask of /16
    - Ex: 172.16.0.0/16
  - Class C has a default subnet mask of /24
    - Ex: 192.168.10.0/24
- Total number of Hosts
  - Total number of possible host IP addresses on a network = 2^H
  - H represents the number of host bits (zeros) in a subnet mask
  - In a single subnet, The first address is the network address and the last address is the broadcast address of the network
    - Ex: 19.168.1.0/24
    - First address (network address): 192.168.1.0
    - Last address (broadcast address): 192.168.1.255
    - The first and the last addresses are not usable
  - Total number of usable IP addresses = 2^H - 2
- why do we need subnetting?
  - Breaks a large network into manageable pieces
  - /24: 254 clients
  - /30: 2 clients only

### 10. Subnetting Example 1
- Steps for subnetting
  - Convert the subnet mask into binary format
  - Determine the number of host bits to be borrowed
  - Determine the increment
  - Add increment to get the new subnets
- Ex: Divide 192.168.1.0/24 into 5 networks
  - Decimal:/24 - 255.255.255.0
  - Binary: 11111111.1111111.1111111.00000000
  - 2^X >= required number of subnets
    - 2^X >=5: X=3, number of bits to borrow = 3
  - New subnet mask is /24+3 = /27
    - 11111111.1111111.1111111.11100000
    - /27 = 255.255.255.224
    - Increment is the power of 2 corresponding to the least significant bit
      - /27 => 11111111.1111111.1111111.11100000
      - The last 1 is at 32 -> increment
    - Add increment to get new subnets
      - 162.168.1.0
      - 162.168.1.32
      - 162.168.1.64
      - 162.168.1.96
      - 162.168.1.128
      - 162.168.1.160
      - 192.168.1.192
      - 192.168.1.224
      - All these networks have a subnet of /27

### 11. Subnetting Example 2
- Divide 172.10.0.0/16 into 10 subnets
  - Decimal: /16 - 255.255.0.0
  - Binary: 11111111.11111111.0000000.00000000
  - 2^X >= 10, X= 4, number of bits to borrow = 4
  - Now the new subnet mask is /16+4 = /20
  - /20 = 11111111.11111111.11110000.00000000
    - /20 = 255.255.240.0
  - The last 1 is at 16 
    - Increment is 16
  - Add increment to get new subnets
    - 172.10.0.0
    - 172.10.16.0
    - 172.10.32.0
    - ...
    - 172.10.224.0
    - 172.10.240.0
    - All these networks have a subnet mask of /20

### 12. Subnetting Example 3
- Divide 124.0.0.0/8 such that each network has 500 hosts
  - Decimal:/8 - 255.0.0.0
  - Binary: 11111111.00000000.00000000.00000000
  - Determine the number of host bits to be fixed
    - 2^X >= required number of hosts + 2, X = 9
  - After fixing 9 host bits, remaining network bits are 23
  - 11111111.11111111.11111110.00000000 => /23
  - The last 1is at 2 of 3rd group
  - New subnets
    - 124.0.0.0
    - 124.0.2.0
    - 124.0.4.0
    - ...
    - 124.0.254.0
    - 124.1.0.0
    - 124.1.2.0
    - ...
    - 124.254.254.0
    - Total usable hosts per subnet = 2^H - 2 = 2^9 - 2 = 512 - 2 = 510

### 13. Subnetting Example 4
- Which subnet does 200.1.1.10/29 belong to?
  - /29 - 255.255.255.248, 11111111.11111111.11111111.11111000
  - Increment is 8
  - New subnets
    - 200.1.1.0
    - 200.1.1.8 <- this subnet owns 200.1.1.10
    - 200.1.1.16
    - ...
    - 200.1.1.248
- Network address of 200.1.1.10/29 is 200.1.1.8
- Broadcast address of 200.1.1.10/29 is 200.1.1.15

### 14. Supernetting
- Allows you to represent smaller networks as a single larger network
- Ex:
  - 10.4.0.0/16
  - 10.5.0.0/16
  - 10.6.0.0/16
  - 10.7.0.0/16
  - Convert the subnetted networks into binary format
  - Find the common bits
    - 00001010.00000100.00000000.00000000
    - 00001010.00000101.00000000.00000000
    - 00001010.00000110.00000000.00000000
    - 00001010.00000111.00000000.00000000
    - ^^^^^^^^^^^^^^^ are common bits
      - supernet address = 10.4.0.0
  - Set the common bits to 1's and others zero
    - 11111111.11111100.00000000.00000000
    - This is the new supernet mask
    - Supernet is 10.4.0.0/14
- Why supernet?
  - Helps with route summarization
  - Reduces the size of routing tables and routing updates

### 15. IPv6
- Successor of IPv4
- 128 bits length
  - 2^128 possible addresses
- IPv4 has 3 types of communication: unicast, multicast, broadcast
- IPv6 introduces Anycast
- Unicast: a single source to a single destination
- Multicast: a single source to multiple destinations
- Anycast: to the nearest node in the group
- Broadcast has been removed from IPv6
- IPv6 provides stateless address auto configuration
  - Can generate a unique address that can be used to communicate over the network
- IPv6 is not compatible with IPv4
  - Dual stack and tunneling
- IPv6 address format
  - 128 bits
  - 8 groups of hexadecimal format
- Rule
  - Leading zeroes in a group can be discarded
    - Ex: 2001:0db8:0000:0c3a:48b5:0000:0001:9177 == 2001:db8:0:c3a:48b5:0:1:9177
  - Any group of two or more zeroes can be replaced with :: but only once
    - Ex: 2001:0000:0000:0000:48b5:0000:0000:9177 == 2001:0:0:0:48b5:0:0:9177 == 2001::48b5:0:0:9177 == 2001:0:0:0:48b5::9177
- IPv6 addresses
  - Unpspecified: used when a computer boots up and has no address assigned. This address might be used before a computer gets an IP from DHCP
    - Denoted by ::/128
    - Equivalent to 0.0.0.0 in IPv4
  - Loopback: ::1/128
    - Equivalent to 127.0.0.1 in IPv4
  - Link local: assigned by the computer to itself from the FE80::/10 range
    - Unique on the subnet
    - Helps the host with automatic address configuration when static/DHCP address is not available
    - Similar to APIPA (Automatic Private IP Address) in IPv4
    - Used for communicating over LAN, not crossing them
  - Unique local: Communicating inside the LAN
    - Similar to RFC 1918 addresses in IPv4
    - Cannot be routed over the internet
    - Range is FC00::/7
  - Global unicast: Public address that can be routed over the internet
    - 2000::/3 to 3FFF::/3
  - Multicast: FF00::/8
    - Similar to 224.0.0.0/4 in IPv4
    - Cannot be used as the source address
- IPv6 subnetting
  - Routing prefix: 48 bits, network address. This is assigned by the internet service provider
  - Subnet: 16 bits
    - Total number of possible subnets for every routing prefix is 2^16
  - Interface ID: 64 bits, host bits
    - For every subnet, total number of hosts will be 2^64

### 16. Class of Service
- Delay is the amount of time it takes for a packet to reach from source to destination
- Jitter is defined as a variation in the delay of received packets and is generally caused by congestion in the IP network
- Class of Service
  - Provides differentiated services when best effort delivery is insufficient
  - With CoS, you can configure classes of service for different applications
  - Assign service levels with different delay, jitter, and packet loss characteristics
  - Implementd using 6bits in the IPv4 and IPv6 headers

### 17. Conneciton-oriented vs Connectionless protocols
- Connection oriented protocols
  - Requires a connection to be established before exchanging data
  - Ex: TCP
    - TCP uses 3-way handshake to establish a connection
    - TCP has features such as: ordered data transfer, retransmission of lost packets, flow control
- Connectionless protocols  
  - Data can be exchanged without setting up a connection
  - Ex: UDP
    - Has a lower overhead
    - Could have loss of data, errors, duplication, out-of-sequence delivery
    - Suitable for broadcasting

### 18. Slides used in ths course
- https://bit.ly/jncia-slides

## Section 2: Junos Fundamentals

### 19. Junos Software Architecture
- Junos kernel is based on FreeBSD Unix
- Junos functionality is compartmentalized into mulitple software processes
- Each process handles a portion of the device's functionality and runs in its own protected memory space
  - When a single process fails, it doesn't affect the entire process
- Control plane - Forwarding plane
  - Connected through internal link

### 20. Routing engine and packet forwarding engine (Control and forwarding planes)
- Routing engine
  - Brain of the device, responsible for performing protocol updates and system management
  - Based on x86 or PowerPC 
  - Maintains routing tables, bridging table, and primary forwarding table
  - Controls the interfaces, chassis components, system management, and access to the device
  - CLI and J-web GUI
  - Creates and maintains one or more routing tables. This is used to create the Forwarding Table that contains the active routes
- Packet Forwarding Engine
  - Usuallay runs on separat HW
    - Can be fault-tolerant
  - Responsible for forwarding transit traffic through the device
  - Receives a copy of the Forwarding Table (FT) from the RE using an internal link
  - Usually runs on separate HW, and in some cases uses Application Specific Integrated Circuits (ASICs) for increased performance
  - Also implements services such as rate limiting, stateless firewall filters, and Class of Service (CoS)

### 21. Protocol Daemons
- Each process that runs in its own protected memory space is known as a daemon
- Protocol Daemons
  - Routing protocol daemon (rpd)
  - Device Control Daemon (dcd)
  - Management Daemon (mgd)
  - Alarm Daemon (alarmd)
  - System Log Daemon (syslogd)
  - On the device, run `show system processes`

### 22. Transit and Exception Traffic
- Transit traffic
  - Traffic that enters and ingress port, is compared agains the forwarding table, and is finally forwarded out an egress port is called transit traffic
  - For the traffic to be forwarded, the forwarding table must have an entry for the destination
  - Transit traffic is handled only by the forwarding plane
  - Can be unicast or multicast
- Exception traffic
  - Exception traffic does not pass through the local device but requires special handling
  - Terminated by Junos device
  - Ex:
    - Packets addressed to the chassis, such as routing protocol updates, SSH sessions, ping, traceroute
    - TCP/IP packets with the IP options field
  - All exception traffic destined for the RE is sent over the internal link that connects the control and forwarding planes
  - Traffic traversing the internal link is rate limited to protect the RE from denial-of-service (Dos) attacks
  - This rate limiter is not configurable

## Section 3: User Interfaces

### 23. CLI Functionality
- Ways to access CLI
  - Out-of-band: using serial console port or management port
    - Use putty, 9600bit/s, 8 bits Parity None, 1 stop bits, Flow control Hardware
  - In-band: using telent or SSH
- Root user vs non-root user
  - Root account provides full administrative access to the device and is referred to as superuser
    - On a new device, root account has no password
    - Root account cannot be deleted
    - On login, shell mode
  - Non-root user
    - By default, there is no pre-exising non-root user
    - On login, operational mode
- Key features of CLI
  - Consistent command names
  - Question mark for completion
    - Junos responds with possible commands
  - Tab and space completion of commands
  - UNIX style utilities and keyboard sequences

### 24. CLI Modes
- Shell mode
  - Prompt ends with `%`
  - directly available when you login as root
  - Non-root user can start with `shell` command
- Operational mode: prompt ends with `>`
  - Run `cli` from shell mode
- Configuration mode: prompt ends with `#`
  - Run `configure` or `edit` from operational mode

### 25. CLI Navigation
- Question mark
  - `?`: shows all commands
  - `show ?`
  - `show interface ?`: shows command options related with interface command
- Spacebar completion
  - Can complete command/options/system defined variables
- Tab completion
  - Can complete user defined variables
- Keyboard shortcut
  - ctrl + a: moves to the beginning
  - ctrl + e: moves to he end
  - ctrl + w: erases word to the left
  - ctrl + u: erases entire line

### 26. CLI Help
- `help topic`: shows usage guidelines for the statement 
- `help reference`: shows summary information about the statement
- `help apropos`: lists all commands and help text that contains a particular string
- `help syslog`: shows information about system log messages
- `help tip cli`: shows a tip about using the CLI

### 27. Output Levels
- Terse: displays the least amount of information, as a list
- Brief: additional information about each element
- Detail: most of information about each element
- Extensive: all of information about each element
- Demo:
  - `show interfaces terse`
  - `show interfaces fe-0/0/* brief`
  - `show interfaces fe-0/0/* detail`

### 28. Filtering Output
- Keywords:
  - compare
  - count
  - display
  - except
  - find
  - last
  - match
  - no-more
  - save
- Demo
  - `show interfaces terse | count`
  - `show | compare old_config.txt`
  - `show | display set | except screen`
  - `show | display set | find polices`
  - `show | display set | match polices`

### 29. Active vs Candidate Configuration

### 30. Configure Command

### 31. More Navigation Commands

### 32. Configuration Hierarchy

### 33. Junos Commit

### 34. Junos Rollback

### 35. Junos Configuration Files

### 36. Junos Load Command

### 37. J-Web

## Section 4: Configuration Basics
