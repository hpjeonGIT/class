## (NEW 2021) CompTIA Network+ (N10-008) Video Training Series
- Instructor : Kevin Wallace & Charles Judd

## Section 2: Module 1 - Introducing Reference Models and Protocols

3 OSI model
- Open Systems Interconnect model
- 7 layers
  1) Physical: bits. Speed, voltage, length, timings of devices/cabling. Hub/Router
  2) Data Link: Frames. Based on MAC (Media Access Control) address. Bridges & Switches
  3) Network: Packets (or Datagrams). IP/ICMP/IPv6. Routes b/w networks(subnets)
  4) Transport: Segments. TCP/UDP
  5) Session: how to open/close/maintain sessions b/w network devices for data transfers
  6) Presentation: Encryption, ASCII, EBCDIC, and compression
  7) Application

4 TCP/IP model
- Different layer model
  1) Network Access (physical + datalink)
  2) Inernet (network)
  3) Transport
  4) Application (session + presentation + application)

5 IP, ICMP, UDP, and TCP
- ICMP: for ping
  - When PC cannot connect to internet, we may check through ICMP/ping
  - Can ping through PC->Switch->Router
```bash
$ ping 192.168.1.1
PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.
64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=6.49 ms
```
- UDP (User Datagram Protocol)
  - Not reliable
  - Voice over ip-phone
- TCP (Transmission Control Protocol)
  - Reliable
  - Connection oriented
  - 3 way handshake
    - Synchronization (SYN): pc -> server
    - Synchronization and Acknowledgement (SYNC-ACK) : server -> pc
    - Acknowledgement (ACK): pc->server
  - PC: sending segments in 1->2->4->8 ...
    - Server responds ACK to confirm the size of segments
    - If there is any loss, PC will resend in the reduced (?) size

6. IP, UDP, and TCP headers
- TCP header
  - Source port: a 16-bit field identifying the sending port
  - Destination port: a 16-bit field identifying the receiving port
  - Sequence number: a 32-bit field specifying the first sequence number if the SYN flag =1. Or the accumulated sequence number if the SYN flag = 0
  - Acknowledgement number: a 32-bit field specifying the next sequence number the sender of an ACK expects
  - Data offset: a 4-bit field specifying the size of TCP header
  - Reserved: A 3-bit field for future use
  - Flags: A series of nine 1-bit fields
  - WindowSize
  - Checksum
  - Urgent Pointer: 
  - Options: A field in the range of 0-320bits
- UDP header
  - Source port
  - Destination port
  - Length
  - Checksum
  - No sequence number/No acknowledgement number - this is why UDP is told to be not reliable (still error check is done through checksum)
- IPv4 header
  - Version
  - IHL
  - Type of Service (ToS)
  - Total Length
  - Identification
  - Flags
  - Fragment Offset
  - Time to Live (TTL)
  - Protocol
  - Header checksum
  - Source IP Address
  - Destination IP Address
  - Options
- IPv6 header
  - Version
  - Traffic class
  - Flow label: a 20 bit field identifying a group of packets 
  - Payload length
  - Next header
  - Hop Limit: prevents routing looops. TTL field of IPv4
  - Source Address
  - Destination address

  7. Maximum Transmission Unit (MTU)
  - The largest frame or packet that can be transmitted or received on an interface
  - If sender's MTU packet size is too big
    - Receiver may fragment the packet into smaller MTU pieces
    - For DF (Don't Fragment) Bit:
      - IPv4 will send ICMP message of "Fragmentation needed ad DF set" to the sender 
      - IPv6 will uses "Packet Too Big" ICMPv6 message

8. Ports and Protocols
- For http, header shows destination port as 80
- Well known ports: 0-1023 offer well-known network services
- Registered ports: 1024-49151. Registered with internet access number authority (IANA)
- Ephemeral ports: 49152-65535 (Dynamic or Private ports)

|Protocol                       | TCP Port  | UDP Port |
|-------------------------------|-----------|----------|
|ftp  (20 for data and 21 for control) | 20 and 21 |          |
|ssh                            | 22        |          |
|sftp                           | 22        |          |
|ftps (ftp over SSL or TLS)     |989 & 990  | 989 & 990|
|telnet                         | 23        |          |
|SMTP                           | 25        |         |
|SMTP over SSL/TLS              | 587       |         |
|DNS                            | 53        | 53      |
|tftp  (trivial FTP. not secure)|           | 69      |
|DHCP                           |           | 67      |
|http                           | 80        |         |
|https                          | 443       |         |
|pop3                           |110        |         |
|pop3 over SSL/TLS              |995        | 995     |
|NTP (sync over time server)    |           | 123     |
|IMAP                           |143        |         |
|IMAP over SSL/TLS              |993        |         |
|LDAP                           |389        |         |
|SNMP (monitor/manage network devices. Temperature/humid data ....)      | 161 & 162 |161 & 162|
|RDP (Remote desktop protocol)  | 3389      |         |
| SIP (voice/video call)        |5060 & 5061|5060 & 5061|
|H.323 (voice/video calls)      |1720       |         |
|SMB (Server Messge Block)      |           | 445     |
|SQL server (by MS)             |1433       | 1433    |
|SQLnet (by Oracle)             | 1521      |         |
|MySQL                          |3306       |         |

## Section 3: Getting help

9. Answering your questions
- http://discord.davidbombal.com

10. Udemy tips and tricks

## Section 4: Module 2 - Network pieces and parts

11. Network pieces and parts

12. Analog modems
- Legacy PSTN (public switched telephone network)
  - PC - modem - PSTN - modem - server
- Modem: Modulator/Demodulator
- Baud: number of tone changes per second
- Bits per second (bps): number of 1s and 0s that can be transmitted over the line
  - 2400 pbs: 2400 baud using one channel
  - 28.8 kpbs: 2400 baud using 12 channels

13. CSMA-CD vs CSMA-CA
- Carrier-Sense Multiple Access with Collision Detection (CSMA/CD)
  - Ethernet bus: packets will collide two pcs try to send packet at the same time
  - Random back off timer
  - May use a hub as intermediate traffic controller
- Carrier-Sense Multiple Access with Collision Avoidance (CSMA/CA)
  - Hidden node problem: may assume there is no collisioin

14. Hubs, switches, and routers
- Ethernet hub
  - Can't differentiate b/w MAC addresses
  - Non-intelligent devices. Each incoming bit is replicated on all other interfaces
- Ethernet switch
  - Using MAC address table, 
  - Frame Flooding (unicast): the switch sends the incoming frame to all occupied and active ports. When a switch pretends to be a hub 
- Router
  - Has IP routing table
  - Default route: 0.0.0.0/0
    - A route used when a router does not have a more specific routing table entry for the destination network

15. Collision and Broadcast domains
- Collision domain: a network segment on which only one packet is allowed at any one time
  - Switch & bridge break up collision domain
  - All ports on a hub belong to one collision domain
    - Half duplex: allows a device on a network segment to either transmit or receive packets
  - Each port on a switch belong to its own collision domain
    - Full duplex: allows simultaneous transmission and reception of packets on a network segment
- Broad cast domain: an area of a network throughout which a broadcast can travel (subnet or a VLAN)
  - All ports on a hub belong to one broadcast domain
  - All ports on a switch belong to one broadcast domain
  - Each port on a router belongs to its own broadcast domain

16. Wireless Access points (AP)
- Contains one or more antennas for communicating with wireless devices

17. Firewalls
- Types
  - Packet filter
  - Stateful firewall: packet inspection
  - Next Generation Firewall (NGFS/layer 7 firewall): deep packet inspection.
- Demilitarized Zone (DMZ)
  - Server on our side but is connected into outside internet

18. Intrusion detection and prevention
- Intrusion Detection System (IDS) Sensor
  - Gets a copy of traffics and Detects the signature of the attack/pattern of packets
  - Dynamically updates firewall
  - But clients can be attacked while analyzing
- Intrusion Prevention System (IPS) Sensor
  - Inspects and reacts to traffic received in-line
  - IPS Sensor is located ahead of router

19. VPN Concentrators
- A dedicated HW which handles the encryption of VPN

20. Load Balancers

21. Advanced filtering appliances
- Next Generation Firewall (NGFS/Layer 7 firewall): An application layer firewall with additional features, such as deep-packet inspection(DPI), Intrusion Prevention System (IPS),and encrypted traffic inspection
  - Ransomware attrack through TLS layer
- Content filter
- Unified Threat Management (UTM) appliance: a dedicated appliance that combines multiple filtering functions such as firewall, IPS, Anti-malware, VPN, and Content filter

22. Proxy server
- Receives internet-bound traffic from local clients and creates its own connection to the intended destination
- Can be used to filter content
- Can be used for caching
- Some proxies require configuration on the client apps

## Section 5: Module 3 - Stay on top of your topologies

23. Describing network topologies

24. Start Topology
- If one link fails, other links continue to work
- Centralized device is a potential single point of failure
- Popular in modern network 

25. Mesh Topology
- Full mesh: topology where each site connects to every other site
  - Number of connections: n*(n-1)/2
  - Not scalable
  - Optimal path
- Partial mesh: each site connects to at least one other site but might optionally connect to other sites
  - Might be suboptional path
  - More scalable

26. Ring Topology
- CSMA/CD
- Token ring: a legacy LAN technology that used a ring topology and had bandwidth options of 4 Mbps or 16 Mbps
- Fiber Distributed Data Interface (FDDI): a legacy LAN technology that operated at 100Mbps and used two counter-rotating rings (to provide fault tolerance) and used fiber optic calbing for its transmission
- Media Access Unit (MAU)
  - Physically it looks like a star topolgy but in functions, it is a ring topology

27. Bus Topology
- 10BASE2 (thinnet): an older ethernet technology using a thin coaxial cable that had a distance limitation of 185m and a badwidth of 10 Mpbs
- 10BASE5 (thickent): an older ethernet technology using a thick coaxial cable that had a distance limitation of 500m and a bandwidth of 10 Mbps
- Logical vs Physical topologies
- The physical topology for a hub is a star but a hub uses a bus as logical topology

28. Point-to-point topology
- Interconnects two devices
- Typically uses a Layer-2 protocol
- Could be a physical/logical point-to-point connection

29. Point-to-multipoint topology

30. Hybrid topology

31. Client-server network
- Client-server architecture
- Clients access a common server
- Server shares resource of file/printer... with clients

32. Peer-to-Peer network
- Clients sahre resource of file/printer...
- Not robust as using a network operating system (NOS)

33. Local Area Network (LAN)
- High speed
- Centrally located

34. Wide Area Network (WAN)
- Typically slower than LANs
- Geographically dispersed sites
- Sites connect to service provider

35. Metropolitan Area Network (MAN)
- Limited Availability
- Very high speed (>10Gbit)
- Redundant
  - Ring topology and reverse direction is available

36. Campus Area Network (CAN)
- High speed
- Interconnects Nearby Buildings
- Easy to add redundancy
  - Partial mesh topology

37. Personal Area Network (PAN)
- Interconnects two devices
  - Bluetooth, infra-red
- Limited distance
- Limited throughput

38. Wireless LAN (WLAN)
- Adds flexibility and mobility for connections
- Wireless clients typically communicate with a wireless access point (AP)
- Channels should be selected to minimize interference

39. Software-Defined WAN (SD-WAN)
- Traditional WAN connections
  - Connected remote sites back to a central site over various WAN technologies
  - Predictable performance and security
  Modern SD-WAN connections
  - Applications on cloud
  - Provides security, QoS, and forwarding
  - The physical topology in an SD-WAN network is called an Underlay Network
  - The logical topology is called an Overlay Network
  - SD-WAN controller can simultaneously send out appropriate configuration commands to routers to provide consistent QoS, security, and predictable performance

40. Industrial Control Systems (ICS) and SCADA
- Supervisory Control And Data Acquisition (SCADA)
- Remote Telemetry Unit (RTU): a SCADA component that can receive information from a SCADA sensor and send instructions to a SCADA control
- SCADA master: A SCADA component that uses a communication network to receive information from on or more RTUs and send instructions to those RTUs

## Section 6: Module 4 - Understanding Network Services

41. Understanding network services

42. Virtual Private Networks (VPNs)
- How to send secure data over insecure network
- Allows client to connect to any internet connection
- Securely use a web browser
- Install a software client
- A split tunnel can be used to keep local traffic from flowing over VPN
- Site-to-Site VPN
  - Can use common broadband technology
  - Transparent to the client devices
  - Can use routers or dedicated VPN concentrators
- Generic Routing Encapsulation Tunnel
  - Does not provide security
  - Can encapsulate nearly any type of data
- IP Security (IPsec)
  - Encryption, Hashing,PSKs or Digital Signature, Applies Serial numbers to packets
  - Can encapsulate unicast IP packets (not flexible)
  - Two modes
    - Transport mode: uses Packet's original header
    - Tunnel mode: Encapsulates entire packet
  - Authentication and Encryption
    - Authentication Header (AH): no encryption
    - Encapsulating Security Protocol (ESP)
- GRE over IPsec

43. Dynamic Multipoint VPNs (DMVPN
- Dynamically forms tunnels b/w sites. Relies on IPSec, mGRE, and NHRP
  - Multipoint GRE (mGRE): Allows a single interface to support multiple GRE tunnels
  - Next Hop Resolution Protocol (NHRP): Used to discover the IP address of device at the far-end of a tunnel

44. Web Services

45. Voice Services
- Voice over IP (VoIP)
- Private Branch Exchange (PBX): Privately owned phone system used in large organization
  - 6000 phones in a campus while 200 lines are connected to phone company
- IP Telephony
  - Call Agent: replacement of PBX
  - Session Initiation Protocol (SIP): A signaling protocol that can setup, maintain, and tear down a session
  - Real-time Transport Protocol (RTP): transport layer protocal that carries voice and video media. Based on UDP

46. DHCP
- Process
  - Laptop -> Discover (Broadcast) -> DHCP server
  - DHCP server -> Offer (Unicast) -> Laptop
  - Laptop -> Request (Broadcast) -> DHCP server
  - DHCP server -> Acknowledgement (Unicast) -> Laptop
- DHCP features
  - MAC Reservation: specific MAC address gets specific IPs
  - Pools (Scopes): range of IPs
  - IP Exclusion: Excluded IPs
  - Scope Options: Default gateway, DNS server, TTL, Option 150
  - Lease time: will renew when necessary
  - DHCPv6

47. DNS
- Fully Qualified Domain Name (FQDN): a complete NDS name that uniquely identifies a host
- Hierarchical DNS structure
  - Root domain
  - Top-level domains (TLDs): .com, .edu, 
  - Second level domains: 
  - Sub-domain
  - Hosts
- Other DNS terms to know
  - Authoritative Name server: the server to which a top-level domain (TLD)   server will forward requests
  - DNS Zone transfer: the process of transferring DNS zone updates from a primary DNS server to a secondary DNS server
  - Revere lookup: 
  - Internal DNS server
  - External DNS server
- DNS Record type
  - A: an address record of IPv4
  - AAAA: IPv6 address
  - CNAME: a canonical name record
  - MX: a mail exchange record maps a domain name to an email
  - PTR: a poiner record points to a canonical name
  - SOA: a start of authority record provides authoritative information about a DNS zone
  - TXT: a text record intended to contain descriptive text
  - SRV: a service locator record
  - NS: a name server record

48. NAT
- Network Address Translation (NAT) theory
  - IPv4 addresses are consumed all
  - RFC 1918 addressing: A request for comments (RFC) publication that specifies private IPV4 address spaces. Specifically, IPv4 has the following IPv4 spaces that can be routed within an orgnization but not on the public internet: 10.0.0.0/8, 172.16.0.0/12, and 192.168.0.0/16
  - Router R1's NAT translation table
    - Maps inside local address(10.1.1.1) into inside Global address (192.0.2.101)
- Port Address Translation (PAT) 
  - In addition to the inside local address, it registers the port number
  - How cable model shares multiple connections inside of a house
- Port Forwarding
  - How to reach an inside server through a single global address?
  - We may map a static NAT using the port number requested

49. NTP
- Network Time Protocol (NTP)
  - Network devices need accurate time
  - To use digial certificates
  - Uses UDP port 123
  - Uses a stratum number to measure the believability of a time source  

50. SDN
- Overview of Software Defined Networking (SDN)
  - Mostly written in Python
  - JSON/XML data structure
- Southbound interfaces (SBI): when SDN controller contacts network devices
- Northbound interfaces (NBI): when SDN controller contacts applications
  - RESTful APIs: like web-server

51. IoT
- Supporting technologies
  - Z-wave: connects to nearby devices
  - Zigbee: faster than Z-wave
  - ANT/ANT+: 
  - Bluetooth
  - Near-Field communication (NFC): smartphone pay
  - Infrared (IR)
  - Radio Frequency Identification (RFID)
  - IEEE 802.11 (WiFi)

52. SIP Trunks
- Session Initiation Protocol (SIP) Trunks
  - Saves on long distance charges and PSTN connections
  - Allows a PBX and an IP telephony system to co-exist
- Real time transport protocol (RTP): a transport layer protocol that gets encapsulated inside UDP segments and is used to transmit voice and video media

## Sectoin 7: Module 5 - Selecting WAN Technologies

54. Packet switched vs circuit switched network
- Circuit switched
  - Traditional phone
  - Voice, data/video is sent over the circuit
  - ISDN
  - Paying over the time circuit is built
  - Dedicated bandwidth
- Packet Switched
  - Always-on
  - voice,data/video is encapsulated in packets
  - Cable model/LAN/wireless network
  - Shared bandwidth

55. Cellular
- 1G: delivered analog voice
- 2G: GSM, CDMA
- 2.5G: Added packet switching with GPRS (General Packet Radio Service)
- 2.75G (EDGE): Increased ata rates with EDGE (Enhanced Data RAtes for GSM Evolution)
- 3G: UMTS (Universal Mobile Telecommunication System) and CDMA2000
- 4G: at least 100Mbps  download speed to qualify as 4G
- 4G LTE: 20 Mbs - 100 Mpbs
- 5G: Higher speed and low latency. mmWave (5Gpbs) and Sub-6GHz

56. Frame Relay
- Two ways of connecting branch offices
  - Dedicated lines 
  - Virtual lines
    - DLCI (Data Link Connection Identifier)

57. ATM
- Asynchronous Transfer Mode (ATM)
  - Legacy method instead of using packets
- ATM cell structure
  - 5 bytes of header
  - 47 Bytes of payload
- VPI (Virtual Path Identifier)/VCI (Virtual Circuit Identifier): uniquely identifies a virtual connection that ATM uses to transport its cells
- UNI (User to Network Interface): Interconnects a user's device with an ATM network
- NNI (Network to Network Interface): Interconnects ATM network

58. Satelite
- VSAT (Very Small Aperture Terminal)
- Two way satelite communication
- Satelite dish is less than 3 meters
- Useful for locations that cannot have a wired internet connection
- 12-100Mbps
- Data experiences more delay
- Senstivie to weather condition

59. Cable
- Cable modem through Cable company
- Hybrid Fiber-Coax(HFC) distribution network: a cable company's infrastructure including both fiber and coax
- Data-Over-Cable Service Interface Specification (DOCSIS): A set of standards specifying the use of different frequency ranges in a cable television network

60. PPP
- Point-to-Point Protocol
- Authentication (PAP/CHAP)
- Compression 
- Error detection and correction
- Multiple links
  - MLP (Multilink PPP)
- Authentication method
  - PAP (Password Authentication Protocol): sends login credentials across the network. Not recommended as it is text
  - CHAP (Challenge Handshake Authentication Protocol): Sends a hash of login credentials across the network

61. PPPoE
- Digital Subscriber Line (DSL) with PPP over Ethernet (PPPoE)
- Ethernet doesn't have feature of authentication
- PPPoE enables authentication

62. DSL
- Became popular as it can use existing phone lines
- DSL Access Multiplexer (DSLAM): a piece of equipment typically located in a telephone central office, that aggregates DSL connections coming in from multiple subscribers
- ADSL (Asynchronous Digital Subscriber Line):
- Load coils: electrical components that create inductance
  - DSL cannot pass 18,000 feet due to load coils

63. Leased lines
- Digital circuits connected in two locations
- Bandwidth
  - T1: 1.544 Mbps
  - E1: 2.048 Mbps
  - T3: 44.736 Mbps
  - E3: 34.368 Mpbs
  - OC-1: 51.84 Mbps
  - OC-3: 155.52 Mbps
  - OC-12: 622.08 Mbps
- T1 & T3 popular in north america and japan
  - E1 and E3 otherwise
- T1
  - Time-Division Multiplexing (TDM)
  - Time sharing among channels
  - 24 channels with 8 data bits peer channel
  - 1 framing bit per frame
  - 24*8 + 1 = 193 bits per frame
  - 193 bit frames * 8000 samples per second = 1.544 Mbps
- Framing
  - CAS: Channel Associated Signaling
  - CCS: Common Channel Signaling
- E1
  - 32 channels
- Line coding
  - Alternate Mark Inversion (AMI)
  - Bipolar Eight-Zero Substition (B8ZS)

64. ISDN
- Integrated Services Digital Network
- Basic Rate Interface (BRI)
  - 1 D channel - 16kbps (Signaling)
  - 2 B channels - 64 kbps (Bearer)
- T1-PRI (Primary Rate Interface)
  - 1 D Channel - 64 kbps (Signaling)
  - 23 B Channels - 64 kbps (Bearer)
- E1-PRI
  - 1 D Channel - 64 kbps (Signaling)
  - 30 B Channels - 64 kbps (Bearer)

65. MPLS
- Multiprotocol Label Switching
  - L2 Header + L3 Header + Payload
  - 32 bit Shim Header b/w L2 and L3 headers
  - Make routing decision based on label, not IP
  - Label info at Shim header
- Customer Edge routers 
- Provider Edge routers
- Label Switch routers

66. Metro Ethernet
- Metropolitan Area Network (MAN)
- Configuration
  - Pure Ethernet
  - Ethernet over SDH (Synchronous Digital hierarchy)
  - Ethernet over MPLS
  - Ethernet over DWDM (Dense Wavelength-Division Multiplexing)

## Section 8: Module 6: Connecting Networks with Cables and Connectors

67. Connecting networks with Cables and connectors
- Coaxial cable, twisted pair cable
- RJ-45, RJ-11
- ST, SC, LC, MTRJ

68. Copper cables
- Coaxial cable
  - Ensulated to prevent EMI(Electromagnetic Interference)
  - Impedance: a circuit's opposition to traffic flow
  - RG-59,6,58,8/U
  - Twinaxial cable: has 2 inner conductors. Mostly used in data centers. 40 or 100Gpbs. 7 meters limitation
- Twisted pair cable
  - Unshielded twisted pair (UTP)
  - Shielded Twisted Pair (STP)
  - Plenum-Rated: will not produce toxin at fire. Must be used at AC or HVAC
  - Category 3: 100BASE-T and 100BASE-T4, 100m
  - Category 5: 100BASE-TX and 1000BASE-T4, 100m
  - Category 5e: 100BASE-TX and 1000BASE-T4, 100m
  - Category 6: 1000BASE-T, 5GBASE-T and 10GBASE-T, 100m (55m for 10GBASE-T)
  - Category 6a: 1000BASE-T, 5GBASE-T and 10GBASE-T, 100m
  - Cateogry 7: 5GBASE-T, 10GBASE-T, and POTS/CATV/1000BASE-T, 100m
  - Category 8: 25GBASE-T and 40GBASE-T, 30-36m (intended for data centers)

69. Fiber cables
- Immune to EMI
- Single mode Fiber (SMF)
- Multi mode fiber (MMF)
  - Multimode Delay Distortion: length is limited as 2km

70. Copper connectors
- RJ-11
  - Telephones, modems, and fax
  - 6 positions with 2 conductors
    - 2 different telephone lines
  - RJ-14 has 6 positions with 4 conductors
- RJ-45
  - Commonly for ethernet cables
  - 8 positions with 8 conductors
- DB-9 (aka serial port) and DB-25
  - Used with older serial connections (model, serial pointer, console on Unix host, mouse)
- F-type
  - For Cable modems
  - Commonly used with RG-6 and RG-59 coaxial cable
- BNC
  - 10 BASE-2
  - Carries radio frequencies for a variety of electronic gear

71. Fiber connectors
- ST (Straight tip): for MMF with a bayonet end
- LC (Luscent connector): for SMF
- SC (Subscriber Cable)
- MTRJ : MMF without bayonet end
- Structure of connectors
  - Ultra Physical Contact (UPC)
  - Angled Physical Contact (APC): not flat surface. 8 degree angle

72. Media converters
- Single-mode fiber to ethernet
- Multi-mode fiber to ethernet
- Fiber to coaxial
- Single-mode fiber to multimode fiber

73. Transceivers
- Gigabit Interface Converter (GBIC)
- Small form factor pluggable (SFP) transceiver
- SFP+ (for 10Gbps)
- Quad SFP+ (40Gbps)
- Bidirectional transceiver (BiDi Transceiver)

74. Termination Points
- 66 Block: PBX
- 110 Block: for cat6.
- Punch down tool
- Patch panel
- Demarcation point: where network maintenance responsibility passes from the WAN provider to the customer
- Smart jack: a network device that can perform diagnostic tests on the connected circuit

75. Cabling Tools
- Crimper: cat6 cable + RJ45
- Cable tester: verifie the integrity of a cable
- Continuity tester: determines if an electrical path exists b/w two end points.
- OTDR(optical time domain reflector): Can locate fiber breakage
- BERT: checks error rates
- Light meter
- Loopback adapter: verifies the integrity of a port.
- speedtest.net
- Wire map tester
- Tone probe: determines a device is physically connected

76. Punch-down blocks
- 66 Blocks: typically supports cat3. Some options support cat 5e
- 110 block : supports cat6a and lower
- Krone: european alternative to a 110block
- BIX (Building Industry Cross-connect): supports cat5e. GigaBIX for cat6

77. T568 Standards
- Color codes
  - ANSI
  - TIA
  - Dictates which color cable to which pin in RJ-45

78. Straight-through vs crossover cables
- Ethernet crossover cable is a type of Ethernet cable used to connect two devices of the same type together, such as two switches or two routers
- A rollover cable or console cable is used to connect to a device’s console port, such as with a router. 
- An Ethernet straight-through cable is a type of Ethernet cable used to connect hosts to a computer network, such as a PC and switch. 
- A Straight Tip (ST) connector is a type of fiber optic connector that has a bayonet-style plug. 
- Auto MDI-X: allows a switch port to dynamically determine which pins to use for transmitting and receiving

79. Ethernet standards
- Ethernet standards for copper cabling
  - 10BASE-T: Cat3 or higher, 10Mpbs, 100m
  - 100BASE-TXc: Cat5 or higher, 100Mbps, 100m
  - 1000BASE-T: Cat5 or higher, 1Gbps, 100m
  - 10GBASE-T: Cat6/6a or higher, 10Gbps, 55/100m
  - 40GBASE-T: Cat8, 40Gbps, 30m
- Ethernet standards for Fiber Optic cabling
  - 100BASE-FX, MMF (multi-mode fiber), 100Mbps, 2km
  - 100BASE-SX, MMF, 100Mbps, 300m
  - 1000BASE-SX, MMF, 1Gbps, 220m (62.5um core) or 550m (50um core)
  - 1000BASE-LX, MMF/SMF, 1Gbps, 550m (multi)/5km (single)
  - 10GBASE-SR, MMF, 10Gbps, 33m (62.5um core)/400m (50um core)
  - 10GBASE-LR, SMF(single-mode fiber), 10Gbps, 10km
- Coarse Wavelength Division Multiplexing (CWDM): typically supports a max 8 channels. Each channel wavelength is separated by 20mn. Max distance is 80km. Does not support amplifiers.
- Dense Wavelength Division Multiplexing (DWDM): max 80 channels. 0.4nm wavelength. Max distance 3000km. Supports amplifiers.
- Bidirectional Wavelength Division Multiplexing (WDM): a single fiber optic strand to carry the transmission and reception of multiple channels simultaneously. Can reduce fiber costs at the expense of fewer channels.

## Section 9: Module 7 - Using Ethernet Switches

80. Using Ethernet switches

81. MAC Addresses
- Address lengths
  - MAC address: 48 bits -> 12 hexadecimal
    - 24bits OUI (organization unit identification) code + 25 bits NIC-specific
  - IPv4 Address: 32bits
  - IPv6 address: 128 bits
- Data frame from laptop
  - Source MAC (PC) + Destination MAC (Gateway) + Source IP + Destination IP + Data
- Switch book-keeps MAC addresses of devices

82. Ethernet switch frame forwarding
- Flooding: occurs when an ethernet switch sends a copy of an incoming frame out all of its ports, other than port on which the frame was received, because the switch hasn't learned the port off of which the destination MAC address is connected
- Ethernet Frame Format
  - 18 bytes header of Dest MAC + Source MAC + TYPE + FCS
  - Preamble 7 + SFD 1 + Dest MAC 6 + Source MAC 6 + Type 2 + Data and Pad 46-1500 + FCS 4
    - MTU (Maximum Transmission Unit) 1500 bytes
    - Frame check sequence (FCS): error check
- Ethernet Jumbo Frame Format
  - Preamble 7 + SFD 1 + Dest MAC 6 + Source MAC 6 + Type 2 + Data and Pad 46-9000 + FCS 4
  
83. VLAN theory
- Virtual LAN: can separate different broadcast domains in a single switch
- Can allocate separate subnets in a single switch
  - Layer 2: those subnets cannot communicate each other. To communicat, the switch must Trunk into a router
  - Layer 3: Without Trunk into a router, subnets can communicate each other

84. Trunking Theory
- Trunk b/w switches, then each subnet can be expanded
  - IEEE 802.1Q Trunk: Adds four tab Bytes to each frame
  - In addition to 18 headers + 1500bytes, extra 4bytes

85. Voice VLANs
- IP phone over VLAN
- A VLAN that can be configured on an Ethernet switch for the purpose of carrying voice packets to and from IP phones
- Class of Service (CoS): A layer 2 quality of service (QoS) marking sent over a trunk, with a value in the range 0-7 and where voice frames are typically set to a CoS value of 5
- IEEE 802.1p: A layer 2 QoS Marking, similar to a CoS marking, that is sent over a non-trunking port. Four bytes are added to a layer 2 frame, with three bits in those four Bytes used for a priority marking, and with the twelve bits in the VLAN field set to all zeros
- Multi-VLAN acccess ports: separate Voice VLAN vs Data VLAN
- Trunk port
- Compatible with both CPD and LLDP-MED
- Frames are dot1Q trunk frames
- Unneeded VLANs should be pruned

86. Ethernet Port Flow Control
- When a switch is floorded with traffic 
  - Sends a PAUSE frame due to congestion
- The time is measured in quanta, which equals 512 bit times
- Bit time = 1/NIC speed
  - A Gigabit Ethernet NIC has 1 nanosecond bit time
- Part of IEEE802.3x standard (1997)
  - All CoS values were subject to the same amount of delay
- Priority-based Flow Control (PFC)
  - Part of IEEE802.1Qbb (2010)
  - Each CoS value is assigned to a different time to pause

87. Power Over Ethernet (PoE)
- Applications
  - When wireless access point doesn't have nearby power outlet
  - IP phone
  - Video surveillance camera
- PoE components
  - Power Source Equipment (PSE)
  - Powered Device
  - Ethernet cable
- PoE standards
  - Cisco Inline Power (7.7 Watts)
  - IEEE 802.3af (15.4 Watts)
  - IEEE 802.3at (30 Watts)
  - IEEE 802.3bt (100 Watts)  

88. Introducing Spanning Tree Protocol (STP)
- Loops are bad in Layer 2 network
  - Each router hop reduces a packet's time to live (TTL) field by 1 until it reaches 0. Then the packet is dropped
  - Layer 2 frames do not have a TTL field, and frames can circulate endlessly in a loop (broadcast storm)
- STP can prevent broadcast storm, which yields network outage

89. STP port states
- STP - the backstory
  - Ethernet bridge: a legacy networking device that made forwarding decision in SW, based on destination MAC addresses
  - Ethernet switch: a modern networking device that makes forwarding decisions in HW, based on destination MAC addresses
- The Time to Live (TTL) issue
- STP Port States
  - 4 Questions
    - Who is the root bridge?
      - The switch with the lowest Bridge ID
      - Bridge ID (BID): Priority 2 bytes + MAC address 6 bytes
    - What are the root ports?
      - The one (and only one) port on a non-root Bridge that is closest to the root bridge, in terms of cost
    - What are the designated ports?
      - The one (and only one) port on each segment that is closes to the root bridge in terms of cost
    - What are the blocking (non-designated) ports?
      - A port that is administratively enabled, but is not a root port nor a designated port

90. STP example
- How to build a STP topology

91. STP Convergence Times
```
         DP (SW1) DP
       / Root Bridge \
      /               \
    RP                RP
(SW1)                   (SW3)
    DP ---------------BLK
```
- Whne DP(SW1)---RP (SW3) is broken
  - Blocking (20sec)
  - Listening(15sec)
  - Learning (15sec)
  - Forwarding
    - Total 50 sec delay then BLK of SW3 becomes RP

92. STP Variants
- Common Spanning TRee (CST)
  - Used by IEEE 802.1D
  - The same STP topology is ued by all VLANs
- PVST+
  - Per-VLAN Spanning Tree (PVST)
  - Each VLAN run its own instance of STP
  - '+' indicates that the switches are interconnected via 802.1Q trunks
- MSTP
  - Multiple Spanning Trees Protocol
  - IEEE 802.1s
- RSTP
  - Rapid Spanning Tree Protocol
  - Typically converges b/w a few milliseconds and about 6 seconds
  - IEEE 802.1w

93. Link Aggregation
- Connects mutiple ports b/w switches
  - Allows higher bandwidth b/w switches
  - Provides load-balancing
  - Creates redundant links
- PAgP: Port Aggregation Protocol
- LACP: Link Aggregation Control Protocol

94. Port mirroring
- Capturing packets b/w client and server for investigation
- How to sniff the packet?
  - Wireshark
- Gets copies of the frame

95. Distributed switching
- Collpased core vs Three-tier designs
- Three-tier architecture
  - Core layer: connecting multiple buildings. Partial mesh topology
  - Distribution layer: SW-SW in access layer
  - Access layer: SW-PCs. Star topology
- Collapsed core: Two tier topology. For small building
  - Collapsed core layer
  - Access loayer
- Spine-leaf design for data centers
  - Spine switches
  - Leaf switches
  - Nodes
  - Spine Switches + Leaf switches corresponds to a single logical switch

## Section 10: Module 8: Demisytifying Wireless Networks

96. Demisytifying Wireless Networks

97. Introduction to Wireless LANs
- Ad hoc wireless LAN
  - Wireless client ~~~ wireless client
  - Bluetooth
- Infrastructure wireless LAN
  - Through access point
    - Acess Point is connected to a switch  
- Mesh wireless LAN
- Autonomous APs
  - Can provide seamless connection over multipe APs but they must have same configuration
- Lightweight APs
  - Doesn't need admin every AP
  - WLAN controller configures one time and the configuration can be distributed to other APs
  - LWAPP: Lightweight Access Point Protocol
  - CAPWAP: Control and Provisioning of wireless access points

98. WLAN Antennas
- Radiation Pattern
  - H plane: xy plane
  - E plane : including z axis
- Omnidirectional antennas
  - Designed to propagate singal in all direcitons
  - Lower gain with a less focused path
  - Better for broad coverage
- Directional antennas
  - Higher gain with a very focused path
  - Better for specifically directing coverage
  - Patch antenna
  - Yagi antenna
  - Dish antenna

99. Wireless Range Extenders
- Regenerates client singals into AP
- Ranges are extended

100. WLAN Frequencies and Channels
- Wireless frequency bands
  - 2.4 GHz band
    - 14 different channels
    - 5MHz b/w channels
    - Exception 12 MHz b/w 13 & 14
    - Channel 14 only allowed in Japan for 802.11b
    - Due to interference, actual signals are 1-3, 4-8, 9-13, each 22Mhz range
  - 5 GHz band

101. WLAN Standards

|standard | year | freq | max band width | transmission method |
|---------|----|------|----------------|------------|
|802.11   | 1997 | 2.4GHz | 1 or 2 Mbps | DSSS for FHSS|
|802.11a  | 1999 | 5GHz | 54 Mbps | OFDM |
|802.11b  | 1999 | 2.4GHz | 11 Mbps | DSSS |
|802.11g  | 2003 | 2.4GHz | 54 Mbps | OFDM |
|802.11n  | 2009 | 2.4 and 5GHz | 150 Mbps | OFDM |
|802.11ac | 2014 | 5GHz | 3.5 Gbps | OFDM |
|802.11ax | 2019 | 2.4 and 5GHz | 9.6 Mbps | OFDMA |

- DSSS (Direct-Sequence Spread Spectrum)
  - A single bit can be sent using a 2MHz frequency range
  - Using Barker 11 Coding, 1bit is transmitted along with 10 extra bits (called chips), which provide protection from interference
  - A symbol is the sequence of 11 bits being sent to encode a single bit
  - Used in the older IEEE 802.11b wireless standard
- FDM (Frequency Division Multiplexing)
- Orthogonal Frequency Division Multiplexing (OFDM): a data transmission technique that sends different signals using different subchannels, where adjacent subchannels are transmitted at right angles to one another
- Quadrature Amplitude Modulation (QAM)
  - 16-QAM: identifies 16 different targets in a constellation, each of which present 4bits
  
| standard | QAM | Bits represented | Supported channel widths|
|----------|-----|------------------|-----------------------|
| 821.11n  |   64-QAM |  6bits | 20/40 MHz|
| 821.11ac |  256-QAM |  8bits | 20/40/80/160 MHz|
| 821.11ax | 1024-QAM | 10bits | 20/40/80/160 MHz|

- Beamforming
  - Combination of constructive/destructive interference
- SU-MIMO (Single-User Multiple Input, Multiple Output)
- MU-MIMO (Multi-User Multiple Input, Multiple Output)
- Orthogonal Frequency Division Multiple Access (OFDMA)
  - IEEE 802.11ax (WiFi6)
  - Target Wake Time (TWT): schedules when a client can send and receive, resulting in less latency and power savings
  - BSS Coloring: allows signals for one SSID on a specific channel to be distinguished from singals for a different SSID using the same channel by **coloring** that traffic

102. Regulartory Impacts of Wireless Channels
- Standard Bodies
  - FCC (North America)
  - MKK (Japan)
  - ETSI (Europe)

## Section 11: Module 9 - Addressing Networks with IPv4

103. Addressing Networks with IPv4

104. Binary numbering
- 10.1.2.3 => 00001010.00000001.00000010.00000011

105. Binary practice exercise #1

106. Binary practice exercise #2

107. IPv4 Address Format
- Classless Inter-Domain Routing (CIDR) Notation: Identifies the number of bits in an IPv4 address's network address, using a forward slash followed by the number of binary 1s in the subnet mask
  - Ex) 10.1.2.3/8

| Dotted Decimal Notation | 10 | 1 | 2 | 3 |
|-------------------------|----|---|---|---|
| IP address (in binary)  | 00001010 | 00000001 | 00000010 | 00000011 |
|Subnet Mask | 11111111 | 00000000 | 00000000 | 00000000 |
|| Network Bits | | Host Bits ||

- 10.1.2.3: IP address with no subnet information
- 10.1.2.3/8: IP address with Prefix notation
- 10.1.2.3 255.0.0.0 : IP address with Dotted Decimal Notation

- IPv4 Address Format

| Address class | Value in first octet | classful mask (dotted decimal) | classful mask (prefix notation) |
|---|--|--|--|
|A | 1-126   | 255.0.0.0     | /8  |
|B | 128-191 | 255.255.0.0   | /16 |
|C | 192-223 | 255.255.255.0 | /24 |
|D | 224-239 | N/A           | N/A |
|E | 240-255 | N/A           | N/A |

- Note that first octet 127 is excluded
  - Loopback IPv4 Address: 127.0.0.1
  - A device can attempt to connect to itsefl by attempting to connect to 127.0.0.1

108. Public vs. Private IPv4 Addresses
- Private IPv4 Addresses
  - Will have overlapping numbers among different private groups as numbers are not infinite

| Address Class |  Address Range | Default subnet Mask |
|--|--|--|
|A | 10.0.0.0 - 10.255.255.255 | 255.0.0.0|
|B | 172.16.0.0 - 172.31.255.255 | 255.255.0.0 |
|B | 169.254.0.0 - 169.254.255.255 | 255.255.0.0 |
|C | 192.168.0.0 - 192.168.255.255 | 255.255.255.0 |

- Automatic Private IP Addressing (APIPA): automatically configures an IPv4 address (beginning 169.254) for clients that neither had an IPv4 address statically  assigned nor dynamically obtained an IPv4 address

109. IPv4 Unicast, Broadcast, and Multicast
- When a server sends video data into SW then to many PCs
  - IPv4 Unicast
    - One-to-one
    - Not scalable
  - IPv4 Broadcast
    - One-to-all
    - There might be some PCs not need video dat
  - IPv4 Multicast
    - One-to-many
    - Identifies PCs need video

110. The need for subnetting

| Address class | Assignable IP addresses |
|--|--|
|A| 16,777,214|
|B| 65,534|
|C| 254 |

- Avoid wasting addresses
  - For network 192.0.2.0/24, if there are two servers only, using .1 & .2, then 252 addresses are wasted
  - For groups of private address ranges, we can find the overlapping address and bits, and can narrow down the range of subnetting
- An IPv4 address contains 2 components: an IP address and a subnet mask. A subnet mask is used with an IP address to differentiate between the network and host portions of an address. A subnet mask is also used to define what addresses can be used within a given range. 

111. Calculating Available subnets
- Class A network: has a number in the range 1-126 in the first octet and a default subnet mask of /8
- Class B network: 128-191 in the first octet and a default subnet mask of /16
- Class C network: 192-223 and /24
- Ex)
  - A subnet mask of 255.255.255.224 is applied to a Class C network of 192.168.1.0/24
  - Network class? C
  - Natural Mask (netmask) ? /24
  - Subnet mask? 255.255.255.224
    - 224 => 11100000
    - 255.255.255.224 = 11111111.11111111.11111111.11100000
      - **Count the number of 1** => 27 => /27
  - Borrowed bits? => 3
  - Number of subnets? 2^3 = 8

112. Calculating Available hosts
- 2^h - 2
  - h: number of host bits
  - why subtract 2?
    - Where all bits are 0
    - Where all bits are 1
- Ex)
  - A subnet mask of 255.255.255.224 is applied to a Class C network 192.168.1.0/24
  - How many hosts can be assigned in each subnet?
    - Number of 1s in subnet mask? => 27
    - How many host bits? 32-27 = 5
    - Number of hosts? 2^5 - 2 = 30

113. Subnetting Practice Exercise #1
- Your company is assigned the 172.20.0.0/16 network. Use a subnet mask that will accommodate 47 subnets while simultaneously accommodating the max. number of hosts per subnet. What subnet mask will you use?
  - This is Class B
  - Implying 47 different departments in a building
  - 2^6 = 64 > 47 > 2^5 = 32. Therefore we need 6 borrowed bits
  - This is Class B and netmask is /16
  - 16 + 6 = 22 bit subnet mask: 11111111.11111111.11111100.00000000
    - /22 or 255.255.252.0
    - 8+2=10. 2^10 - 2 = 1022?

114. Subnetting Practice Exercise #2
-  Your company is assigned the 172.20.0.0/16 network. Use a subnet mask that will accommodate 100 hosts per subnet while maximizing the number of available subnets
  - This is Class B
  - To have 100 hosts, 7 host bits (2^7-2 = 126 > 100)
  - 32-7 = 25. /16 is Class B and we need 9 borrowed bits
  - 11111111.11111111.11111111.10000000 or /25 or 255.255.255.128

115. Calculating Usable Ranges of IPv4 Addresses
- Find the first usable IP address by adding a binary 1 to the network address
- Find the last usable IP addresss by subtracting a binary 1 from the Directed Broadcast Address

116. Subnetting Practice Exercise #3
- You want to apply a 26-bit subnet mask to 192.168.0.0/24 network address space. What are the subnets and what are the usable address ranges in each subnet?
  - 26-24 = 2 borrowed bits
  - 11111111.11111111.11111111.11000000 or /26 or 255.255.255.192
  - Interesting octet is 4th octet
  - Block size = 256 - 192 = 64  
  - To determine the first subnet, set borrowed bits + host bits as 0
    - 192.168.0.0/26
  - Determine additional subnets by adding the block size in the interesting octet
    - 192.168.0.0
    - 192.168.0.64
    - 192.168.0.128
    - 192.168.0.192

| subnet | directed broadcast | Usable IP ranges |
| -- | -- | --|
|192.168.0.0   | 192.168.0.63  | 192.168.0.1 - 192.168.0.62    |
|192.168.0.64  | 192.168.0.127 | 192.168.0.65 - 192.168.0.126  |
|192.168.0.128 | 192.168.0.191 | 192.168.0.129 - 192.168.0.190 |
|192.168.0.192 | 192.168.0.255 | 192.168.0.193 - 192.168.0.254 |

## Section 12: Module 10 - Addressing Networks with IPv6

117. Addressing Networks with IPv6
- Typically written in hexadecimal number

118. Hexadecimal Numbering
- 198 = 16*12+6 = 0xC6
- 0x2F = 16*2 + 15 = 00101111 in nibbles
- 0xBC = 16*11 + 12 = 10111100

119. IPv6 Address Format
- Prefix + Host
- 32 hexadecimal numbers
- 8 "quartets" of 4 hexadecimal digits separated by a colon
- One hexadecimal digit represents 4 binary bits
- 128 bits total length
- No broadcasts
  - But all node multicast
- No fragmentation (MTU discovery performed for each session)

120. Shortening an IPv6 Address
- Omit leading zeros in a quartet
  - Represents consecutive quartets containing all zeros with a double colon (only once per address)
  - 23A0:201A:00B2:0000:0000:0000:0400:0001/64 => 23A0:201A:B2::400:1/64

121. IPv6 Address shortening exercise
- 2000:0000:0000:0000:1234:0000:0000:000B/64 => 2000::1234:0:0:B/64

122. IPv6 Global Unicast
- 001 (3bits) + Global Routing Prefix (45bits) + subnet ID (16bits) + Interface ID (64bits)
- Addressing starts with 2000::/3

123. IPv6 Multicast
- Joining multicast group
- 11111111(8bits) + Flags (4bits) + Scope (4bits) + Group ID (112bits)
- Flags 
  - 4 bits: ORPT
  - 0: reserved and set to 0
  - R: if set to a 1, P and T must also be set to 1. This would indicate that a Rendezvous Point (RP) address was embedded in the address
- Scope Examples
  - 1: Interface-Local scope
  - 2: Link-Local scope
  - 4: Admin-Local scope
  - 5: Site-Local scope
  - 8: Organization-Local Scope
  - E: Global Scope  
- FF02::1 All nodes in the link-local scope
- FF02::2 All routers in the link-local scope

124. IPv6 Link Local
- FE80::/10

125. IPv6 Unique Local
- Starts with FC00::/7
- Not routable in the public internet
- Similar to IPv4 private addresses
- L bit set as 1

126. IPv6 Loopback
- ::1
  - localhost
  - 127 Zeros
  - 127.0.0.1 of IPv4
  - Can be used to verify the IPv6 stack is operating on a device

127. IPv6 Unspecified
  - `::`
  - All zeros over 128 bits
  - Used for a client's source address when sending an neighbor solicitation message
  - Used for a client's source address when sending a Router solicitation message

128. IPv6 Solicted-Node Multicast
- Begins with FF02::1:FF/104
- Address ends with the last 24 bits of the corresponding IPv6 address
- Used instead of an IPv4 Address Resolution Protocol (ARP) broadcast
- Also used for Duplicate Address Detection (DAD)

129. EUI-64 Address
- 64bit Extended Unique Identifier (EUI-64)
- Uses the MAC address of an interface to create 64bit interface ID
  - But a MAC address is only 48 bits long
  - Split the 48 bit MAC address in the middle
  - Insert FF.FE in the middle
  - Replace dot to colon delimiter
  - Convert the first 2 hex digits to binary
  - Flip the 7th bit

130. IPv6 Autoconfiguration

131. IPv6 Traffic Flows
- Unicast: one to one
- Multicast : one to many
- Anycast: one to nearest

132. Dual Stack
- When IPv4 and IPv6 coexist
- The PC must have IPv4 and IPv6 address (dual stack)

133. Tunneling IPv6 through an IPv4 network
- Sending IPv6 packet encapsulated in IPv4 packet
- Only needed during a network's migration to IPv6

134. IP Address Mangement (IPAM)

## Section 13: Module 11 - Explaining IP Routing

135. Explaining IP Routing
- Following packets

136.  Packet flow in a routed network

137. Static and default routes

138. Routing protocols

139. RIP

140. OSPF

141. EIGRP

142. BGP

143 Subinterfaces
