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
- `2000:0000:0000:0000:1234:0000:0000:000B/64` => `2000::1234:0:0:B/64`

122. IPv6 Global Unicast
- 001 (3bits) + Global Routing Prefix (45bits) + subnet ID (16bits) + Interface ID (64bits)
- Addressing starts with `2000::/3`

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
- Connects networks, route between them (InterVLAN routing), and possibly provide internet connectivity. This device is usually a multilayer switch or a router. Routing within the same VLAN and Intranet communication doesn’t require a default gateway. 

136.  Packet flow in a routed network
- How it routes packets in the network?
- ARP (Address Resolution Protocol)
  - A Data Link protocol used to dynamically map an IP address to a MAC addres
- Flow:
  - In addition to source/destination IP, MAC address of the corresponding server is necessary
  - PC -> R1 -> R2 -> server
  - Packets leave the PC and arrives at R1
  - R1 does ARP and finds R2
  - R2 does ARP and finds the server
- Static vs dynamic routing protocol

137. Static and default routes
- Static routes
  - May not scale well
  - Administratively added routes
  - Very Believable (AD=1)
  - Specfies a next hop to reach a network
- Default routes
  - Doesn't have large routing table
  - Can be static or dynamic
  - Used if a router doesn't have a more specific route entry

138. Routing protocols
- RIP: a distance Vector routing protocol. not very scalable
- OSPF: a link state routing protocol
- EIGRP: an advanced Distance Vector routing protocol
- BGP: a path vector routing protocol. 
- Consideration
  - Scalability
  - Vendor interoperability
  - Familarity
  - Convergence
- Administrative distance (AD)
  - Directly connected : 0
  - Static : 1 (by default)
  - EIGRP : 90
  - OSPF : 110
  - RIP : 120
- IGP (Interior Gateway Protocol)
  - RIP
  - OSPF
  - EIGRP
- Autonomous System (AS):network under a single administrative control
- EGP (Exterior Gateway Protocol)
  - BGP
- Routine protocol comparision

| Routing Protocol | Distance-vector | Linke-state | Path-Vector|
|--|--|--|--|
| RIP   | v |   |    |
| OSPF  |   | v |    |
| EIGRP | v |   |    |
| BGP   |   |   |  v |

- Hop count: The metric used by RIP that measures the number of routers that must be crossed to reach a destination network
- Link state database: maintained by  OSPF that contains information about network topology
- Dijkstra Algorithm: assigns costs to links and calculates the shortest path b/w any two points in a network

139. RIP (Routing Information Protocol)
- RIPv1
  - Not on modern network anymore. Early 90s
  - Broadcasts routing info to every devices like printers, PCs, ...
    - Every 30 sec
  - No VLSM Support
  - IPv4 only
- VLSM (Variable Lenght Subnet Mask) support: the ability of a routing protocol to advertise a network with a non default subnet mask
- RIPv2
  - Multicasts (224.0.0.9)
  - VLSM support
  - Authentication
  - IPv4
- RIPng
  - Multicasts (FF02::9)
  - IPv6
  - Uses hop count: 19 as infinite
  - Full & triggered updates
  - Split horizon: prevents a routing protocol from an advisement out the inteface on which it was received
    - Prevents hop count contamination by broken devices
  - Poison Reverse: sends an advisement with an infinite metric for an unreachable route
    - When AD becomes 16 due to contamination then removes it from the table

140. OSPF
- Most of corporate network
- One of Link state routing protocols
  - Every router has a map of the network
- Open shortest path first (OSPF) characteristics
- Open standard
- Establishes adjacencies with other routers
- Sends Link State Advertisement (LSAs) to other routers in an area
- Constructs a Link State Database from received LSAs
- Runs the Dijkstra Shortest Path First (SPF) Algorithm to determine the shortest path to a network
- Attempts to inject the best path for each network into a router's IP routing tabe
- OSPF terminology
  - Hello: A protocol used to discover OSPF neighbors and confirm reachability
  - Link State Advertisement (LSA): information a router sends and receives about network reachability (not a packet)
  - Link State Update (LSU): a packet that carries LSAs
  - Link State Request (LSR): Used by a router to request specific LSA
  - Link State Acknowledgement (LSAck) : Used by a router to confirm it received an LSU
- Neighborship vs Adjacencies
  - Neighbors are routers that:
    - Reside on the same network link
    - Exchange hello messages
  - Adjacencies are routers that:
    - Are neighbors
    - Have exchanged Link State Updates (LSUs) and Database Description (DD) packets
    - No. of adjacencies = n*(n-1)/2 where n is the number of routers
      - Not scale well
    - Need for the designated routers (DR)
      - Backup Designated router (BDR) as well
      - Adjacencies only need to be formed with DR and BDR
- Can divide OSPF ares into multiple OSPF
  - Applies Dijkstra algorithm to each area
  - Area Border Routers (ABRs)
  - Multi-area OSPF networks must have a backbone area numbered 0 or 0.0.0.0
- OSPF cost
  - cost = Reference BW / Interface BW
  - The default reference BW is 100,000,000 bits per second (100Mbps)
```
               R2
             /    \
   100 Mbps /      \ 100 Mbps
           /        \
         R1 --------- R3
         |  10 Mpbs  |
100 Mbps |           | 100 Mpbs
         |           |
        SW1         SW2 
         |           |
        PC1         PC2
```
- Cost for R1-R2-R3 = 1 + 1 + 1 
- Cost for R1-R3 = 10 + 1 = 11

141. EIGRP
- Enhanced Interior Gateway Routing Protocol
- Fast convergence
- Scalable
  - > 500 routers
- Load balancing over unequal cost links
- Classless (VLSM support)
- Communicates via multicast (224.0.0.10 or FF02::A)
- Was Cisco-proprietary but now open
- EIGRP Metric calculation
  - Bandwidth
  - Delay
  - Reliability
  - Load
  - MTU
- EIGRP path selection
  - Feasible Distance (FD): EIGRP's metric to a network
  - Successor route: EIGRP's preferred route to a network
  - Feasible Successor Route: EIGRP's backup route to a network
- EIGRP Feasibility condition: An EIGRP route is a feasible successor route if the Reported Distance (RD) from our neighbor is less that the Feasible Distance (FD) of the successor route
- Feasibility Condition
```
              R2
         /          \
 10,000 /            \ 5,000
       /              \ 
      /                \
   R1 ------ R3 ------- R5 ------10.1.1.0/24
      \ 7,000   10,000 /   1000
       \              /
 4,000  \            / 17,000
         \          /
             R4
```
- B/W R1 and 10.1.1.0/24:

| Neighbor |RD      |FD     | (Feasible) Successor?|
|----------|--------|-------|----------------------|
| R2       | 6,000  | 16,000| Successor |
| R3       |11,000  | 86,000| Feasible Successor |
| R4       |18,000  | 22,000|  X |

- If R2 and R3 crash, R1 will send a query through the network and R4 will be connected - but not immediatly

142. BGP
- Border Gateway Protocol
- Internet b/w organization
- Autonomous System (AS)
- Exterior Gateway Protocol (EGP)
- Forms neighborships
- Neighbor's IP address is explicitly configured
- A TCP session is established b/w neighbors
- Advertises Address Prefix and Length (called Network Layer Reachability information(NLRI))
- Advertises a collection of Path attributes used for path selection
- Path vector routing protocol
- BGP path attributes
  - Weight
  - Local Prefernce
  - Originate
  - AS Path Length
  - Origin type
  - Multi-exit Discriminator (MED)
  - Paths
  - Router ID

143. Subinterfaces

## Section 14: Module 12 : Streaming Voice and Video with United Communications

144. Streaming Voice and Video wtih united communications

145. Voice over IP
- PBX(Private Branch Exchange)
  - A privately owned telephone switching system
- Tie Line (aka Tie Trunk)
  - Interconnects privately owned telephone switching systems
- PSTN (Public Switched Telephone Network): the worldwide telephone system, made up of an interconnection of multiple telephone companies
- RTP (Realtime Transport Protocol)
- Digitizing voice
  - Nyquist theorem: 2x of the highest frequency
- Pulse Amplitude Modulation (PAM)
- Pulse Code Modulation (PCM)
- Quantization Noise - background noise in PCM

| Codec | Bandwidth (Payload only) | Bandwidth over Ethernet |
|--|--|--|
|G.711 | 64 kbps | 87.2 kbps |
|G.729 | 8 kbps | 31.2 kbps |
|iLBC | 13.3 or 15.2 kpbs | 28.8 or 38.4 kpbs |

146. Video over IP
- Terms to know
  - Frames per second (fps)
  - Refresh rate
  - Interlaced video - a half of lines shown next frame
  - Progressive vide
  - Pixel
  - Aspect ratio
  - Compression standards:
    - MPEG-1
    - MPEG-2
    - MPEG-3
    - MPEG-4
    - H.264 (MPEG-4 AVC)

147. Unified communication networks

148. Quality of Service (QoS)
-  A feature set used to engineer or prioritize various traffic types based on classifications marked in traffic
- When there is congestion in network
- Buffer (aka Queue)
  - Memory allocated by a router interface to temporarily store packets the interface is unable to send at the moment
- Do you need QoS?
  - Periodic congestion
    - If you have congestion 24/7, you need better bandwidth
- 3 Categories of QoS
  - Best effort: Not Strict
    - FIFO (First in First Out)
  - DiffServ (Differentiated Services): Less Strict
    - A collection of QoS mechanicsm that classify traffic types and assign policies to those traffic classes
  - IntServ (Integrated Services): Strict
    - A QoS mechanism that allows an application to reserve bandwidth for that application by using RSVP
- Common QoS Mechanisms
  - Classification and Marking
  - Queuing
  - Congestion Avoidance
    - TCP slow start: Occurs when TCP reduces its window size because of dropped or delayed traffic segments
    - TCP synchronization (aka Global synchronization): occurs when all TCP flows simultaneously go into TCP slow start due to a queue overflow
    - Weighted Random Early Detection (WRED): Introduces the possibility of discard for packets with specific markings at specific queue depths
    - Random Early Detection (RED): Introduces the possibility of packet discard, regardless of markings, at specific queue depth
  - Policing and Shaping
    - Traffic conditioners: a category of QoS mechanisms that limits bandwidth for a class of traffic
  - Link Efficiency
    - Link Fragmentation and Interleaving (LFI): a QoS mechanism that fragments large packets and interleaves smaller packets with the fragments

149. QoS Markings
- Class of Service (CoS)
  - IEEE 802.1Q Frame: Tag Control Information Bytes
    - PRI + CFI + VLAN ID
      - PRI is CoS Bits
- Type of Service (ToS) Byte
  - Traffic Class Byte in IPv6
  - IP Precedence : 1+2+3
  - DSCP: 1+2+3+4+5+6
- Differentiated Services Code Point (DSCP) Values
- Random Early Detection (RED)
  - Output Queue
    - Introduces Max/Min Threshold

150. QoS Traffic Shaping and Policing
- Traffic Conditioners
  - Shaping: Delays excess traffic rather than dropping it
    - Used on slower speed interfaces
  - Policing: Drops traffic rather than delaying it
    - Used on higher speed interfaces
- Shaping example
  - CIR (Committed Information Rate ): Average speed over the period of second
  - Bc (Commited Burst): Number of bits (for shaping) or bytes (for policing) that are deposited in the token bucket during a timing interval
  - Tc (Timing Interval) = The interval at which tokens are deposited in the token bucket
  - CIR = Bc/Tc (64,000 bps = 8000 bits / .125 sec)

## Section 15: Module 13 : Virtualizing Network Devices

151. Virtualizing Network Devices

152. Virtualized Devices
- Virtualized servers: shares a single NIC
- Hypervisor: SW that can create, start, stop and monitor multiple virtual machines
  - Type-1 (native or bare metal): Runs directly on the server's hardware, not on OS
  - Type-2 (hosted): Runs in an traditional OS
- Virtualization
  - Virtual NIC: SW associated with a unique MAC address, which can be used by a VM to send/receive packets
  - Virtual Switch: SW that can connect to other virtual switches, virtual NICs and to a physical NIC
- Virtual services
  - Virtual firewall
  - Virtual router
  - Virtual SLB (Server Load Balancer)

153. Virtual IP
- Virtual Router Redundancy Protocol (VRRP)
  - When a gateway goes down - backup gateway?
  - Use virtual Router associated with MAC address
    - Can coordinate BACKUP gateway wtih advertisement interval of 1 sec
  - Standard
  - RFC3768
  - Master and Backup Routers
  - MAC Address: 0000.5e00.01XX
  - Preempt Enabled by default
  - Default master advertisement interval: 1sec
  - Default Master Down Interval: 3\* Master_advertisement_interval + (256- VRR priority)/256
  - Multicast Address: 224.0.0.18
  - Can use interface IP address as virtual IP address
- First-Hop Redundancy Protocol (FHRP): A protocol that allows a backup router to take over if a client's default router goes down

154. Storange Area Network (SAN) Technology
- Storage Area network (SAN): A network containing storage devices that can make those storage devices appear to be locally attached to servers, seen in places such as Data Centers. 
- Direct-Attached Storage (DAS): Storage that is physically attached to a computer
- Block level storage: an approach to storing and retrieving data based on a block of bytes and bits as opposed to storing and retrieving an entire file. Mail server, RDBMS, ...
- File-level storage: an approach to storing and retrieving data based on the transfer of entire file, as opposed to transferring blocks of bytes or bits. Regular files on disk/LAN
- NAS (network-attached storage): a network appliance that acts as a file server (file-level storage) and can be accessed over an ethernet network
- Fibre channel (FC): a technology that allows high-speed block-level access to storage devices over an FC network (not over an ethernet network)
  - Needs Host bus adapter, not NIC
- Fiber channel over ethernet (FCoE)
  - Fibre channel on ethernet
  - Expensive solutions
- iSCSI (Internet Small Computer System Interface): a technology that allows SCSI commands to be sent inside of IP packets, thus allowing block-level access to a remote storage device over an IP network

155. Using Infiniband for SANS
- Competing over ethernet or FC
- FCoIB (Fibre Channel over IB)

156. Cloud Technologies
- XaaS (Anything as a Service)
- Types of cloud servcies
  - Public
  - Private
  - Hybrid
  - Community
- Typical Cloud Services
  - IaaS: Infrastructure
  - SaaS: SW
  - PaaS: Platform. Development environment.
  - Naas: Network. VPN
  - Daas: Desktop
- Elastic Provisioning: Adding/removing resources on an as-needed basis

157. Accessing Cloud Services
- How enterprise connects to Cloud Provider?
  - Internet? Not safe
    - VPN
  - Private WAN
    - MPLS
    - Metro ethernet
  - Intercloud exchange
    - Selects among different cloud providers
    - Remap services as necessary

158. Infrastructure as Code (IaC)
- Uses configuration files (or code) to
  - Provision infrastructure devices
    - Spinning up new servers
    - Defining virtualized switches, routers, and firewalls
  - Configure an existing infrastructure
    - Defining network parameters on an infrastructure device
    - Configuring routing protocols on a virtualized router
  - Deploy and manage applications
    - Install applications on servers
    - Configure application parameters
    - Apply updates and patches to applications
- Sample tools
  - Terraform
  - puppet
    - agents on device
  - CHEF
  - ANSIBLE
    - Playbook: configuration instructions
      - YAML format
    - Inventory: contains a list of devices
    - No agent required on device

159. Multi-tenancy

## Section 16: Module 14 : Securing a Network

160. Securing a network

161. General Security and availability
- Security Goals
  - Confidentiality
    - Through encryption
  - Integrity
    - Avoid corruption/data modification
  - Availability
- Confidentiality
  - Firewalls: stateful inspection
  - Access Control Lists (ACLs)
    - Ex) SSH, no telnet
  - Encryption
    - Symmetric: source/destination have symmetric (shared) key
      - Data Encryption Standard (DES): older encryption algorithm with 56-bit key
      - 3DES: Uses three 56-bit DES (total 168)
      - AES (Advanced Encryption Standard): 128/192/256 bit keys
    - Asymmetric encription: uses a pair of keys, a public key and a private key
      - RSA (Rivest, Shamir, Adleman)
      - Certifciate Authority (CA): a trusted third-party that can sign (ie, encrypt using the CA's private key) a device's digital certificate, allowing the recipient of the digital certificate to validate it using the CA's public key
      - Ex) amazon.com has x.509.v3 digital certificate with public and private keys
        1. A client wants a secure connection
        2. Server sends digital certificate with public key
        3. Client authenticates the certificate with the CA's public key (built in Web-browser)
        4. Client generates a string and encrypts it with amazon.com's public key
        5. Client sends encrypted string to amazon.com
        6. amaonz.com decrypts the string with its private key
- Integrity
  - Hashing algorithms
    - MD5 (Message Digest 5)
    - SHA-1 (Secure Hash Algorithm 1): produces 160bit hash digest
    - HMAC (Hash-based Message authentication code): uses a shared secret key, in conjunction with a hashing algorithm, to create a hash digest. Prevents hash modification
- Availability
  - The five nies
    - 99.999 percent uptime
    - 5min of downtime per year
  - Sample threats
    - Improperly Formatted Data
    - DoS
    - DDos
  - Prevention
    - OS patches
    - IDS, IPS, and Firewall appliances
      - Intrusion Detection System (IDS) sensor: checks incoming traffic with a database of well-known attacks
      - Intrusion Prevention System (IPS) sensor: checks the signature of traffic with the database and stops it if detected
        
162. Vulnerabilities and Exploits
- Vulnerability: a flaw in a secured system
- Exploit: SW that can take advantage of a vulnerability
- Common Vulnerabilities and Exploits (CVE)
  - MITRE corporation operates the National Cybersecurity FFRDC (Federally Funded Research and Development Center)
  - Their CVE program identifies, defines, and publishes cybersecurity vulnerabilities
- Zero-day attack: an exploit launched against a newly discovered vulnerability, before the deveoper can patch the vulnerability

163. Denial of Service Attacks
- DoS: overwhelms the target with traffic
- SYN Flood: initiates multiple TCP 3-way handshake but never completes them
- UDP Flood: sends a large volume of UDP segments, and the victim cannot verify the sender's IP address
- HTTP Flood: Sends a continuous stream of HTTP instruction (GET or POST)
- DNS Reflected: Spoofs their IP address to be the IP address of the victim and sends a large number of DNS queries to multiple publicly available DNS servers

164. On-path Attacks
- Aka Man-in-the-middle (MiTM) Attack 
- A malicious user injects inside a communication flow b/w two systems, enabling them to intercept or manipulate that flow's traffic
  - MAC Flooding: attacker floods switch with so many fake MAC addresses, the switch's MAC address table fills to capacity
  - ARP Poisoning: Unsolicated ARP replies are sent to the victim, claiming the attacker's MAC address is the MAC address of the victim's default gateway
    - Now the attacker can intercept the traffic
    - Dynamic ARP Inspection (DAI) is used to inspect an ARP to make sure it is legitimate. VLANs, Port Security, and ACLs can’t prevent ARP poisoning
  - Rogue DHCP: The attacker's DHCP server tells the victim that the IP address of the default gateway is the attacker's IP address
    - DHCP snooping: Assigns specific ports for DHCP server 

165. VLAN Hopping Attacks
- Switch spoofing: attacker pretends to be an Ethernet switch and sets up a trunk carrying traffic for all VLANs
- Double tagging: attacker adds two VLAN tags to a frame, the outer tag is a trunk's native VLAN, and the inner tag is the target VLAN

166. Social Engineering Attacks
- Phishing
- Tailgaiting (Piggybacking)
- Shoulder surfing

167. Other Common Attacks
- Insider threat: a malicious user that is a part of an organization
- Logic bomb: a malicious piece of code that can perform some destructuve action based on a time or an event
- Rogue access point: a wireless AP installed on a network without proper authorization
- Evil twin: a rogue AP appearing to be a legitimate wireless AP
- War driving: Driving around a geographical area in an attempt to find WiFi hotspots
- Malware
- DNS poisoning
- Ransomware
- Spoofing: when malicious users falsify their MAC or IP addresses, in an attempt to conceal their identity
- Deauthentication: an attack where a malicious user sends a deauthentication frame along a spoofed IP address to wireless AP, which causes a legitimate user to be dropped from the wireless network, making the client to reconnect to rogue access point
- Brue force

168. Common Defense Strategies
- Best practice
  - Change default credentials
  - Avoid Common passwords
  - Upgrade firmware
  - Patch and update
  - Perform file hashing
  - Disable unnecessary services
  - Use secure protocols
  - Generate new keys
  - Disable unused ports
    - IP ports
    - Device ports
- Mitigating network threats
  - Signature management: keep attack signatures current on devices, such as IDS and IPS sensors
  - Device hardening: apply a collection of best practice procedures to secure network devices
  - Change the native VLAN: configure a trunk's untagged VLAN to a non-default value
  - Define privileged user accounts : instad of a single admin account
  - File Integrity Monitoring
  - Role separation: assign different sets of permissions to different categories of users
  - Honeypot deployment: configure a host that does not contain sensitive information
    - A honeynet is a computer network that intentionally has vulnerabilities embedded into it for the sole reason of analyzing how an attacker would attempt to breach a network. Honeynets contain honeypots (A single system intentionally made unsecure to bait attackers)
  - Penetration test (aka pen testing)
  - Network segmentation: subdivide a network into different segments using VLANs and DMZs
  - Defense in Depth: multiple layers of security, as opposing to have a single security solution
  - Zero trust: no user is given a default set of permissions
  - Least privilege: users are given the minimum set of privileges that still allows them to perform their work

169. Switch Port Defense
- MAC flooding attack
  - Attackers send series of MAC addresses, yielding MAC table overflow
  - Port security: sets allow max addresses to have  

170. Access Control Lists
- ACLs
  - Can permit or deny traffic
  - Can be applied inbound or outbound
  - Processed top-down
  - Implicit deny any statement at the bottom
  - Can be standard or extended
  - More specific ACEs placed near the top
  - Standard ACLs placed near destination
  - Extended ACLs placed near source

171. Wireless Security Options
- Authentication: username/passwd
- Encryption
- Wired Equivalent Privacy (WEP)
  - The security standard specified by IEEE802.11
  - RC4 encryption, which is trivial to crack
- Primary modes of key distribution
  - Two modes of key distribution
  - Pre-shared key (PSK) model (aka personal mode): matching keys are preconfigured on wireless clients and AP
    - Pre-shared key might be stolen
  - Enterprise mode: credentials from an authentication server (eg RADIUS server), then session key is provided during a permitted session
- Enhanced Encryption Protocols
  - Temporal Key Integrity Protocol (TKIP)
    - Improved encryption, compared to WEP
  - Advanced Encryption Standard (AES)
    - Significantly stronger than TKIP
- Enhanced Security Protocols
  - WiFi Protected Access (WPA)
    - Used TKIP for enhanced encryption
    - Upgraded security in SW without requiring new HW
    - Used a longer initialization vector, 24 bits -> 48 bits
  - WiFi Protected Access II (WPA2)
    - A requirement for WiFi certification in 2006
    - Required support for AES
    - Required more processing power than WPA
    - Susceptible to KRACK vulnerability (found in 2016)
    - CCMP (Counter Mode Cipher Block Chaining Message Authentication Code Protocol) is an encryption protocol used with WPA2. CCMP replaced the functionality of TKIP (Temporal Key Integrity Protocol) in WPA2.
  - WiFi Protected Access III (WPA3)
    - Uses 192 bit AES encryption (for Enterprise mode)
    - Uses Protected Management Frames (PMFs) to prevent other devices from spoofing management frames
    - Ues Simultaneous Authentication of Equals (SAE) to require interaction with the network before generating a key, to prevent dictionary attacks
    - Prevents eavesdropping on public network (or networks with pre-shared keys)
    - Replaces WiFi Protected Setup (WPS) with Device Provisioning Protocol (DPP)
- Isolating Wireless Access
  - Guest Network Isolation: isolates wireless clients from an organization's internal network, while allowing the guest clients to access one another and the internet
  - Wireless Client Isolation: Isolates wireless clients from any other local network devices, with exceptions such as a DHCP server and default gateway
- MAC filtering
  - Only allows a device on a network if its MAC address is allowed
- Geofencing: can use a mobile device's GP location to permit or deny netowork access, or to grant or revoke network permissions
  - In shopping malls
- Captive portal: redirects users to connect to a network page where the user might be prompted to provide information or agree to terms of use

172. Extensible Authentication Protocols (EAPs)
- PC -> AP (authenticator) -> Authentication server (RADIUS)
  - PC/AP will get a session key
- Extensible Authentication Protocol - transport layer security (EAP-TLS)
  - One of the original authentication method by IEEE 802.1X
  - Authenticates end users and RADIUS servers using a Message Authentication Code derived from the digital certificates of the end users and RAIDUS servers
  - Requires a Certificate Authority (CA)
  - Allows a client to login using their credentials stored in a MS Active Directory database
- EAP-FAST (Flexible Authentication via Secure Tunneling)
  - A client uses a Protected Access Crendential (PAC) to request access to the network
  - Consists of two or three phases:
    - Phase 0 (optional): A client's PAC is dynamically configured
    - Phase 1: The client and AAA server use the PAC to establish a TLS tunnel
    - Phase 2: The client sends user information across the tunnel
- Protected Extensible Authentication Protocol (PEAP)
  - PEAP version 0 (EAP-MSCHAPv2): use MS Active Directory to store user credentials
  - PEAPv1/EAP-GTC (Generic Token Card): Uses generic databases (LDAP and OTP) for authentication

173. Authentication Servers
- AAA
  - Authentication: who are you?
  - Authorization: what are you allowed to do?
  - Accounting: what did you do?

| TACACS+ | RAIDUS|
|---------|-------|
| Cisco-Proprietary| Industry-standard|
| TCP | UDP|
| Separates AAA functions | Combines AAA functions|
|Two-way challenge response | One-way challenge Response|
| Encrypts entire packet | Only encrypts password|

- Kerberos
  - Key Distribution Center (KDC)
    - Authentication Server
    - Ticket Granting Server
  - Client -> sends username/passwd to Authentication server through hash -> Authentication server sends encrypted key to Ticket granting server -> Ticket granting allows the client to use file server
  - Clients never decrypts key
  - Some implementations use public key certificates
- Single Sign-on
  - Lightweight Directory Access Protocol (LDAP)
  
174. User Authentication
- Multi-factor authentication
  - What a user KNOWS: passwd
  - What a user HAS: key card, text message
  - What a user IS: biometric scanner
  - WHERE a user is: location/GPS signal
  - What a user DOES: drawing pattern, clicking a button
  - IEEE 802.1X
  - Network access control
    - Posture validation: applies a set of requirements to the login process
      - Minimum version of anti-virus SW before joining the network
      - OS version/patch status
  - MAC filtering
  - Captive Portal: wifi in a hotel, asking name/id

175. Physical Security
- Detection
  - Motion detection
  - Asset Tracking Tags
  - Video surveillance
  - Tamper detection : lock
- Prevention
  - Badges
  - Biometrics
  - Training
  - Access Control Vestibule (aka mantrap)
  - Locks: doors, racks, cabinets
  - Smart lockers
- Equipment disposal
  - Erase configuration/factor reset
  - Sanitize device
  - Darik's Boot and Nuke (DBAN): an open source application that securely erases hard drives
  - Hammering

176. Forensic Concepts
- Network Forensics
  - Detect suspicious activity
  - Incident investigation
- Cateogries
  - Catch-it-as-you-can
    - Needs large storage
  - Stop, look, listen
- Tools
  - wireshark.org
  - syslog

177. Securing STP
- Switch port protection
  - BPDU guard: a Cisco feature that shuts down a port if a BPDU is received
    - PortFast
  - Root guard: a switchport feature used to prevent another switch on the port which the feature is enabled on from changing its root bridge

178. Router Advertisement (RA) Guard
- SLAAC(Stateless Address Auto-Configuration): IPv6 feature that allows a client to dynamically determine its IPv6 address using EUI-64 address and a router advertisement (RA) message containing its network Prefix
- Potential threats
  - Convincing a client that attacker is the default router
  - Sending incorrect SLAAC information to client
  - RA guard

179. Securing DHCP
- DHCP snooping
  - Blocks untrusted DHCP offer

180. IoT security concerns
- Many IoT devices were not designed with security in mind
- IoT devices might use weak encryption to preserve processing power
- Many users leave the default passwords on IoT devices
- SW patches might not be automatically deployed
- IoT security best practice
  - Use strong passwd
  - Place IoT devices on their own VLAN

181. Cloud Security
- TLS tunnel 
- VPN connection
- Private WAN
- CASB (Cloud Access Security Broker): SW sits b/w users and cloud resources
  - Monitors traffic to enforce security polices
  - Generates alerts if any malicious activity is detected
  
182. IT Risk Management
- Five steps Model
  - Identify Attack Targets
  - Rank Data
  - Determine Risk Levels
    - Risk level = Probability of breach \* Financial impact of breach
  - Set Risk Tolerances
- Monitor
- Terms
  - Threat vs Vulnerability
  - Posture Assessment
  - Penetration Testing
  - Process and Vendor Assessments

## Section 17: Module 15: Monitoring and Analyzing Networks

183. Monitoring and analyzing networks

184. Device Monitoring Tools
- Security Information and Event Management (SIEM)
  - Collection of HW/SW
- Syslog
- Interface statistics
- CPU and memory statistics
- Monitoring Processes
  - Log reviewing
  - Port scan
  - Vulnerability scan
  - Patch management
  - Compare with baseline data
  - Packet analysis 
    - wiresharks.org
  - Netflow Collector

185. SNMP
- Simple Network Management Protocol
  - SNMP Manager
  - SNMP agent: sits on devices
  - Management Information Base (MIB)
  - Object Identifier (OID)
  - Trap Notification
  - Query/respond
- SNMP security options
  - version 1: Community strings
  - version 2c: Community strings
  - version 3: Encryption, Integrity Checking, and Authentication Services

186. Remote Access Methods
- Site-to-Site VPN : office environment
- Client-to-Site VPN : home or hotel
- Internet Protocol Security (IPsec)
  - May need configuration on all devices
- Secure Sockets Layer (SSL)
- Transport Layer Security (TLS)
  - Web-browser already has security layers
- Datagram Transport Layer Security (DTLS)
- Remote Desktop Protocol (RDP)
- Virtual Network Computing (VNC)
- http vs https
- ftp or tftp vs sftp or ftps

187. Environment Monitoring
- Temperature & humidity
  - Network Operations Center
  - Environmental Monitor: sends an alert through email, text, ...
- Power
  - Uninterruptible Power Supply (UPS)
  - Generator

188. Wireless Network Monitoring
- Wireless Survey SW: superimposes a heat map on a physical map

## Section 18: Module 18: Examining Best Practices for Network Administration

189. Examining Best Practices for Network Administration

190. Safety Procedures

191. Wiring Management
- Intermediate Distribution Frame (IDF): A common location within a building in which cables from nearby offices terminate (wiring closet)
- Main Distribution Frame (MDF)
- Minimum bend radius - do not bend optical cable too much
- Writing Best practice
  - Plenum cables: fire-resistant
  - Avoid spaghetii wirting
  - Use correct cable lengths
  - Label both ends of the cable

192. Power Management
- UPS
- SPS: mechanically switches to battery at outage. Might not be acceptable for some devices.

193. Rack Management
- Rack options
  - 2-Post
  - 4-Post
  - Rails
  - Lockable
- Rack management
  - Optimize cooling

194. Change Control
- Did anything change?
- Change control system
  - Team members must be aware of each other's change

195. High Availability
- The 5 nies of availability
  - 99.999%
  - 5 min per year
- Fault tolerance
  - Might impact performance
  - Increases complexity
- Multiple ISPs for internet access

196. Cloud High Availability
- Tier 1: 99.671% availability, 24hrs/year
- Tier 2: 99.741% availability, 22hrs/year
- Tier 3: 99.982% availability, 1.6hr/year
- Tier 4: 99.995% availability, 20.6min/year

197. Active-Active vs. Active-Passive
- From HQ to multiple ISPs
  - Active(ISP1)-Active(ISP2): higher througput
  - Active(ISP1)-Passive(ISP2): ISP2 as a backup
- First-Hop Redundancy Protocol (FHRP): a protocol that can provide redundancy to a subnet's default gateway
  - Hot Standby Router Protocol (HSRP): active-passive
    - Hello every 3 sec
  - Virtual Router Redundancy Protocol (VRRP):  ative-passive
    - Advertisement every 1 sec
  - Gateway Load Balancing Protocol (GLBP): active-active
    - Active Virtual Gateway (AVG) responds to the gateway queries with different MAC addresses, coordinating traffic

198. Disaster Recovery
- Types of backups
  - Full
  - Differential: backs up changes since last full backup
  - Incremental: backup all changes since last full, differential, or incremental backup
  - Snapshot: backup including state information
- Cold site
  - Power
  - HVAC
  - Floor space
  - No HW yet
- Warm site
  - Power
  - HVAC
  - Floor space
  - Server HW
- Hot Site
  - Power
  - HVAC
  - Floor space
  - Server HW
  - Synchronized already
- Service Level Agreement (SLA): The promise you make to your users about how long a system will be down in the event of a disaster
- Recovery Time Objective (RTO): The maximum amount of time a system will be offline after a disaster
- Recovery Point Objective (RPO): The maximum amount of data that can be lost due to a disaster
- Mean Time B/w Failures (MTBF): The average amount of time before a product fails
- Mean Time To Repair (MTTR): The average amount of time required to repair a failed product

199. Standards, Polices, and Rules
- Privileged User Agreement
- Password Policy
- On-Boarding/Off-Boarding Procedures
  - For new-hire/leaving employees
- Licensing Restrictions
- International Export Controls
- Data Loss Prevention
- Remote Access Policies
- Incident Response Polices
- BYOD Policy
- Acceptable Use Corporate Resource Policy
- NDA
- System Life Cycle
- Safety Procedures and Policies

200. Documentation
- Same as 199?

201. Site Survey

## Section 19: Module 17: Troubleshooting Networks

202. Troubleshooting Networks

203. 7-Step Troubleshooting Methodology
1) Identify the problem
  - Gather information
  - Duplicate the problem if possible
  - Question Users
  - Identify Symptoms
  - Determine if anything changed
  - Approach multiple problems individually
2)  Establish a theory of probable cause
  - Question the obvious
  - Consider Multiple Approaches
    - Top-to-Bottom/Bottom-to-Top of OSI model
    - Divide and Conquer
3) Test the theory to determine the cause
4) Establish a plan of action to resolve the problem and identify potential effects
5) Implement the solution or escalate as necessary
6) Verify full system functionality, and if applicable, implement preventive measures
7) Document Findings, Actions, and Outcomes

204. CLI Troubleshooting Utilities
- telnet -> ssh
- arp -a : gateway IP and MAC address
- nslookup
- dig www.google.com : nameserver utility
- host www.cnn.com
- ifconfig
- ping
- iptables --help
- sudo tcpdump -c 5 -v : captures 5 packets
- nmap :finds which ports are open
  - Not available in default distribution
- traceroute
- Windows command
  - arp -a
  - ipconfig
    - ipconfig/release : releases DHCP address
    - ipcofig /renew : gets new DHCP address
  - nbtstat -c
  - netstat : IP session status
  - tracert : shows router hop up to destiny IP
  - pathping : similar to tracert but collects statistics

205. Network Appliance Commands
- show config
- show running-config
- show startup-config
- show ip route
- Full duplex : both of send/receive
- Half duplex : one of send/receive
- CRC (Cyclic Redundancy Check): checks packets for errors
- Giant: an error that occurs when a frame's size is bigger than its MTU, yields FCS error
- RUNT: frame whose size is too small (less than 64 bytes for an ethernet frame)
- debug ospf hello
- Debugging is very expensive

206. Device Metrics and Sensors

207. Environmental Metrics and Sensors

208. Common LAN issues
- Attenuation: signal degradation by time
- Latency
  - 150ms for VoIP
- Jitter
- Crosstalk
- EMI
- Open/Short
- Incorrect Pin-Out
- Incorrect cable type
- Bad port
- Transceiver Mismatch
- TX/RX Reverse
- Duplex/Speed Mismatch
- Damaged Cables
- Bent pins
- Bottlenecks
- VLAN Mismatch
- Network Connection LED status indicator

209. Common Wireless Network Issues
- Reflection
- Refraction
- Absorption
- Attenuation
- Effective Isotropic Radiated Power (EIRP)
- Incorrect Antenna Type
- Incorrect Antenna Placement
- Channel Overlap
- Association Time
- Captive Portal Issues
- AP Disassociation
- Overcapacity
- Distance Limitations
- Wrong SSID
- Security Type Mismatch
- Power Levels
- Signal-to-Noise Ratio

210. Common Network Service Issues
- Name resolution
  - ping 8.8.8.8 # google dns server
- Wrong default gateway IP
- Wrong subnet mask
- Overlapping IP Addresses
- Overlapping MAC Addresses
- Expired IP Addresses
- Rogue DHCP server
- Untrusted SSL Certificate
- Wrong time
- Exhausted DHCP Scope
- Blocked ports (by firewall)
- Incorrect Firewall settings
- Incorrect ACL settings
- Service not Responding
- HW Issue

211. General Networking Issues
- Device Misconfiguration
- Missing routes
- Routing loops
- Interface Status
- Baselines
- Collisions
- Broadcast Storm
- Multicast Flooding
- Asymmetrical Routing
- DNS issues
- NTP issues
