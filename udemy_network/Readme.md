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
