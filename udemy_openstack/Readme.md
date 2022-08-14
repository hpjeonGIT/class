## Udemy - OpenStack Essentials
- Instructor: Ugur Oktay

## Section 1: Introduction

1. Why should you learn Openstack?

2. About the course

## Section 2: Cloud Computing Overview

3. Introduction to Cloud Computing
- Traditional on-prem. data center vs Cloud computing
- Cloud computing
  - SaaS: for SW
  - PaaS: for SW development
  - IaaS: for SW insfrastructure
  - Features:
    - On-demand self-service
    - Broad network access
    - Resource pooling
    - Rapid elasticity
    - Measured service

4. Cloud Enabling Technologies and comparison with Traditional IT
- Virtualization
- Automation and orchestration

5. Cloud Computing Delivery and Deployment Types
- Delivery model
  - SaaS
  - PaaS
- Deployment model
  - Public
  - Private
  - Hybrid

## Section 3: Introduction to Openstack

6. Introduction
- IaaS
- Through dashboard, all resouces are managed
- Common 6 core components
  - Nova
  - Neutron
  - Cinder
  - Swift
  - Keyston
  - Glance 

7. Overview of Component Services
- Overview
  - Nova: handles computing nodes
  - Neutron: handles network
  - Cinder: Block storage for instance
  - Swift: Object storage
  - Keyston: central registration
  - Glance: OS image
- Q: where load balancer is located?

8. Architecture

## Section 4: Openstack Installation

9. About Packstack & HW requirements
- Packstack is a utility that uses Puppet modules to deploy RDO components over SSH
- Q: Do we need Puppet? This is not free
- Q: No Ubuntu support? UBUNTU is not supported

10. CentOS Installation on VirtualBox

11. Openstack Installation
- /etc/environment
```bash
LANG=en_US.utf-8
LC_ALL=en_US.utf-8
```
- Disable SELinux by disabling /etc/selinux/config
- sudo yum install -y centos-release-openstack-train
- yum install -y yum-utils
- yum-config-manager --enable openstack-train
- yum update -y
- yum install -y openstack-packstack
  - This is a collection of puppet scripts
- id address show
- packstack --allinone --provide-demo=n --os-neutron-ovs-bridge-mappings=extnet:br-ex --os-neutron-ml2-mechanism-drivers=openvswitch --os-neutrol-l2-agent=openvswitch --os-neutron-ovs-bridge-interfaces=br-ex:enp0s3 --os-neutron-ml2-type-drivers=vxlan,flat --os-neutron-ml2-tenant-network-types=vxlan
  - May take > 0.5 hr
- ip address show
  - The existing interface will be updated with the ovswitch
- neutron net-create external_network --provider:network_type flat --provider:physical_network extnet --router:external
- Configure a new public subnet
  - neutron subnet create --name public_subnet --enable_dhcp-False --allocation-pool-start=192.168.0.100,end=192.168.0.120 --gateway=192.168.0.1 external_network 192.168.0.0/24

12. Verification
- Open browser and check horizon dashboard
- Overview of health monitor
  - System Information -> Compute Services/Block Storage Services/Network Agents
  - Compute -> Hypervisor
- cd /etc/systemd/system/multi-user.target.wants/ && ls *.service

## Section 5: Horizon Dashboard

13. Overview
- Additional third party tools like billing can be added to Horizon
- Default openstack uses http but https can be used as well

14. Dashboard Walkthrough
- The passwd can be found in the keystonerc_admin file
- Create Project -> Set up quotas of memory, disk, vcpus, ...
- Trunk port
  - Ref: https://www.techopedia.com/definition/27008/trunk-port
  - A port on a network switch that allows data to flow across a network node for multiple virtual local area networks or VLANs

## Section 6: CLI Client

15. Working from the CLI
- Needs to install CLI package, not default
- Can provide credentials from command
  - Not recommended
- Provide as system file
  - keystonerc_admin
- Commands:
  - openstack server list
  - source keystonerc_admin

16. Unified CLI client
- Previously each project had its own CLI
- Unified command line
  - .nova boot -> openstack server create
  - .neutro net-create -> openstack network create
  - .glance image-list -> openstack image list
  - .cinder create -> openstack volume create
  - Not all commands are available unified CLI

## Section 7: Identity Service - Keystone

17. Introduction & Important Identity Concepts
- Keystone
  - Common authentication system
  - Central catalog of services and endpoints
  - Supports LDAP, AD, MySQL
  - Provides a token for subsequent auth. requests
- Keystone concepts
  - User
  - Project (or tenant)
  - Role
    - Globally scoped
    - Project scoped
  - Token
  - Catalog

18. Keystone Architecture
- Authentication process flow
  - user -> credentials -> keystone
  - keystone -> token -> user
  - user -> token & VM request -> nova
  - nova <-> token verification <-> keystone
  - nova -> token & image request > glance
  - glance <-> verify token <-> nova
  - glance -> image -> nova
  - nova -> token & VIF plugin to net request -> neutron
  - neutron <-> verify token <-> keystone
  - neutron <-> token + verify user access to VIF <-> nova
  - neutron -> successful response -> nova
  - nova -> successful response -> user
- Polices and authorisation
```
        project
      /         \
 user -- role --- resource -- action
```
- Keystone API
  - Policies
  - Token
  - Catalog
  - Identity
  - Assignment
  - Credentials
- Nitty Gritty
  - Uses internal SQL database to store data

19. Managing Keystone from CLI
- openstack endpoint list
- openstack catalog list
- openstack endpoint show <ID>
- openstack command list

## Section 8: Image Service - Glance

20. Overview & Architecture
- Glance
  - Stores cloud vm images and snapshots
  - RESTful API
  - Swift or other object storage backend
- Architecture
  - REST API
  - Glance domain controller
  - Glance Store drivers
    - File system
    - Swift
  - Registry layer
  - Database abstraction layer
    - Database
- Image types
  - Raw
  - VHD (Hyper-V)
  - VID (VirtualBox)
  - Qcow2(Qemu/KVM)
  - VMDK, OVF (VMware)
  - docker
- CirrOS
  - Very small (13MB) and suitable for test OS image

21. Managing Glance from CLI
- openstack image list
- Download cirros OS image
- curl -o /root/cirros-0.3.4.img http://download.cirros-cloud.net/0.3.4/cirros-0.3.4-x86_64-disk.img
- openstack image create --min-disk 2 --private --disk-format qcow2 --file <image_file> <image_name>
- openstack image show <image_name_or_ID>

## Section 9: Networking Service - Neutron

22. Introduction to Neutron
- Network connectivity as a service 
- Network, subnet and port abstraction
- Plugins support many technologies
- Modular architecture
- Central or distributed deployment
- Benefits of Neutron
  - Rich topologies
  - Technology agnostic
  - Pluggable open architecture
  - Enables advanced services
    - Load balancing, VPN, firewall
- Base terminology and abstractions
  - Nova : VM/VIF(Virtual interface)
  - Neutron: Virtual port/virtual subnet/L2 Virtual network

23. Architecture
- Neutron Server 
  - REST API
    - Exposes logical resources like subnets, ports, ...
  - Plugin
    - Only one active
    - Optional extension support
  - Queue
    - Enables bidirectional agent communications
- Neutron Architecture
  - Database
  - Neutron Server
  - Queue
    - L2 agents
    - L3 agents
    - DHCP agents
    - Advanced services
- ML2: Modular Layer 2 Plugin
  - For VXLAN TypeDriver
  - VLAN TypeDriver
  - GRE TypeDriver
- Plugin Exensions
  - Adds logical resources to the REST API
  - Discovered by server at startup
  - Common extensions: DHCP, L3, Quota, Security Groups, Provider Networks
- Which neutron component resides where?
  - Controller node: neutron-server, nova-api, nova-scheduler, nova-conductor, keystone-all, glance-api, glance-registry, rabiitmq-server
  - Network node: neutron-*plugin-agent, neutron-dhcp-agent, neutron-l3-agent, neutron-metering-agent
  - Compute node: nova-compute, neutrol-*plugin-agent

24. Provide & Project Networks
- Per tenant networking
- Provider networks
  - Flat
  - VLAN
  - VXLAN
  - GRE
  - Layer 2 features only

25. Supported Network Types
- Traffic segmentation is a must
  - For scalability, security and network management
  - Large networks degrade performance
  - Need to separate tenant traffic
- Choices
  - Local network
    - Isolated networks tha live on a single compute node
    - For test and POC environment only
  - Flat network
    - No segmentation
    - No 802.1Q tagging or segmentation
    - Single broadcast domain
    - Not scalable
  - VLAN
    - Layer 2 implementation
    - Separate broadcast domains
    - IEEE 802.1Q VLAN tagging
    - Better security by network segmentation
    - Implemented by all routers and switches/most of NIC cards
    - Limited to 4096 VLANs
    - Inter VLAN will need L3 routers
  - GRE and VXLAN Tunneling
    - Layer 3 protocols
    - GRE (Generic Routing Encapsulation)
      - MAC in IP encapsulation
      - Not supported by most NICs
        - more CPU overhead
    - Virtual Extensible LAN (VXLAN)
      - MAC in UDP encapsulation
      - 24bit VLAN address bits

26. Common Neutron Agents
- L2 agent
  - Runs on compute node
  - Communicates with neutron server via RPC
  - Watch/notify when devices added/removed
  - Wires up new devices
    - Network segment
    - Security group rules
- OVS L2 agent
  - Open vSwitch
  - Open source virtual switch (openvswitch.org)
  - VLAN, GRE, VXLAN
- L3 agent
  - Runs on network node
  - Uses linux namespaces
  - Metadata agent
  - Supports HA

27. Neutron Features & Functionality
- Security Groups
  - Set of IP tables rules
  - Ingress/Egress Rules
  - Support overlapping IPs
  - IPv6 support
  - Applied per VIF
  - VMs with multiple VIFs supported
  - Statesful: if one way connection is allowed, the other way is done automatically
- NAT
  - Source address translation
  - Destination address translation
  - Port address translation
- Floating IP addresses
  - Neutron L3 agent's task
  - Static one to one mapping
    - Private address <-> publicly routable address
  - Makes instances reachable from outside
- Distributed Virtual Router (DVR)
  - L3 agent is centralized on network nodes
    - Inter subnet VM traffic hits L3 agent
    - Any traffic from outside world hits L3 agent
  - DVR
    - Runs on each compute node
    - Mitigates the performance impact
- Network namespaces
  - Isolated copy of network stack
    - Scope limited to each namespace
    - Each namespace has its own network devices, routing tables, IP addresses
    - Can reuse addresses
  - Explicit configuration needed to connect b/w namespaces
  - `ip netns`

28. Managing Neutron from CLI
- openstack network agent list
- systemctl status neutron-server
- ovs-vsctl show
- openstack network create -h
- openstack network create <name>
- openstack subnet create <subnet_name> --subnet-range <IP_subnet> --dns-nameserver <IP> --network intnet
- ip netns # shows namespace
- ip netns exec <namepspace_name> exec ip address show
- openstack router create <name>
- openstack router add subnet R2 subnet1
- neutron router-gateway-set R2 external_network # adding gateway
- openstack security group list
- openstack project list
- openstack security group rule create --src-ip 0.0.0.0/0 --protocol icmp --ingress <sec_group_ID>
- openstack security group rule create --src-ip 0.0.0.0/0 --dst-port 22 --protocol tcp --ingress <sec_group_ID>
- openstack subnet list
- openstack floating ip create --subnet <subnet_ID> <network_ID>
- openstack server add floating ip <instance_name> <floating_IP>
- ping <floating_IP>

## Section 10: Compute Service - Nova

29. Introduction to Nova
- Provides instance lifecycle management
- Multple hypervisors supported
  - KVM
  - QEMU
  - UML
  - VMware
  - Xen
  - LXC
  - Bare metal
- Key pairs
  - A way of authentical w/o passwords
  - Injected to the image with the help of cloud-init process
  - Not specific to openstack

30. Architecture
- REST API listens on TCP port 8774
- Nova-compute intefaces with the hypervisor
- Nova-scheduler handles hypervisor selection for instances
- Legacy Nova networking: will deprecate in future
- L2 agent: 
  - Neutron-openvswitch-agent
  - Neutron-linuxbridge-agent
- Metadata service: nova-metadata-api

31. Launching an instance
- Prerequistes for launching an instance
  - An image
  - A network
  - A flavor
    - Resources of memory, vCPU, storage
- openstack flavor list
  - types of instances available
  - Ex) m1.tiny, m1.small, m1.medium, m1.large, m1.xlarge
- Instance creation
  - openstack server create --image <image> --flavor <flavor> --nic net-id=<net-id> <instance-name>
- openstack image list
- openstack image show rhel-guest-image-XXXX
  - Check disk size, memory size
- openstack network list

32. Launching an instance (cont.)
- openstack server create --image <image> --flavor <flavor> --nic net-id=<net-id> <instance-name>
  - Instantly generates status but it is still building VM
- Instance scheduling
  - Filter scheduler kicks in when creating the instance
  - Checks if the resource is available
  - Applies weights to access the request
  - Database is updated with instance state
- Compute agent
  - Prepares for instance creation
    - Communicates with Neutron/Cinder for network/attaching a volume
  - Communicates with hypervisor to create the VM
  - Updates instance state in database using conductor
    - No direct access due to security issue

33. Grouping Compute nodes
- Segregation of compute resources
  - Provide logical groupings: Data center, geographical region, power source, rack, network resource
  - Differentiate specific HW: GPU, Fast NICs, SSDs
- Regions
  - Complete openstack deployments where: 
    - Implement their own API endpoints, compute, storage, network etc
    - Shares as many services as required
  - By default all services in one region
- Host aggregates
  - Logical grouping of compute nodes based on metadata
    - SSD, GPU, ...
  - Implicitly targetable
    - Admins defines host aggregate with metadata and flavor to match
- Availability Zones
  - Logical groups of hosts based on factors like
    - Geo location (country, city, datacenter, rack)
    - Network layout
    - Power source
  - Explicitly user targetable
    - User can specify AZ
  - Host aggregates are made explicitly targetable by creating them as an AZ
  - Unlike host aggregates hosts can't be in multiple AZs

34. Managing Nova from CLI
- openstack compute service list
- openstack flavor list
- openstack flavor create -h
- openstack flavor create --id <ID> --ram <amount_of_ram> --disk <disk_size> --public <name>
- openstack server create -h
- openstack keypair create mykeypair >> mykeypair.key
- openstack image list
- openstack network list
- openstack server create --image <image_name> --key-name <keypair_name> --flavor <flavor_ID> --nic net-id =<network_ID> <instance_name>
- openstack server show <name>
- openstack server image create --name <snapshot_name> <instance_name> # making a shapshot of the instance
- openstack aggregate create --property SSD=true # creating host-aggregate
- openstack console url show --novnc <instance_name>
- openstack console log show

## Section 11: Block Storage Service - Cinder

35. Overview of storage in openstack

| | Ephemeral storage | Blck storage | Object storage | Shared File system storage |
|--|--|--|--|--|
|Pupose| run OS and scratch space. a part of instance | Add additional persistent storage to a VM | Store data/files include VM images | add additional shared storage to a VM shared|
|accessed through | a file system | a block device partitioned/formatted/mounted as dev/vdc | REST API | s shared file system device share which can be partitioned, formatted, and mounted |
| accessible from | within a VM | within a VM | anywhere | within a VM |
|persists until | VM is terminated | deleted by user |deleted by user | deleted by user|
| managed by| openstack compute | openbstack block storage | openstack object storage | openstack shared file system service |
| typical usage example| 10gb first disk, 30gb second disk | 1 TB disk | 10s of TBs of dataset storage | 1 TB shared disk|

36. Cinder - Introduction & Capabilities
- persistent block level storage devices for use with compute instances
- One to one mapping in the instance
- Cannot share b/w multiple instances
- Volumes
  - Persistent R/W block stroage
  - Attached to instances as secondary storage
  - Can be used as root volme to boot instances
  - Volume lifescycle management
    - Create, delete, extend volumes
    - Attach/detach volumes
  - Manages volume management
- Snapshots
  - A read only copy fo a volume
  - Create/delete snapshots
  - Create a volume out of a snapshot
- Backup
  - Done from CLI
  - Backup stored in Swift
    - Needs a container
- Quotas
  - Enforced per project
    - Number of volumes
    - Storage space in gigabytes
    - Number of snapshots

37. Cinder Architecture
- Data path and control path are completely different

38. Managing Cinder form CLI
- Attache a volume to a single instance at a time
- cinder service-list
- cinder service-disable
- cinder service-enable
- openstack command list |grep openstack.volume2
- openstack volume create --size 1 vol1 
- openstack volume list # make sure if the creation was successful
- openstack server list
- SSH into the instance and check ls /dev

## Section 12: Object Storage Service - Swift

39. Introduction to Swift Object Storage Service
- Scalable, distributed, replicated object storage
- Simple, Powerfule Restful API
- High concurrency support - lots of users
- Pooled storage capacity
- Ex: wikipedia, movies, ...
- Object: file + metadata
- Need for object storage
  - Traditional file based: app -> file
  - Object storage - http namespace: SaaS/mobile devices -> replicate objects as necessary
    - Built-in web-service capability

40. Characteristics of Swift
- Data consistency
  - Strict consistency
    - All replicas are written to completion in all regions and zones before the write operation is considered successful
  - Eventual consistency
    - All replicas written at the same time, only most replicas are reuqired to declare success
- Durability with replicas
  - Swift stores multiple replicas
  - 3 replicas is the default
    - Good balance b/w durability and cost effective ness
    - Can be changed
  - Stores MD5 checksums with each object
- Data placment in Swift
  - Geographic region
    - Availability zone
      - Server
        - Disk
- Swift essentials: https://sorage.example.com/v1/AUTH_acct/cont/obj
  - v1: API version
  - AUTH_acct: account
  - cont: container
  - obj: object

41. Swift Architecture
- Swift API
  - Write an object: PUT /v1/account/container/object
  - Read an object: GET /v1/account/container/object
- System components
  - Authentication +  load balancing + ssl
  - proxy
  - account + container + object service
  - replication + consistency
  - standard x86 servers with disks
- Swift Architecture
  - HTTP <-> proxy <-> ring <-> account/object/container service

42. Managing Swfit from CLI
- Can run as a standalone, without openstack
- openstack object store account show
- openstack container list
- openstack container create <name>
- openstack container list
- openstack object create -h
- openstack object create <container_name> <file_name>
- swift tempurl
- swift post -m "Temp-URL-key:<arbitrary_word>
- openstack object store account show
- swift tempurl get <seconds> <path> <key>
- openstack endpoint show <swift_ID>
- Same process can be done in Horizon dashboard
- Now file can be access from a web-browser

## Seciton 13: Working with Horizon Dashboard

43. Introduction
- Sample run of building a project

44. Requirements for Launching an instance
- Compute node -> hypervisor (ex:KVM) -> app/vm
- Instance (Nova)
  - Must
    - Image (Glance)
    - Network (Neutron)
    - Size of the instance (Nova)
  - Optional
    - Security settings
      - ACLs (Neutron)
      - Key pair (Nova)
    - Persistent storage (Cinder)

45. Creating the image & flavor

46. Network environment for the instance
- Floating IPs
  - Public IPs <-> Private IPs
  - Senders/receivers are not aware of the details of mappings

47. Setting up the network
- Gateway IP is not required
  - Automatically configured
- Default subnet uses DHCP
- DNS as 8.8.8.8
- Project -> Network -> Network Topology
  - Create Router

48. Optional Configuration
- Security configuration
  - Security group
  - SSH key pair
    - Enables passwordless login
- Adding a persistent volume

49. Security configuration & instance launch
- Project -> Network -> Security Groups
  - Manage Security group
    - Add rule for Ingress/Egress
    - Add CIDR for more restriction

50. Testing & Managing the instance

## Seciton 14: Multi Node Design & Scaling Openstack

51. Scaling Openstack
- Deployment scale
  - Walmart: 200K cpus
- Horizontal scaling (scale out)
  - Adding more machines to the pool of resource
- Vertical scaling (scale up)
  - Adding more power (CPU/RAM) to an existing machine
- How openstack components scale?
  - REST APIs can scale natively with load balancers, one API end point should be representing them
  - Some components do not scale naturally
    - Databse 
    - Message queue

52. Compute Node
- All-in-one-node: all services in a single node
- Let's add compute node
  - nova compute, neturon-pluing-agent, hyper visor will run on each compute node
- Storage for instances
  - Non-compute node based shared file system
    - Disks storing the running instances are hosted in servers outside of the computing nodes
    - Compute hosts are stateless
  - On compute node storage - shared file system
    - Each compute node is specified with a significant amount of disk space
    - Data locality is lost
    - Instance recovery might be complicated
  - On compute node storage - nonshared file system
    - Local disks on compute node
    - Heavy IO doesn't affect other nodes
    - Instance migration is complicated

53. Network Node
- Handles virtual netoworking needs of cloud consumers
  - Routing
  - NAT/floating IPs
  - Creates logical routers as instance gateway
  - Creates and manages namespaces
- L4-L4 advanced services
  - Load balancer as a service
  - Firewall as a service
- DVR can offload network node

54. Storage Node

55. Controller Node

56. Multi node design and minimum node requirements
- 2 node setup: controller node + compute node
- 3 node setup: controller node + compute node + network node
- 4 node setup: controller node + compute node + network node + storage node

## Section 15: Expanding the cluster: Adding a Compute Node

57. Preparing the node
- How to add a node to an existing system

58. Compute node installation
- packstack: a utility that uses Puppet modules to deploy various parts of OpenStack on multiple pre-installed servers over SSH automatically

59. Installation Troubleshooting & Verification

## Section 16: Final Section

60. Openstack logs
- Standard logging levels at increasing severity
  - TRACE, DEBUG, INFO, AUDIT, WARNING, ERROR, and CRITICAL
- To enable logs
  - For neutron, /etc/neutoron/netron.conf -> debug=true
  - For nova, /etc/nova/nova.conf -> debug=true
  - For keyston, /etc/keystone/loggin.conf and look at logger_root and handler_file sections
  - For horizon, /etc/openstack_dashboard/local_settings.py
  - For cinder, edit the configuration file on each node with cinder role
  - For glance, /etc/glance/glance-api.conf and /etc/glance/glance-regisry.conf
- Location of log files
  - nova service: /var/log/nova
  - glance service: /var/log/glance
  - cinder service: /var/log/cinder
  - keystone service: /var/log/keystone
  - neutron service: /var/log/neutron
  - horizon: /var/log/apache2/
  - swift/dnsmasg: /var/log/syslog
  - libvirt: /var/log/libvirt/libvirtd/log
  - cinder-volume: /var/log/cinder/cinder-volume.log

61. Glossary of Terms
- access control list (ACL): A list of permissions attached to an object. An ACL specifies which users or system processes or IPs have access to objects or networks. It also defines which operations can be performed on specified objects. Each entry in a typical ACL specifies a subject and an operation. For instance, the ACL entry (10.1.1.0/24, permit) for a network gives permission to access to 10.1.1.0/24 network.
- account: The Object Storage context of an account. Do not confuse with a user account from an authentication service, such as Active Directory, /etc/passwd, OpenLDAP, OpenStack Identity, and so on.
- active directory: Authentication and identity service by Microsoft, based on LDAP. Supported in OpenStack.
- active/active configuration: In a high-availability setup with an active/active configuration, several systems share the load together and if one fails, the load is distributed to the remaining systems.
- active/passive configuration: In a high-availability setup with an active/passive configuration, systems are set up to bring additional resources online to replace those that have failed.
- address pool: A group of fixed and/or floating IP addresses that are assigned to a project and can be used by or assigned to the VM instances in a project.
- admin API :A subset of API calls that are accessible to authorized administrators and are generally not accessible to end users or the public Internet. They can exist as a separate service (keystone) or can be a subset of another API (nova).
- administrator: The person responsible for installing, configuring, and managing an OpenStack cloud.
- Advanced Message Queuing Protocol (AMQP): The open standard messaging protocol used by OpenStack components for intra-service communications, provided by RabbitMQ, Qpid, or ZeroMQ.
- alert: The Compute service can send alerts through its notification system, which includes a facility to create custom notification drivers. Alerts can be sent to and displayed on the dashboard.
- API endpoint: The daemon, worker, or service that a client communicates with to access an API. API endpoints can provide any number of services, such as authentication, sales data, performance meters, Compute VM commands, census data, and so on.
- API extension: Custom modules that extend some OpenStack core APIs.
0 API extension plug-in: Alternative term for a Networking plug-in or Networking API extension.
- API version: In OpenStack, the API version for a project is part of the URL. For example, example.com/nova/v1/foobar.
- Application Programming Interface (API): A collection of specifications used to access a service, application, or program. Includes service calls, required parameters for each call, and the expected return values.
- Address Resolution Protocol (ARP): The protocol by which layer-3 IP addresses are resolved into layer-2 link local addresses.
- attach: The process of connecting a VIF or vNIC to a L2 network in Networking. In the context of Compute, this process connects a storage volume to an instance.
- attachment (network): Association of an interface ID to a logical port. Plugs an interface into a port.
- auditor: A worker process that verifies the integrity of Object Storage objects, containers, and accounts. Auditors is the collective term for the Object Storage account auditor, container auditor, and object auditor.
- authentication: The process that confirms that the user, process, or client is really who they say they are through private key, secret token, password, fingerprint, or similar method.
- authentication token: A string of text provided to the client after authentication. Must be provided by the user or process in subsequent requests to the API endpoint.
- authorization: The act of verifying that a user, process, or client is authorized to perform an action.
- availability zone: An Amazon EC2 concept of an isolated area that is used for fault tolerance. Do not confuse with an OpenStack Compute zone or cell.
- bandwidth: The amount of available data used by communication resources, such as the Internet. Represents the amount of data that is used to download things or the amount of data available to download.
- bare: An Image service container format that indicates that no container exists for the VM image.
- base image: An OpenStack-provided image.
- Benchmark service (rally): OpenStack project that provides a framework for performance analysis and benchmarking of individual OpenStack components as well as full production OpenStack cloud deployments.
- block device: A device that moves data in the form of blocks. These device nodes interface the devices, such as hard disks, CD-ROM drives, flash drives, and other addressable regions of memory.
-block migration: A method of VM live migration used by KVM to evacuate instances from one host to another with very little downtime during a user-initiated switchover. Does not require shared storage. Supported by Compute.
- Block Storage API: An API on a separate endpoint for attaching, detaching, and creating block storage for compute VMs.
- Block Storage service (cinder): The OpenStack service that implement services and libraries to provide on-demand, self-service access to Block Storage resources via abstraction and automation on top of other block storage devices.
- catalog: A list of API endpoints that are available to a user after authentication with the Identity service.
- catalog service: An Identity service that lists API endpoints that are available to a user after authentication with the Identity service.
- ceilometer: Part of the OpenStack Telemetry service; gathers and stores metrics from other OpenStack services.
- cell: Provides logical partitioning of Compute resources in a child and parent relationship. Requests are passed from parent cells to child cells if the parent cannot provide the requested resource.
- CentOS: A Linux distribution that is compatible with OpenStack.
- Ceph: Massively scalable distributed storage system that consists of an object store, block store, and POSIX-compatible distributed file system. Compatible with OpenStack.
- cinder: Codename for Block Storage service.
- CirrOS: A minimal Linux distribution designed for use as a test image on clouds such as OpenStack.
- cloud computing: A model that enables access to a shared pool of configurable computing resources, such as networks, servers, storage, applications, and services, that can be rapidly provisioned and released with minimal management effort or service provider interaction.
- cloud controller: Collection of Compute components that represent the global state of the cloud; talks to services, such as Identity authentication, Object Storage, and node/storage workers through a queue.
- cloud controller node: A node that runs network, volume, API, scheduler, and image services. Each service may be broken out into separate nodes for scalability or availability.
- cloud-init: A package commonly installed in VM images that performs initialization of an instance after boot using information that it retrieves from the metadata service, such as the SSH public key and user data.
- Common Internet File System (CIFS): A file sharing protocol. It is a public or open variation of the original Server Message Block (SMB) protocol developed and used by Microsoft. Like the SMB protocol, CIFS runs at a higher level and uses the TCP/IP protocol.
- Common Libraries (oslo): The project that produces a set of python libraries containing code shared by OpenStack projects. The APIs provided by these libraries should be high quality, stable, consistent, documented and generally applicable.
- compute instance/instance: A virtual machine launched by openstack or namely the nova component of openstack. Each instance use some amount of ram, cpu and storage resources which can be chosen before spinning up the instance.
- Compute API (Nova API): The nova-api daemon provides access to nova services. Can communicate with other APIs, such as the Amazon EC2 API.
- compute controller: The Compute component that chooses suitable hosts on which to start VM instances.
- compute host: Physical host dedicated to running compute nodes.
- compute node: A node that runs the nova-compute daemon that manages VM instances that provide a wide range of services, such as web applications and analytics.
- Compute service (nova): The OpenStack core project that implements services and associated libraries to provide massively-scalable, on-demand, self-service access to compute resources, including bare metal, virtual machines, and containers.
- compute worker: The Compute component that runs on each compute node and manages the VM instance lifecycle, including run, reboot, terminate, attach/detach volumes, and so on. Provided by the nova-compute daemon.
- conductor: In Compute, conductor is the process that proxies database requests from the compute process. Using conductor improves security because compute nodes do not need direct access to the database.
- Config Drive: The configuration drive is used to store instance-specific metadata and is present to the instance as a disk partition labeled config-2. The configuration drive has a maximum size of 64MB. One use case for using the configuration drive is to expose a networking configuration when you do not use DHCP to assign IP addresses to instances.
- console log: Contains the output from a Linux VM console in Compute.
- container: Organizes and stores objects in Object Storage. Similar to the concept of a Linux directory but cannot be nested. Alternative term for an Image service container format.
- controller node: Alternative term for a cloud controller node.
- credentials: Data that is only known to or accessible by a user and used to verify that the user is who he says he is. Credentials are presented to the server during authentication. Examples include a password, secret key, digital certificate, and fingerprint.
- daemon: A process that runs in the background and waits for requests. May or may not listen on a TCP or UDP port. Do not confuse with a worker.
- Dashboard (horizon): OpenStack project which provides an extensible, unified, web-based user interface for all OpenStack services.
- DHCP agent: OpenStack Networking agent that provides DHCP services for virtual networks.
- disk format: The underlying format that a disk image for a VM is stored as within the Image service back-end store. For example, AMI, ISO, QCOW2, VMDK, and so on.
- distributed virtual router (DVR): Mechanism for highly available multi-host routing when using OpenStack Networking (neutron).
- Django: A web framework used extensively in horizon.
- DNS record: A record that specifies information about a particular domain and belongs to the domain.
- domain: An Identity API v3 entity. Represents a collection of projects, groups and users that defines administrative boundaries for managing OpenStack Identity entities. On the Internet, separates a website from other sites. Often, the domain name has two or more parts that are separated by dots. For example, yahoo.com, usa.gov, harvard.edu, or mail.yahoo.com. Also, a domain is an entity or container of all DNS-related information containing one or more records.
- Domain Name System (DNS): A system by which Internet domain name-to-address and address-to-name resolutions are determined. DNS helps navigate the Internet by translating the IP address into an address that is easier to remember. For example, translating 111.111.111.1 into www.yahoo.com. All domains and their components, such as mail servers, utilize DNS to resolve to the appropriate locations. DNS servers are usually set up in a master-slave relationship such that failure of the master invokes the slave. DNS servers might also be clustered or replicated such that changes made to one DNS server are automatically propagated to other active servers. In Compute, the support that enables associating DNS entries with floating IP addresses, nodes, or cells so that hostnames are consistent across reboots.
- Dynamic Host Configuration Protocol (DHCP): A network protocol that configures devices that are connected to a network so that they can communicate on that network by using the Internet Protocol (IP). The protocol is implemented in a client-server model where DHCP clients request configuration data, such as an IP address, a default route, and one or more DNS server addresses from a DHCP server. A method to automatically configure networking for a host at boot time. Provided by both Networking and Compute.
- east-west traffic:  Server to server communication, it’s mostly Layer 2 traffic that stays inside your cloud or datacenter. See also north-south traffic.
- EC2: The Amazon commercial compute product, similar to Compute.
- encapsulation: The practice of placing one packet type within another for the purposes of abstracting or securing data. Examples include GRE, MPLS, or IPsec.
- endpoint: See API endpoint.
- endpoint registry: Alternative term for an Identity service catalog.
- ephemeral image: A VM image that does not save changes made to its volumes and reverts them to their original state after the instance is terminated.
- ephemeral volume: Volume that does not save the changes made to it and reverts to its original state when the current user relinquishes control.
- ESXi: An OpenStack-supported hypervisor.
- external network: A network segment typically used for instance Internet access.
- extra specs: Specifies additional requirements when Compute determines where to start a new instance. Examples include a minimum amount of network bandwidth or a GPU.
- filter: The step in the Compute scheduling process when hosts that cannot run VMs are eliminated and not chosen.
- firewall: Used to restrict communications between hosts and/or nodes, implemented in Compute using iptables, arptables, ip6tables, and ebtables.
- FireWall-as-a-Service (FWaaS): A Networking extension that provides perimeter firewall functionality.
- fixed IP address: An IP address that is associated with the same instance each time that instance boots, is generally not accessible to end users or the public Internet, and is used for management of the instance.
- flat network: Virtual network type that uses neither VLANs nor tunnels to segregate project traffic. Each flat network typically requires a separate underlying physical interface defined by bridge mappings. However, a flat network can contain multiple subnets.
- flavor: Alternative term for a VM instance type.
- flavor ID: UUID for each Compute or Image service VM flavor or instance type.
- floating IP address: An IP address that a project can associate with a VM so that the instance has the same public IP address each time that it boots. You create a pool of floating IP addresses and assign them to instances as they are launched to maintain a consistent IP address for maintaining DNS assignment.
- generic routing encapsulation (GRE): Protocol that encapsulates a wide variety of network layer protocols inside virtual point-to-point links.
- glance: Codename for the Image service.
- glance API server: Alternative name for the Image API.
- glance registry: Alternative term for the Image service image registry.
- GlusterFS: A file system designed to aggregate NAS hosts, compatible with OpenStack.
- guest OS: An operating system instance running under the control of a hypervisor.
- HAProxy: Provides a load balancer for TCP and HTTP-based applications that spreads requests across multiple servers.
- high availability (HA): A high availability system design approach and associated service implementation ensures that a prearranged level of operational performance will be met during a contractual measurement period. High availability systems seek to minimize system downtime and data loss.
- horizon: Codename for the Dashboard.
- horizon plug-in: A plug-in for the OpenStack Dashboard (horizon).
- host: A physical computer, not a VM instance (node).
- host aggregate: A method to further subdivide availability zones into hypervisor pools, a collection of common hosts.
- hybrid cloud: A hybrid cloud is a composition of two or more clouds (private, community or public) that remain distinct entities but are bound together, offering the benefits of multiple deployment models. Hybrid cloud can also mean the ability to connect colocation, managed and/or dedicated services with cloud resources.
- Hyper-V: One of the hypervisors supported by OpenStack.
- hypervisor: it’s a software that runs on top bare metal servers that enables quickly switching between processes. This allows for you to have multiple operating systems running on top of a physical server  and each of those operating systems could access to an abstracted form of the hardware like NICs, graphics cards, memory and similar.
- Identity service (keystone): The project that facilitates API client authentication, service discovery, distributed multi-project authorization, and auditing. It provides a central directory of users mapped to the OpenStack services they can access. It also registers endpoints for OpenStack services and acts as a common authentication system.
- Identity service API: The API used to access the OpenStack Identity service provided through keystone.
- ID number: Unique numeric ID associated with each user in Identity, conceptually similar to a Linux or LDAP UID.
- Identity API: Alternative term for the Identity service API.
- IETF: Internet Engineering Task Force (IETF) is an open standards organization that develops Internet standards, particularly the standards pertaining to TCP/IP.
- image ID: Combination of a URI and UUID used to access Image service VM images through the image API.
- Image service (glance): The OpenStack service that provide services and associated libraries to store, browse, share, distribute and manage bootable disk images, other data closely associated with initializing compute resources, and metadata definitions.
- Infrastructure-as-a-Service (IaaS): IaaS is a provisioning model in which an organization outsources physical components of a data center, such as storage, hardware, servers, and networking components. A service provider owns the equipment and is responsible for housing, operating and maintaining it. The client typically pays on a per-use basis. IaaS is a model for providing cloud services.
- Input/Output Operations Per Second (IOPS): IOPS are a common performance measurement used to benchmark computer storage devices like hard disk drives, solid state drives, and storage area networks.
- instance: A running VM, or a VM in a known state such as suspended, that can be used like a hardware server.
- instance ID: Alternative term for instance UUID.
- instance state: The current state of a guest VM image.
- instance tunnels network: A network segment used for instance traffic tunnels between compute nodes and the network node.
- instance type: Describes the parameters of the various virtual machine images that are available to users; includes parameters such as CPU, storage, and memory. Alternative term for flavor.
- instance type ID: Alternative term for a flavor ID.
- interface: A physical or virtual device that provides connectivity to another device or medium.
- interface ID: Unique ID for a Networking VIF or vNIC in the form of a UUID.
- Internet Control Message Protocol (ICMP): A network protocol used by network devices for control messages. For example, ping uses ICMP to test connectivity.
- Internet Small Computer System Interface (iSCSI): Storage protocol that encapsulates SCSI frames for transport over IP networks. Supported by Compute, Object Storage, and Image service.
- iptables: Used along with arptables and ebtables, iptables create firewalls in Compute. iptables are the tables provided by the Linux kernel firewall (implemented as different Netfilter modules) and the chains and rules it stores. Different kernel modules and programs are currently used for different protocols: iptables applies to IPv4, ip6tables to IPv6, arptables to ARP, and ebtables to Ethernet frames. Requires root privilege to manipulate.
- JavaScript Object Notation (JSON): One of the supported response formats in OpenStack.
- kernel-based VM (KVM): An OpenStack-supported hypervisor. KVM is a full virtualization solution for Linux on x86 hardware containing virtualization extensions (Intel VT or AMD-V), ARM, IBM Power, and IBM zSeries. It consists of a loadable kernel module, that provides the core virtualization infrastructure and a processor specific module.
- keystone: Codename of the Identity service.
- Layer-2 (L2) agent: OpenStack Networking agent that provides layer-2 connectivity for virtual networks.
- Layer-2 network: Term used in the OSI network architecture for the data link layer. The data link layer is responsible for media access control, flow control and detecting and possibly correcting errors that may occur in the physical layer.
- Layer-3 (L3) agent: OpenStack Networking agent that provides layer-3 (routing) services for virtual networks.
- libvirt: Virtualization API library used by OpenStack to interact with many of its supported hypervisors.
- Lightweight Directory Access Protocol (LDAP): An application protocol for accessing and maintaining distributed directory information services over an IP network.
- Linux bridge: Software that enables multiple VMs to share a single physical NIC within Compute.
- Linux Bridge neutron plug-in: Enables a Linux bridge to understand a Networking port, interface attachment, and other abstractions.
- Linux containers (LXC): An OpenStack-supported hypervisor.
- live migration: The ability within Compute to move running virtual machine instances from one host to another with only a small service interruption during switchover.
- load balancer: A load balancer is a logical device that belongs to a cloud account. It is used to distribute workloads between multiple back-end systems or services, based on the criteria defined as part of its configuration.
- load balancing: The process of spreading client requests between two or more nodes to improve performance and availability.
- Load-Balancer-as-a-Service (LBaaS): Enables Networking to distribute incoming requests evenly between designated instances.
- Logical Volume Manager (LVM): Provides a method of allocating space on mass-storage devices that is more flexible than conventional partitioning schemes.
- management API: Alternative term for an admin API.
- management network: A network segment used for administration, not accessible to the public Internet.
- manila: Codename for OpenStack Shared File Systems service.
- message broker: The software package used to provide AMQP messaging capabilities within Compute. Default package is RabbitMQ.
- message bus: The main virtual communication line used by all AMQP messages for inter-cloud communications within Compute.
- message queue: Passes requests from clients to the appropriate workers and returns the output to the client after the job completes.
- metadata service: metadata is useful for accessing instance-specific information from within the instance. The primary purpose of this capability is to apply customizations to the instance during boot time if cloud-init or cloudbase-init is configured on your Linux or Windows image, respectively. However, instance metadata can be accessed at any time after the instance boots by the user or by applications running on the instance.
- Modular Layer 2 (ML2) neutron plug-in: Can concurrently use multiple layer-2 networking technologies, such as 802.1Q and VXLAN, in Networking.
- network: A virtual network that provides connectivity between entities. For example, a collection of virtual ports that share network connectivity. In Networking terminology, a network is always a layer-2 network.
- Network Address Translation (NAT): Process of modifying IP address information while in transit. Supported by Compute and Networking.
- network controller: A Compute daemon that orchestrates the network configuration of nodes, including IP addresses, VLANs, and bridging. Also manages routing for both public and private networks.
- Network File System (NFS): A method for making file systems available over the network. Supported by OpenStack.
- network ID: Unique ID assigned to each network segment within Networking. Same as network UUID.
- network manager: The Compute component that manages various network components, such as firewall rules, IP address allocation, and so on.
- network namespace: Linux kernel feature that provides independent virtual networking instances on a single host with separate routing tables and interfaces. Similar to virtual routing and forwarding (VRF) services on physical network equipment.
- network node: Any compute node that runs the network worker daemon.
- network segment: Represents a virtual, isolated OSI layer-2 subnet in Networking.
- Network Time Protocol (NTP): Method of keeping a clock for a host or node correct via communication with a trusted, accurate time source.
- Networking API (Neutron API): API used to access OpenStack Networking. Provides an extensible architecture to enable custom plug-in creation.
- Networking service (neutron): The OpenStack project which implements services and associated libraries to provide on-demand, scalable, and technology-agnostic network abstraction.
- neutron: Codename for OpenStack Networking service.
- neutron API: An alternative name for Networking API.
- neutron plug-in: Interface within Networking that enables organizations to create custom plug-ins for advanced features, such as QoS, ACLs, or IDS.
- north-south traffic: North-south" traffic is client to server traffic, between the data center and the rest of the network (anything outside the data center). 
- non-persistent volume: Alternative term for an ephemeral volume.
- nova: Codename for OpenStack Compute service.
- Nova API: Alternative term for the Compute API.
- nova-network: A Compute component that manages IP address allocation, firewalls, and other network-related tasks. This is the legacy networking option and an alternative to Networking.
- object: A BLOB of data held by Object Storage; can be in any format.
- object auditor: Opens all objects for an object server and verifies the MD5 hash, size, and metadata for each object.
- object replicator: An Object Storage component that copies an object to remote partitions for fault tolerance.
- object server: An Object Storage component that is responsible for managing objects.
- Object Storage API: API used to access OpenStack Object Storage.
- Object Storage service (swift): The OpenStack core project that provides eventually consistent and redundant storage and retrieval of fixed digital content.
- Ocata: The code name for the fifteenth release of OpenStack. The design summit will take place in Barcelona, Spain. Ocata is a beach north of Barcelona.
- Open Virtualization Format (OVF): Standard for packaging VM images. Supported in OpenStack.
- Open vSwitch: Open vSwitch is a production quality, multilayer virtual switch licensed under the open source Apache 2.0 license. It is designed to enable massive network automation through programmatic extension, while still supporting standard management interfaces and protocols (for example NetFlow, sFlow, SPAN, RSPAN, CLI, LACP, 802.1ag).
- Open vSwitch (OVS) agent: Provides an interface to the underlying Open vSwitch service for the Networking plug-in.
- Open vSwitch neutron plug-in: Provides support for Open vSwitch in Networking.
- OpenStack code name: Each OpenStack release has a code name. Code names ascend in alphabetical order: Austin, Bexar, Cactus, Diablo, Essex, Folsom, Grizzly, Havana, Icehouse, Juno, Kilo, Liberty, Mitaka, Newton, Ocata, Pike, Queens, and Rocky. Code names are cities or counties near where the corresponding OpenStack design summit took place. An exception, called the Waldon exception, is granted to elements of the state flag that sound especially cool. Code names are chosen by popular vote.
- Oslo: Codename for the Common Libraries project.
- persistent volume: Changes to these types of disk volumes are saved.
- Platform-as-a-Service (PaaS): Provides to the consumer an operating system and, often, a language runtime and libraries (collectively, the “platform”) upon which they can run their own application code, without providing any control over the underlying infrastructure. Examples of Platform-as-a-Service providers include Cloud Foundry and OpenShift.
- plug-in: Software component providing the actual implementation for Networking APIs, or for Compute APIs, depending on the context.
- policy service: Component of Identity that provides a rule-management interface and a rule-based authorization engine.
- pool: A logical set of devices, such as web servers, that you group together to receive and process traffic. The load balancing function chooses which member of the pool handles the new requests or connections received on the VIP address. Each VIP has one pool.
- port: A virtual network port within Networking; VIFs / vNICs are connected to a port.
- port UUID: Unique ID for a Networking port.
- private image: An Image service VM image that is only available to specified projects.
- private IP address: An IP address used for management and administration, not available to the public Internet.
- private network: The Network Controller provides virtual networks to enable compute servers to interact with each other and with the public network. All machines must have a public and private network interface. A private network interface can be a flat or VLAN network interface. A flat network interface is controlled by the flat_interface with flat managers. A VLAN network interface is controlled by the vlan_interface option with VLAN managers.
- project: Projects represent the base unit of “ownership” in OpenStack, in that all resources in OpenStack should be owned by a specific project. In OpenStack Identity, a project must be owned by a specific domain. It’s a container for resources like instances, users, networks and so on. It is the method that provides multi tenancy for the openstack environment. You could also see it called  as“tenant”, this is due to the fact that in previous versions of openstack, the two terms were used interchangeably. They are exactly the same thing.
- project ID: Unique ID assigned to each project by the Identity service.
- promiscuous mode: Causes the network interface to pass all traffic it receives to the host rather than passing only the frames addressed to it.
- proxy node: A node that provides the Object Storage proxy service.
- proxy server: Users of Object Storage interact with the service through the proxy server, which in turn looks up the location of the requested data within the ring and returns the results to the user.
- public API: An API endpoint used for both service-to-service communication and end-user interactions.
- public image: An Image service VM image that is available to all projects.
- public IP address: An IP address that is accessible to end-users.
- public key authentication: Authentication method that uses keys rather than passwords.
- public network: The Network Controller provides virtual networks to enable compute servers to interact with each other and with the public network. All machines must have a public and private network interface. The public network interface is controlled by the public_interface option.
- Puppet: An operating system configuration-management tool supported by OpenStack.
- Python: Programming language used extensively in OpenStack.
- RabbitMQ: The default message queue software used by OpenStack.
- RAM filter: The Compute setting that enables or disables RAM overcommitment.
- raw: One of the VM image disk formats supported by Image service; an unstructured disk image.
- reboot: Either a soft or hard reboot of a server. With a soft reboot, the operating system is signaled to restart, which enables a graceful shutdown of all processes. A hard reboot is the equivalent of power cycling the server. The virtualization platform should ensure that the reboot action has completed successfully, even in cases in which the underlying domain/VM is paused or halted/stopped.
- Red Hat Enterprise Linux (RHEL): A Linux distribution that is compatible with OpenStack.
- reference architecture: A recommended architecture for an OpenStack cloud.
- Remote Procedure Call (RPC): The method used by the Compute RabbitMQ for intra-service communications.
- replica count: The number of replicas of the data in an Object Storage ring.
- replication: The process of copying data to a separate physical device for fault tolerance and performance.
- request ID: Unique ID assigned to each request sent to Compute.
- resize: Converts an existing server to a different flavor, which scales the server up or down. The original server is saved to enable rollback if a problem occurs. All resizes must be tested and explicitly confirmed, at which time the original server is removed.
- RESTful: A kind of web service API that uses REST, or Representational State Transfer. REST is the style of architecture for hypermedia systems that is used for the World Wide Web.
- ring: An entity that maps Object Storage data to partitions. A separate ring exists for each service, such as account, object, and container.
- ring builder: Builds and manages rings within Object Storage, assigns partitions to devices, and pushes the configuration to other storage nodes.
- role: A personality that a user assumes to perform a specific set of operations. A role includes a set of rights and privileges. A user assuming that role inherits those rights and privileges.
- Role Based Access Control (RBAC): Provides a predefined list of actions that the user can perform, such as start or stop VMs, reset passwords, and so on. Supported in both Identity and Compute and can be configured using the dashboard.
- role ID: Alphanumeric ID assigned to each Identity service role.
- router: A physical or virtual network device that passes network traffic between different networks.
- scheduler: A Compute component that determines where VM instances should start. Uses modular design to support a variety of scheduler types.
- secure shell (SSH): Open source tool used to access remote hosts through an encrypted communications channel, SSH key injection is supported by Compute.
- security group: A set of network traffic filtering rules that are applied to a Compute instance.
- SELinux: Linux kernel security module that provides the mechanism for supporting access control policies.
- server: Computer that provides explicit services to the client software running on that system, often managing a variety of computer operations. A server is a VM instance in the Compute system. Flavor and image are requisite elements when creating a server.
- server image: Alternative term for a VM image.
- server UUID: Unique ID assigned to each guest VM instance.
- service: An OpenStack service, such as Compute, Object Storage, or Image service. Provides one or more endpoints through which users can access resources and perform operations.
- service catalog: Alternative term for the Identity service catalog.
- service ID: Unique ID assigned to each service that is available in the Identity service catalog.
- service project: Special project that contains all services that are listed in the catalog.
- service registration: An Identity service feature that enables services, such as Compute, to automatically register with the catalog.
- share: A remote, mountable file system in the context of the Shared File Systems service. You can mount a share to, and access a share from, several hosts by several users at a time.
- shared storage: Block storage that is simultaneously accessible by multiple clients, for example, NFS.
- snapshot: A point-in-time copy of an OpenStack storage volume or image. Use storage volume snapshots to back up volumes. Use image snapshots to back up data, or as “gold” images for additional servers.
- Software-defined networking (SDN): Provides an approach for network administrators to manage computer network services through abstraction of lower-level functionality.
- static IP address: Alternative term for a fixed IP address.
- storage back end: The method that a service uses for persistent storage, such as iSCSI, NFS, or local disk.
- storage manager: A XenAPI component that provides a pluggable interface to support a wide variety of persistent storage back ends.
- storage node: An Object Storage node that provides container services, account services, and object services; controls the account databases, container databases, and object storage.
- storage services: Collective name for the Object Storage object services, container services, and account services.
- subnet: Logical subdivision of an IP network.
- swift: Codename for OpenStack Object Storage service.
- swift storage node: A node that runs Object Storage account, container, and object services.
- TempURL: An Object Storage middleware component that enables creation of URLs for temporary object access.
- tenant: A group of users; used to isolate access to Compute resources. An alternative term for a project.
- Tenant API: An API that is accessible to projects.
- tenant endpoint: An Identity service API endpoint that is associated with one or more projects.
- tenant ID: An alternative term for project ID.
- token: An alpha-numeric string of text used to access OpenStack APIs and resources.
- token services: An Identity service component that manages and validates tokens after a user or project has been authenticated.
- user: In OpenStack Identity, entities represent individual API consumers and are owned by a specific domain. In OpenStack Compute, a user can be associated with roles, projects, or both.
- Virtual Central Processing Unit (vCPU): Subdivides physical CPUs. Instances can then use those divisions.
- Virtual Disk Image (VDI): One of the VM image disk formats supported by Image service.
- Virtual Extensible LAN (VXLAN): A network virtualization technology that attempts to reduce the scalability problems associated with large cloud computing deployments. It uses a VLAN-like encapsulation technique to encapsulate Ethernet frames within UDP packets.
- Virtual Hard Disk (VHD): One of the VM image disk formats supported by Image service.
- virtual IP address (VIP): An Internet Protocol (IP) address configured on the load balancer for use by clients connecting to a service that is load balanced. Incoming connections are distributed to back-end nodes based on the configuration of the load balancer.
- virtual machine (VM): An operating system instance that runs on top of a hypervisor. Multiple VMs can run at the same time on the same physical host.
- virtual network: An L2 network segment within Networking.
- Virtual Network Computing (VNC): Open source GUI and CLI tools used for remote console access to VMs. Supported by Compute.
- Virtual Network InterFace (VIF): An interface that is plugged into a port in a Networking network. Typically a virtual network interface belonging to a VM.
- virtual networking: A generic term for virtualization of network functions such as switching, routing, load balancing, and security using a combination of VMs and overlays on physical network infrastructure.
- virtual port: Attachment point where a virtual interface connects to a virtual network.
- virtual private network (VPN): Provided by Compute in the form of cloudpipes, specialized instances that are used to create VPNs on a per-project basis.
- virtual server: Alternative term for a VM or guest.
- virtual switch (vSwitch): Software that runs on a host or node and provides the features and functions of a hardware-based network switch.
- virtual VLAN: Alternative term for a virtual network.
- VirtualBox: An OpenStack-supported hypervisor.
- VLAN network: The Network Controller provides virtual networks to enable compute servers to interact with each other and with the public network. All machines must have a public and private network interface. A VLAN network is a private network interface, which is controlled by the vlan_interface option with VLAN managers.
- VM disk (VMDK): One of the VM image disk formats supported by Image service.
- VM image: Alternative term for an image.
- VMware API: Supports interaction with VMware products in Compute.
- VNC proxy: A Compute component that provides users access to the consoles of their VM instances through VNC or VMRC.
- volume: Disk-based data storage generally represented as an iSCSI target with a file system that supports extended attributes; can be persistent or ephemeral.
- Volume API: Alternative name for the Block Storage API.
- volume driver: Alternative term for a volume plug-in.
- volume ID: Unique ID applied to each storage volume under the Block Storage control.
- volume manager: A Block Storage component that creates, attaches, and detaches persistent storage volumes.
- volume node: A Block Storage node that runs the cinder-volume daemon.
- volume plug-in: Provides support for new and specialized types of back-end storage for the Block Storage volume manager.
- volume worker: A cinder component that interacts with back-end storage to manage the creation and deletion of volumes and the creation of compute volumes, provided by the cinder-volume daemon.
- vSphere: An OpenStack-supported hypervisor.
- weight: Used by Object Storage devices to determine which storage devices are suitable for the job. Devices are weighted by size.
- weighted cost: The sum of each cost used when deciding where to start a new VM instance in Compute.
- weighting: A Compute process that determines the suitability of the VM instances for a job for a particular host. For example, not enough RAM on the host, too many CPUs on the host, and so on.
- worker: A daemon that listens to a queue and carries out tasks in response to messages. For example, the cinder-volume worker manages volume creation and deletion on storage arrays.
- Xen: Xen is a hypervisor using a microkernel design, providing services that allow multiple computer operating systems to execute on the same computer hardware concurrently.
- ZeroMQ: Message queue software supported by OpenStack. An alternative to RabbitMQ. Also spelled 0MQ.
