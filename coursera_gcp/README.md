# Google Cloud Fundamentals:  Core Infrastructure training course. 

# Week 1

## Course Introduction
- In the end of training,
1. Identify the purpose and value of Google Cloud products and services
2. Choose among and use application deployement environments on Goolge Cloud
3. Choose among and use Google Cloud storage options
4. Interact with Google Cloud Services
5. Describe ways in which customers have used Google Cloud

# Week 2

### Cloud Computing Overview
- Cloud computing definitinos by NIST
  1. Customers get computing resources that are on-demand and self-service
  2. Customers get access to those resources over the internet, from anywhere
  3. The provide of those resources allocates them to users out of that pool
  4. Resources are elastic-which means that they're flexible, so customers can be
  5. Customers pay only for what they use, or reserve as they go
- Why cloud?
  1. Colocation
  2. Virtualized data center
  3. Container based architecture
- Every company will be a data company eventually

### IAAS/PAAS
- IaaS offers
  1. Raw Compute
  2. Storage
  3. Network capabilities
  - Customers pay for what they allocate
- PaaS offers
  1. Library
  - Customers pay for what they use
- SaaS
  - Runs on cloud like gmail/google doc

## Google cloud network
1. Highest possible throughput
2. Lowest possible latencies
3. 100+ content caching nodes worldwide
4. High demand content is cached for quicker access
- Google cloud 5 geological locations
  - US, South America, Europe, Asia, Australia
  - Locations -> Regions -> Zones
  - Multi-region applications 
  - 103 zones over 34 Regions

## Environmental Impact
- 2% of world's electricity
- Carbon free by 2030

## Security
- 1Billion+ users every year for goolge cloud/services
- Google infrastructure security
  - Hardware layer
    - Hardware design and provenance
    - Secure boot stack
    - Premises security
  - Service deployment layer
    - Encryption of inter-service communication
  - User identity layer
    - User identity
  - Storage services layer
    - Encryption at rest
  - Internet communication layer
    - Google Front End (GFE)
    - Denial of Services (DoS) protection
  - Operational security layer
    - Intrusion detection
    - Reducing insider risk
    - Employee Universal Second Factor (U2F) use
    - SW development practices

## Open source ecosystems 

## Pricing and Billing 
- Per-second billing
- Can create alerts per Budgets
- Reports/quotas available
  - Rate or allocation quota

# Week 3

## Google Cloud resource hierarchy

|          |                  |
|---------|-------------------|
| Level 4 | Organization node |
| Level 3 | Folder |
| Level 2 |  Project |
| Level 1 | Resoruces (table, bucket, ...) |

- Each project is billed separately
- Project ID cannot be changed
- Project Name is user-created
- Project number is globally unique and used internally
- Resource management tool
  - Gather a list of projects
  - Create/Update/delete projects
  - Reover deleted projects
  - Acces REST API
- Folders can be used to group projects

## Identity and Access Management (IAM)
- Who access to what?
  - IAM: admin can apply policies that define who can do what on which resources
- IAM role
  - Basic: owner, editor, viewier, billing admin
    - Might be too broad for sensitive data sharing
  - Predefined
  - Custom
    - Least privilege model

## Service accounts

## Cloud Identity
- Gmail account -> Google cloud console -> Google Groups
- With Cloud Identity, organizations can define policies and manage their users and groups using the Google Admin console

## Interacting with Google Cloud
- Google cloud console
  - Simple web-based GUI
  - Find resources, check the health, control over management
  - SSH in the browser
- Cloud SDK and Cloud shell
  - gcloud tool: command line interface for Google Cloud products and services
  - gsutil: command line for Cloud storage
  - bq: a command line tool for BigQuery
  - Debian based VM with a persistent 5GB home directory
- APIs
  - Goole APIs Explorer shows what APIs are available
  - Java, Python, PHP, C#, GO, Node.js, Ruby, C++
- Cloud Mobile app
  - Start, stop and use SSH to connect into compute engine instances
  - Stop and start Cloud SQL instances
  - Up to date billing information
  - Alerts and incidence management

## Coursera: Getting started with Google Cloud platform and Qwiklabs

# Week 4

## Virtual Private Cloud networking
- Virtual Private Cloud (VPC)
  - Can run code, store data, host websites, ...
- VPC networks connect Google Cloud resources to each other and to the internet
- Example
  - Over two zones in a single region
    - Can be resilient to disruptions

## Compute engine
- IaaS Solution-> Compute Engine
  - Can create and run VM on Google infrastructure
  - No upfront investment
  - Elastic up to thousands
- VM
  - Can be created using the Google Cloud console, Google Cloud CLI, or Compute Engine API
  - Can run Linux or Windows
  - Bills by second. 1 min minimum.
- Preemptible/spot VM
  - Jobs may be stopped and restarted

## Scaling virtual machines
- Autoscaling: may add/reduce compute engine instances
- Very Large VM: large memory/more cores for data analytics

## Important VPC Compatibilities
- Routing tables: 
  - Built in
  - No router provisioning or managing
  - Forward traffic from one instance to another

## Cloud Load Balancing
- Cloud load balancing
  - Track loads on instances
  - Fully distributed, SW defined, managed service
  - HTTPS, TCP, SSL, UDP traffic
  - Cross-region/multi-region failover

## Cloud DNS and Cloud CDN
- 8.8.8.8: public DNS
- Edge caches: 
- Cloud CDN
  - Low network latency

## Connecting networks to Google VPC
- IPsec VPN protocol
  - uses Cloud Router to make the connection dynamic
  - Not always the best option due to security concerns or bandwidth reliability
- Direct Peering
  - Puts a router in the same public datacenter as a Google point of presence (PoP)
- Carrier Peering
  - Gives a direct access from an on-premisses network through a service provider's network
  - Not covered by a Google Service Level agreement
- Dedicated Interconnect
  - Allows for one or more direct, private connections to Google
- Partner Interconnect
  - Useful if a data center cannot reach a dedicated interconnect

# Week 5

## Google Cloud Storage options
- Type of data
  - Structured data
  - Unstructured data
  - Transactional data
  - Relational data
- Available services
  - Cloud storage
  - Cloud sql
  - Cloud spanner 
  - Firestore
  - Cloud Bigtable

## Cloud Storage
- SW developer/IP operations
- Object storage: not in file or folder structures
  - Binary form
  - Associated metat data
  - Globally unique identifier
- Cloud storage
  - Google's object storage product
  - Scalable globally
  - Website content
  - Disaster recovery
  - Direct download
- Purpose of Cloud storage
  - Binary large-object (BLOB) storage
    - Online content
    - Backup and archiving
    - Storage of intermediate results
- Cloud storage files are organized into buckets
- Immutable objects
  - Cannot be edited
  - But a new version is created
  - Overwriting or versioning is available
- IAM + ACL (Access control list) are necessary
  - Mostly IAM is sufficient
  - For finer control, ACLs are created
- Lifecycle polices for efficiency
  - Define life time of objects
  - Define versioning limits
  - Define due days

## Cloud storage: storage classes and data transfer
- Four primary storage classes in Cloud storage
  - Standard storage: hot data
  - Nearline storage: once per month
  - Coldline storage: once every 90 days
  - Archive storage: once a year, online backup, disaster recovery
- Unlimited storage
- Worldwide accessibility and location
- Low latency and high durability
- GEo redundancy
- No minimum fee
- Pay only for what you use
- Encrypts data on the server side
- Use HTTPS/TLS
- How to transfer data to Cloud storage
  - Online transfer
  - Storage Transfer service (> TB or PB)
  - Transfer appliance (shipping)

## Cloud SQL
- MySQL, PostgreSQL, SQL server
- Mundane tasks
  - Patching/updates
  - Backup
  - Configuring replications
- Doesn't require any SW installation or maintenance
- Scale up to 64 CPU cores, 400GB+ RAM, 30TB of storage
- Supports automatic replication scenarios
- Supports managed backups
- Encrypts customer data on Goolge's internal networks
- Includes a network firewall

## Cloud Spanner
- Scales horizontally
- Strongly consistent
- Speaks SQL
  - Hybrid of SQL and NoSQL

## Firestore 
- Flexible
- Horizontally scalable
- NoSQL cloud database

## Cloud Bigtable
- NoSQL big data database service
- Handles massive workloads
- High throughput
- Can be chosen when:
  - Mor than 1TB of semi-structured or structured data
  - High throughput or rapidly changing
  - NoSQL data

## Comparing storage options
- Cloud storage: image, movies, ...
- Cloud SQL: Up to 64TB, customer order, ...
- Spanner: Petabytes
- Firestore: Terabytes
- Cloud Bigtable: Petabytes
- Big Query is not a solution from storage option but more of different services

# Week 6

## Introduction to containers
- IaaS: share compute resources using VMs
- Containers
  - Independent workload scalability
  - OS and HW absraction layer
  - Scales like PaaS but gives nearly the same flexibility of IaaS
- Configurable system (traditional way)
  - Customizable installation, configuration, and building
  - Larg, slow and costly  
- App engine
  - Access to programming services to write your code
  - Seamless, independent and rapid scaling
  - No fine-turning the underlying architecture to save cost

## Kubernetes
- Kubernetes
  - Open source platform for managing containerized workloads and services
  - A set of APIs to deploy containers on a set of nodes called a cluster
- A pod: smallest unit in Kubernetes

## Google Kubernetes Engine (GKE)
- GKE 
  - Cluster of compute engine
- gcloud command -> GKE -> Kubernetes cluster
  - Deploy and manage applications
  - Perform administration tasks
  - Set policies
  - Monitor workload health

## Hybrid and multi-cloud

## Anthos
- A hybrid and multi-cloud solution
- Framework rests on kubernetes and GKE On-prem
- Provides a rich set of tools for monitoring and maintenance
- Google Cloud <------> On prem data center
  - GKE on-prem provides container services on on-prem datacenter
  - Connects to Google Kubernetes engine through Cloud marketplace

# Week 7

## App Engine
- A fully managed, serverless platform for developing and hosting web applications at scale
- Choose
  - Language
  - Libraries
  - Framework
- No server provisioned
- Built-in Services and APIs
  - NoSQL datastore
  - Memcachne
  - Load balancing
  - Health checks
  - Application logging
  - User authentication API
- Provides SD
  - APIs and libraries
  - Sandbox environment on your local computer
  - Deployment tools
- Use Cloud console's web-based interface
  - Create new applications
  - Configure domain names
  - Change the versioning
  - Examine access and error logs

## App Engine environments
- Standard
  - Containers are configured
  - For many applications
  - Persistent storage with queries, sorting, and transactions
  - Automatic scaling and load balancing
  - Must use specified version of Java, Python, PHP, Go, Node.js, and Ruby
  - Develop web app and test locally
  - Deploy to App engine with SDK
  - App Engine scales and services the app
- Flexible
  - Customize the environment
    - Docker containers, compute engine
  - VM instances are restawrted on a weekly basis
  - Supports
    - Microservices
    - Authorization
    - SQL & NoSQL
    - Traffic splitting
    - Logging
    - Search
    - Versioning
    - Security scanning
    - Memcache
    - CDN

|    | standard  | flexible|
|----|-----------|---------|
|instance startup | seconds | minutes|
| SSH access | No |Yes|
| Write to local disk | No | Yes |
| Suuport for 3rd party binaries| For some | Yes|
| Network access | Via app engine | Yes|
| Pricing model| Pay per instance class | Pay for resource allocation per hour|

## Google Cloud API management tools
- API
  - A clean, well-defined interface
  - Underlying implementation can change
  - Changes to API are made with versions
- 3 API management tools in Google cloud
  - Cloud endpoints
    - Distributed API management system
    - Service proxy
    - Provides an API console, hosting, logging, monitoring, and other featuers
    - Supports App Engine, GKE, and Compute Engine
    - Clients include Android, iOS, and javascript
  - API Gateway
    - Provides secure access to your backend services through a well-defined REST API
    - Clients consume your REST APIs to implement standalone apps
  - Apigee API management 
    - Specific focus on business problems like rate limiting, quotas, and analytics
    - May use a SW service from other companies, not Google Cloud
  
## Cloud run
- A managed compute platform that can run stateless containers
- Serverless, removing the need for insfrastructure management
- Built on Knative, an open API and runtime environment built on Kubernetes
- Can automatically scale up and down from zero almost instantaneously
- Step to deploy
  - Write your code
  - Build and package using container image
  - Deploy to Cloud Run
  - Cloud run then starts your container on demand to handle requests, and ensures that all incoming requests are handled by dynamically adding and removing containers
- Container based workflow vs source-based workflow
  - Cloud run can remove the headache regarding the building containers

# Week 8

## Development in the cloud
- Google Cloud methods for development
  - Cloud source repositories
    - Run own git instances
    - Use a hosted git provider
    - Or cloud source repo using App Engine and Compute Engine
      - Diagnostics tools such as debugger/error reporting
  - Cloud functions
    - Example: uploaded images might converted into different formats, generating thumbnails
    - Lightweight, event-based, async compute solution
    - Create small, single-purpose functions that respond to cloud events
  - Terraform

## Development: Infrastructure as code
- Creating environment
  - Time consuming and labor-intensive
  - Use template (Terraform)
- Create a template file using HashiCorp Configuration Language (HCL) that describes what the components of the environment must look like
- Terraform uses that template to determine the actions needed to create the environment you template describes

# Week 9

## The importance of monitoring
- Monitoring is the foundation of product reliability
  - Reveals what needs urgent attention
  - Shows trends in application usage patterns
  - Helps improve an application experience

## Measuring performance and reliability
- Four golden signals
  - Latency
    - Directly affects the user experience
    - Indicates emerging issues
    - Tied to capacity demands
    - Measures system improvements
    - How to measure?
      - Page load latency
      - Query duration
      - Service response time
      - Transaction duration
      - Time to first response
  - Traffic
    - Indicates the system demand
    - Historic trends for capacity planning
    - Core measure for calculating infrastructure spend
    - Measurement
      - N. of HTTP requests per second
      - N. of requests for static/dynamic content
      - Network IO
      - N. of concurrent sessions
      - N. of transactions per second
      - N. of retrievals per second
      - N. of active requests
      - N. of write/read ops
      - N. of active connections
  - Saturation
    - Indicates how full the service is
    - Focuses on the most constrained resources
    - Tied to degrading performance as capacity is reached
    - Metric
      - % memory utilization
      - % thread pool utilization
      - % cache utilization
      - % disk utilization
      - % CPU utilization
      - Disk quota
      - Memory quota
      - N. of available connections
      - N. of users on the system
  - Error
    - Indicates that something is failing
    - Indicates confugraion/capacity issues
    - Indicates service level objective violations
    - May mean that it is time to send an alert
    - Metric
      - Wrong answers or incorrect content
      - N. of 400/500 HTTP codes
      - N. of failed requests
      - N. of exceptions
      - N. of stack traces
      - Servers that fail liveness checks
      - N. of dropped connections

## Understanding SLIs, SLOs, and SLAs
- SLI (Service Level Indicator)
  - Metrics for service reliability
  - N. of good events / count of all valid events
- SLO (Service Level Objective)
  - Metrics for target reliability
  - Specific
  - Measurable
  - Achievable
  - Relavant
  - Time-bound
- SLA (Service Level Agreement)
  - Commitments made to your customers that your systems and applications will have only a certain amount of down time
  - The minimum levels of service that you promise to provide to your customers
  - What happens when you break that promise
    - Ex: Refund at longer outage

## Integrated observability tools
- Monitoring
- Logging
- Error reporting
- Debugging

## Monitoring tools
- A thousand of streams of metric data in Google cloud
- Cloud monitoring
  - Provides visibility into the performance, uptime, and overall health of cloud powered applications
  - Collects metrics, events, and metadata from projects, logs, services, systems, agents, custom code, and various applications like Cassandra, Nginx, Apach web server, ...
  - Ingests data and generates insights

## Logging tools
- Cloud logging
  - Collect
  - Store
  - Search
  - Analyze
  - Monitor
  - Alert
- Type of logs
  - Cloud audit logs
    - Who did what, where?
    - Admin activity
    - Data access
    - System event
    - Access transparency
  - Agent logs
    - Fluentd agent: Compute Engine, ...
  - Network logs
    - VPC flow
    - Firewall rules
    - NAT gateway
  - Service logs
    - Standard Out/Error

## Error reporting and debugging tools
- Error reporting
  - Counts, analyzes, and aggregates the crashes in your running cloud services
  - Time chart, occurences, affected user acount, ...
- Cloud profiler

# Week 10

## Course summary
