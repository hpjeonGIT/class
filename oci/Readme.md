## Getting Started with Oracle Cloud Infrastructure

- Oracle cloud introduction
    - Applications
        - SaaS
        - Enterprise rosource, supply chain management, Human capital management, advertising and customer experience, industry applications, ISV and custom applications
    - Infrastructure
        - OCI, 40 Regions, 
        - Core: compute, storage, networking
        - Database: Oracle, open source
        - Developer services
        - Containers/functions
        - Application integration
        - ML/AI
        - Data Lakehouse
        - Analytics and BI
        - Security, observability, management, compliance, and cost management/governance

- OCI Architecture
    - Regions
    - Availabilty Domains (AD)
        - Low latency
    - Fault Domains (FD)
        - Distribute cloud resources
    - Choosing a region
        - Location: for low latency
        - Data residency and compliance
        - Service availability: 
    - AD: physical infrastructure not shared
    - FD: grouping of HW/infrastructure within an AD, logical data center within an AD

- Demo: OCI console walkthrough
- Compute Services
    - Compute instances/server
    - cpu/memory/storage
    - 3 types
        - vm:
        - bm: dedicated physical server
        - dedicatd vm host: not sharing?
    - Shapes: template that determines the resource
        - Fixed shape: BM, VM, cannot bu customized
        - Flexible
    - Vertical scaling: change VM. Stop intance and resize
    - Autoscaling: enables large-scale deployment, scale out or in
        - Metric based or schedule-based autoscaling
- Demo: Compute
    - Check Compartment (where resource sits)
    - Image: SW configuration like OS
    - Shape: VM vs BM, AMD or ARM or Intel
- Storage Services:
    - Block Volume: Virtual storag that can be attached to VM
        - always persistent: even vm is gone, still available
    - File storage: shared storage for compute, nfs
        - For multiple instances
        - ideal for containers, big data, analytical workload
    - Object storage: storage for web
        - Internet scale, HP storage
        - unstructured data like logs, videos
- Demo: storage
    - Create Buckets        
        - Edit visibility - visible on web or not
    - Create Block Volume
        - Backup feature (can be deselected)
- Networking service
    - Virtual Cloud Network (VCN)
        - Public subnet
        - Private subnet
    - Gateway
        - Internet
        - NAT
        - Service
    - Dynamic Routing Gateway
- Demo: networking    
- Security services: 
    - Shared security model
        - Oracle for infra
        - customer for securing workload    
            - Provided as security services
            - web application/network firewall, security list, 
            - Identity and Access Management: IAM, MFA, 
            - OS management, Bastion, dedicated host
            - Vault key, certificates
            - Vulnerabilty scanning, security advisor
- Database services
    - Autonomous database: automates backup, patching, upgrading, tuning
    - MySQL database service with heatwave: only on cloud for ML     
- Developer Services
    - Containers: SW packages
    - Microservices:
    - Declarative API: 
    - Immutable infrastructure
    - Service Meshes
- Observability and management services
    - Monitoring: CPU, memory
    - Observability Pillars
        - Metrics
        - Logs
        - Traces
        - Events    
    - Foundational services
        - Monitoring: single pane of glass view with dashboards
        - Logging: service, custom, audit
        - Events: Rules and Actions
- Multicloud
    - Avoid reliance on a single provider
    - OCI-Azure Interconnect
    - Oracle Database Service for Azure
- Hybrid cloud
    - Combination of private and public cloud
    - Dedicated Region Cloud@Customer
        - OCI services running in customer data center    
    - Oracle Cloud VMware solution: in public cloud
    - Exadata Cloud@Customer: cloud autonomous databases, running in customer data center
    - Roving Edge Infrastructure: OCI compute and storage for remote, disconnected scenarios
    
    
Q: from demo, definition of compartment may need description    
- What is CIDR?
    
            

## Oracle cloud infrastructure foundations

2.2. OCI Architecture
- Regions
- Availability domains (AD)
- Fault domains (FD)
    - Logical data center within AD
    - Will not share a singple point of HW
- Choosing a region
    - Location
    - Data residency and compliance
    - Service availability
    
2.3 OCI Distributed Cloud
- Hybrid cloud services
    - Dedicated REgion cloud@customer
        - At customer location.
    - Oracle cloud VMware solution
    - Autonomous DB on ExadataCloud@customer
    - Roving Edge infrastructure
- OCI-Azure interconnect
    - 12 Azure interconnect regions
    - high optimized, secure, and unified cross-cloud experie3nce
    - no charge on egress/ingress of data
- Oracle database services for azure
    - Azure customer can use services from OCI

2.5 Demo: OCI console walk-through
- Dashboard at cloud.oracle.com

3. Expert Tips for Exam

4. Identity and Access Management
4.1 IAM Introduction



