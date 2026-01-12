## The Complete Guide to AI Infrastructure: Zero to Hero

## Section 1: Introduciton to the complete guide to AI infrastructure: Zero to Here

### 1. Introduction to The Complete Guide to AI Infrastructure: Zero to Hero

## Section 2: Week 1: Introduction to AI Infrastructure

### 2. 1. What Is AI Infrastructure? The Engine Behind AI
- AI infrastructure as 3 critical layers
  - HW layer
  - SW layer
  - Operational layer
- Why AI needs specialized infrastructure
  - Massive compute requirements
  - Parallel processing architecture
  - Extreme data scale
  - Low-latency inference
- Key components of AI infrastructure
  - Compute: CPU, TPU, NPU, ...
  - Storage: vector database
  - Networking
  - Orchestration: kubernetes
  - Monitorying & Security

### 3. 2. CPUs vs GPUs vs TPUs – Computing Power for AI
- Your HW selection impacts:
  - Performance and Training speed
  - Cost efficiency
  - Scalability
- CPU: orchestration, small inference, preprocessing
- GPU: parallel processing powerhouses
- TPU: optimized for Tensorflow and JAX framework
- How to choose right HW
  - Evaluate your worklad
  - Consider hybrid architectures
  - Factor in total cost
  
### 4. 3. Training vs Inference – Two Faces of AI Workloads
- AI workloads
  - Training: intensive computation resources, data-hungry, accuracy-focused
  - Inference: low latency and high availability, deployment flexibility such as GPU, edge devices, mobile phones
- Infrastructure differences
  - Training
    - Distributed training clusters with high-speed interconnects
    - Petabyotle scale storage
    - Tools: PyTorch DDP, Horovod, DeepSpeed
  - Inference
    - Optimized serving systems with autoscaling capabilities
    - API gateways and load balancers
    - Tools: Triton inference server, TorchServe, TensorRT
    
### 5. 4. AI Infrastructure Layers – Hardware, Software, Ops
- Layered view of AI infrastructure
  - HW
    - Compute
    - Storage
    - Networking: RDMA capable interconnects, specialied load balancers
  - SW
    - Frameworks: PyTorch, TensorFlow, JAX
    - Acceleration libraries: CUDA, cuDNN, NCCL, oneDNN
    - Containers & orchestration: docker, kubernetes
    - MLops tools: MLFlow, Kubeflow, Weights & Biases
  - Operations
    - CI/CD pipelines: automated testing and deployment workflows
    - Monitoring & observability: Prometheus, Grafana
    - Security and compliance

### 6. 5. Case Studies: Infrastructure Behind ChatGPT, DALL·E
- ChatGPT infrastructure
  - Trillions of tokens
  - Thousands of A100/H100 GPUs with NVlink interconnects
  - DeepSpeed and Megatron-LM for model parallelism
  - Triton inference server with API gateways 
- DALL-E infrastructure
  - Multimodal complexity
  - Storage demands for petabytes
  - Compute requirements
  - Inference architecture: cloud-based API with autoscaling 
- Lessons for AI Engineers
  - Distributed systems expertise is essential
  - Model serving must balance latency and throughput
  - Efficiency improvements save mllions in cost

### 7. 6. Industry Landscape: Cloud Providers & AI Chips
- The AI infrastructure ecosystem
  - Cloud providers: deliver on-demand compute, scalability, and managed AI services
    - AWS, Google Cloud, MS Azure
    - CoreWeave, Lambda Labs
  - Chip manufacturers: supply raw computational horsepower
    - Nvidia, Google TPUs, AMD
    - Cerebras, Graphcore, SambaNova
- Required skills for AI engineers
  - Multi-environment expertise
  - Balance lock-in vs performance
  - Cost-performance optimization

### 8. 7. Lab – Spin Up Your First AI VM

## Section 3: Week 2: Linux Foundations for AI Engineers

### 9. 8. Why Linux Dominates AI Infrastructure
- The language of AI infrastructure

### 10. 9. Navigating the Linux Shell – Bash Basics

### 11. 10. Filesystems, Directories, and Permissions

### 12. 11. Package Managers – apt, yum, pip

### 13. 12. Process Management and Monitoring
- Tools
  - ps aux
  - top/htop
  - pgrep
  - iotop
  - iftop
- Logging and system monitoring tools
  - dmesg: kernel/driver message, HW error
  - journalctl: systemd service logs and evetns
  - /var/log/
  - Prometheus + Grafana

### 14. 13. Scripting for Automation in AI Workflows

### 15. 14. Lab – Set Up Ubuntu for AI Development

## Section 4: Week 3: Cloud Infrastructure Basics

### 16. 15. Introduction to Cloud for AI Workloads
- Why cloud?
  - On-demand resources
  - Elastic scaling
  - Pay-as-you-go
- AI workloads in the cloud
  - Training
  - Inference
  - Data pipelines
  - MLOps: end-to-end automation for model deployment, monitoring and maintenance
- Managed AI Services
  - AWS sagemaker
  - Google Vertex AI
  - Azure ML studio
  - HuggingFace on Cloud
- Challenges of Cloud AI
  - Cost management
  - Vendor lock-in
  - Network cost: larget dataset tranferring over multiple regions
  - Skills gap
- Cost optimization strategies
  - Spot instance
  - Storage tiering
  - Usage monitoring
  - Autoscaling
  - Workload balancing
- Cloud & edge AI integration
  - Key applications
    - Autonomous vehicles
    - IoT sensor network
    - AR
    - Retail computer vision
    - Industrial quality control
  - Benefit of edge deployment
    - Ultra low latency
    - Works in disconnected environment
    - Reduces bandwidth costs
    - Preserves data privacy
- Essential skills for cloud AI
  - Linux fundamentals
  - Virtualization
  - Cloud storage and networking
  - MLOps tools
  - Cost management
  - Security best practices    

### 17. 16. Compute Instances – Choosing CPU vs GPU
- CPU economics
  - Lower hourly cost ($0.05-0.50/hr)
  - Good for long-running, low intensity jobs
  - Widely available
- GPU economics
  - Higher hourly cost ($1-40/hr)
  - Spot/preemptible instances cut costs by 70-90%
- CPU responsibilities
  - Data ingestion and preprocessing
  - Request routing and load balancing
  - Workflow orchestration
  - System monitoring and logging
- GPU reponsibilities
  - Model training
  - High-throughput inference
  - Matrix/tensor operations
  - Specialized kernel execution

### 18. 17. Networking Basics – VPCs, Firewalls, Load Balancers
- Virtual Private Cloud
  - IP address management
  - Network topology control
  - Security boundaries
  - Cloud-provider integration
- Subnets: logical division within your VPC
  - Public
    - Connection to the internet gateway
    - Host public-facing ML APIs
    - Run inference endpoints
  - Private
    - Host training clusters
    - Store sensitive datasets
  - This separation creates a **defense-in-depth strategy** for AI infrastructure
- Firewalls: First line of defense for AI workloads
  - Rule based control
  - ML infrastructure protection
  - Multi-tenant isolation
- Security groups vs network ACLs
  - Security grouips
    - Instance level fireall protectin
    - Stateful
    - Support allow rules only
    - Applied to specific instances    
  - Network ACLs (Access Control Lists)
    - Subnet-level 
    - Stateless
    - Support both allow and deny rules
    - Applied to all instances in subnet
- Load balancers
  - Horizontal scaling
  - High availability
  - Traffic management
  - Inference endpoint scaling
- Types of load balancers
  - By protocol layer
    - Layer 4 (transport)
      - Routes based on IP address and port
      - Handles TCP/UDP traffic
      - Lower overhead, higher throughput
      - Great for raw traffic distribution
    - Layer 7 (application)
      - Routes based on HTTP/HTTPS headers
      - Direct traffic by URL path
      - Supports content based routing
      - Ideal for complex API endpoints
  - By network positions      
    - Internal Load balancers
      - Distribute within private subnets
      - Not accessible from internet
      - For microservices communication
    - External load balancers
      - Public-facing endpoints
      - Distribute internet traffic
      - Protect from DDoS attacks
      - Front public AI APIs
- Special network
  - Training workloads
    - Ultra-fast internal networks like infiniband, RDMA, or NVLink
  - Inference workloads
    - Need public endpoints with load balancers
- Networking best practices for AI infrastructure
  - Restrict administrator access
  - Protect databases and storage
  - Encrypt all traffic
  - Monitor network traffic
  - Infrastructure as code

### 19. 18. Cloud Storage – Object, Block, and File Systems
- Object storage
  - Highly scalable storage for unstructured data
  - Cost-effiective for large datasets
    - Images, video, text, ...
    - Data lake implementation
    - Model weight preservation
    - Backup and archival
  - API-driven access patterns
  - AWS S3, Google cloud storage, Azure Blob
- Block storage
  - Fast disk volumes for compute instances
    - Local drive    
  - Low-latency performance
  - Instance attached resources
  - AWS Elastic block store, Google persistent disk, Azure managed disks
- File storage
  - Shared access via filesystem protocols
    - NFS, SMB
  - Multi-server mounting capability
  - Team collaboration-friendly
  - AWS EFS, Azure Files, Google Cloud Filestore

### 20. 19. Hands-On with AWS EC2 for AI

### 21. 20. Hands-On with Google Cloud GPU Instances

### 22. 21. Lab – Compare Cost & Performance Across Clouds

## Section 5: Week 4: Containerization Foundations

### 23. 22. Why Containers Are Critical for AI
-The role of containers in AI
  - Eliminate inconsistency
  - Ensure that AI model run across the same environment
- What containers solve:
  - Dependency hell
  - Environment drift
  - Scaling difficulty
  - Portabilty issues
  - Reproducibility challeneges
- How containers work
  - Namespaces: process isolation
   -Cgroups: resource allocatin
- Terminology
  - Images: immutable blueprints/templates
  - Containers: Running instances of images
  - Docker: Container runtime
  - Kubernetes: Orchestrates containers at scale

| | VM | containers |
|--|---|---|
| Resources| Full OS emulation | Shared kernel architecture |
| Startup performance| Heavier resource footprint | Lightweight |
| Isolation level | Strong isolation | Optimal for scaling AI workloads |
| Deployment speed| Slow startup | Fast boot|

- Containers in the AI workflow
  - Development
  - Training
  - Collaboration
  - Deployment
  - Research
- GPU support in containers
  - Enables GPU passthrough to containerized applications
  - Eliminate complex CUDA/driver compatibility issues  
- Best practices for AI containers
  - Use lightweight base images
  - Pin dependencies exactly: exact version like tensorflow==2.12.0 not tensorflow >=2.12.0
  - Optimize image size
  - Implement private registries
  - Automate container builds: implement CI/CD pipelines
    
### 24. 23. Docker Basics – Images and Containers
- Docker image: 
  - A complete package containing OS libraries, dependecies, configuration, and code
  - Immutability once built, ensuring consistency across environment
- Docker container
  - Running instance of a docker image
  - Can be started, stopped, paused, or deleted as needed
  - Multiple containers can run simultaneously on one machine, each with its own isolated environment
- Docker workflow
  - Build
  - Run
  - Share
  - Deploy
- Docker commands
  - docker images # list images on your system
  - docker ps -a # list all containers
  - docker build -t myimage
  - docker run -it myimage
  - docker exec -it container_id bash
- Dockerfile: build your own images
  - FROM:
  - RUN:
  - COPY
  - WORKDIR
  - ENV
  - EXPOSE
  - CMD
- Best practices for AI containers
  - Use official base images
  - Keep images small
  - Pin specific versions
  - Don't run as root
  - Automate builds

### 25. 24. Building a Container for an AI App
- Why containerize an AI app?
  - Portability
  - Deployment simplicity
  - Reproducibility
  - API Foundation
- Demo AI App
  - Image classifier
  - Pretrained ResNet18 CNN model
  - FastAPI server  
- Project structure
  - ap-app/app.py # FastAPI code
    - Asynchronous handler for better performance
    - Simple JSON response format
  - ap-app/model.pt # pytorch file
  - model/requirements.txt
```bash
fastpi==0.103.0
uvicorn==0.23.2
torch==2.1.0
torchvision==0.16.0
pillow==10.0.0
```
  - model/Dockerfile   
```bash
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt
RUN pip install -r requirements.txt
COPY ..
CMD ["uvicorn", "app.app", "--host", "0.0.0.0","--port", "8000"]
```
- Build the image: `docker build -t ai-app:latest .`
- Run the container: `docker run -it -p 8000:8000 ai-app:latest`
  - Now accessible at `http://localhost:8000/docs`

### 26. 25. Networking and Volumes in Docker
- Volumes: persistent dat stores for containers, created, and managed by docker
  - `docker volume create`
  - A volume's contents exist outside the lifecycle of a given container - data in the volume is persistent
    - Writable layer in the docker is destroyed with it
- Why networking & volumes matter
  - Container isolation need connection
  - Connection needs connection to users, services, and each other
  - Data persistence is guaranteed by a volume
- Docker networking basics
  - Private IPs: every container gets its own IP address
  - Default bridge: default network is **bridge**
  - Nework access
- Port mapping
  - Essential for inference endpoints
  - Without mapping, services are only accessible internally
- Docker Volme Basics
  - Without volume, all data is lost when containers are deleted or restarted
  - How volume solve this
    - Persis data on host sysmte
    - Mount the special syntax
      - `docker run -v /host/data:/container/data ai-app`
- Named volumes
  - Perfect for shared datasets or model checkpoints, preventing duplicated data and improving portability
  - Steps
      1. Create named volume: `docker volume create model-store`
      2. Mount in container: `docker run -v model-store:/app/models ai-app`
      3. Share across containers: `docker run -v model-store:/mkodels inference-api`
          - Same volume used in multple containers
          - No repeated downloads
          - May use NFS for sharing among users
          - May mount RDBMs default data directory
- Bind mounts
  - Developer-frinedly mounts
  - Map host directory directly to container: `docker run -v $(pws)/notebooks:/workspace jupyter`            
  - Syncs local code changes instantly
  - No rebuild needed for code tweaks
  - But can overwrite container files

### 27. 26. Docker Compose for Multi-Service AI Systems
- One YAML file
- AI systems need multiple services
  - Model server
  - API Gateway
  - Database
  - Worker
- Docker compose basics
  - Define services, networks, and volumes
  - Version control friendly
  - Reproducible ML pipelines
  - Commands
```
docker-compose up
docker-comkpose up d
docker-compose down
```
- Example compose file
```yaml
version:"3.9"
services:
  api:
    build: /.api
    ports:
    - "8000:8000"
    depends_on:
    - model
model:
  image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8
  volumes:
  - ./models:/models
```
- Compose facilitates network configuration
- Best practice
  - Environment variables: use `.env` file for secretes and configuration (API_KEY)
  - Version pinning
  - Module YAML files
  - Health checks
  - Documentation
  - Resource limits

### 28. 27. Best Practices: Lightweight AI Containers
- Why lightweight containers:
  - Faster builds and deployments
  - Cost efficiency
  - Enhanced security
  - Efficient scaling
- Common problems with heavy AI containers
  - Bloated images (10+GB)  
  - Registry bottlenecks
  - Performance issues
  - Security vulnerabilities
  - Resource waste
- Choose minimal base images
  - Splim python
  - Ultra light options
  - Official ML images
- Multi-stage builds
  - Can reduce size significantly
- Minimize dependencies
  - Dependency optimization
  - Regular maintenance
- Layer cahcing for faster builds
  - Install dependencies before copying code to leverage Docker's layer caching
  - Proper ordering of layers in a dockerfile ensures that dependencies are cached, speeding up iterative builds
- Best practice in the security of lightweight containers
  - Remove root privileges
  - Regular updates: security patches
  - Vulnerability scanning: `docker scan myimage:latest`
- Optimize for GPU containers
  - Use Nvidia offical images
  - Avoid full CUDA toolkit
  - Match CUDA versions

### 29. 28. Lab – Containerize a PyTorch Model

## Section 6: Week 5: Kubernetes Fundamentals

### 30. 29. What Is Kubernetes and Why AI Needs It
- Why Kubernetes?
  - Distributed computing: coordination across clusters of CPUs/GPUs
  - Operational complexity: deployment and updates at scale
  - Production reliability: high availability and fault tolerance
- Core benefits of kubernetes
  - Scalability
  - Resilience
  - Portability
  - Efficiency
  - Automation
- Kubernetes architecture
  - Control plane
    - API Server: frontend for the control plane
    - Scheduler: assigns pods to nodes
    - Controller manager: maintains desired state
    - etcd: distributed key-value store for all cluster data
  - Worker nodes
    - Multiple nodes running node components
  - Node components
    - Kubelet: Ensures containers are running in pods
    - Kube-proxy: maintains network rules
    - Container Runtime: runs the containers
    - Pods: Groups of one or more containers
  - Pods & containers
    - Pods containing one or more containers
- Kubernetes + MLOps tools
  - Kubeflow
  - MLflow
  - Nvidia Triton

### 31. 30. Pods, Nodes, and Clusters Explained
- Pod
  - Smallest deployable unit
  - Container encapsulation
  - Shared resources
  - Ephemeral by design: disposable and recreated/restarted anytime
- Pods in AI workflow
  - Training pod
  - Inference pod
  - Data pod
  - Monitoring pod
- Node
  - A physical server or VM
  - Each node runs a Kubelet agent
  - contributes their CPU/GPU, memory, storage to the cluster
  - Execution layer
  - GPU nodes, CPU nodes, edge nodes
- Cluster
  - A collection of nodes managed by a control plane
  - Provides a unified pool of compute, storage, and networking resources
  - Handles scaling, failover, and resource allocation
- Clusters in AI workflows
  - Training clusters
  - Hybrid clusters  
  - Scalable deployments
  - Multi-region clusters
- How pods, nodes, and clusters interact
  - Pod: containerized ML workload - training, inference service
  - Node: Machine that provides compute resources to run pods
  - Cluster: Collection of nodes managed as a single entity
  - Scheduler matches pod requirements to node capabilities across the cluster, optimizing resource utilization

### 32. 31. Deployments and Services for AI Apps
- Why deployments and services?
  - Pods are ephemeral and can fail anytime
  - Deployments manage pod lifecycles
  - Services provide stable nework endpoints, making your AI APIs consistently accessible
- Deployment
  - Kubernetes object
  - Manages pod lifecycles
  - Ensures that the desired number of replicas are running
  - Handles updates and rollbacks with zero downtime
  - replicas in compose file
- Service
  - A Kubernetes abstraction
  - Provides a stable network endpoint
  - Load-balances traffic across all available replicas
  -  Exposes AI inference APIs to users and connect microservices within ML pipeline
- Types of services
  - ClusterIP: ideal for internal AI microservices
  - NodePort: Good for development and testing
  - LoadBalancer: For production AI APIs
  - Headless: Pod-to-pod communication. Useful for stateful workloads
- Deployments + Services together
  - Deployment ensures that the right number of pods are always running
  - Service provides a stable endpoint for accessing those pods
  - Together, they enable production-grade AI inference APIs with high availability

### 33. 32. ConfigMaps, Secrets, and Volumes
- Three Pillars
  - ConfigMaps for configuration
  - Secrets for credentials
  - Volumes for data persistence  
- ConfigMap
  - A kubernetes object that stores non-sensitive configuration data as key-value pairs
  - Decouple configuration from container images
  - Inject settings as environment variables or files
  - Update configuration without rebuilding containers
  - Share common settings across multiple pods
- For AI workloads, ConfigMaps typically store:
  - Batch size and learning rates
  - Model paths and feature toggles
  - Logging levels and monitoring settings
- ConfigMap example
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-config
data:
 BATCH_SIZE: "64"
 MODEL_PATH: "/models/resnet.pt"
 LOG_LEVEL: "INFO"
 FEATURE_GATE_ADVANCED_METRICS: "true"
```
- Application into a Pod:
```yaml
containers:
- name: inference-service
  image: ai-company/inference:v1.2
  envFrom:
  - configMapRef:
    name: ai-config
```
- Secret:
  - Encrypted key-value stores like passwords, tokens, and keys
  - Base64 encoded, encrypted at rest, and only distributed to nodes that need them
  - Keep credentials out of container images by using files within volumes or mount as environment variables
  - AI-specific use cases: API_KEYS, dBase credentials, access tokens
- Volume
  - Persistent storage
  - Storage types:  block storage, NFS, or specialized
  - Access modes: ReadWriteOnce (RWO) ReadOnlyMany (ROX), or ReadWriteMany(RWX) depending on workload needs
- Best practices
  - Security
    - Never hardcode credentials in images or ConfigMaps
    - Rotate Secrets regularly, especially in production
  - Organization
    - Use namespace to separate development/staging/production configs
    - Label resources clearly for tracking and auditing
    - Document ConfigMap options for ML engineers
    - Version control your config (not secrets) with code
  - Performance
    - Use SSD-backed storage for model serving
    - Consider ReadWriteMany volumes for shared datasets
    - Be aware of volume mount performance impacts
    - Use cloud managed storage for very large datasets

### 34. 33. Horizontal Pod Autoscaling for AI Inference
- Why autoscaling matters for AI
  - Variable traffic
  - Cost optimization
  - Performance impact
- Kubernetes Horizontal Pod Autoscaler (HPA) provides the dynamic scaling mechanism needed to balance those competing concerns
- HPA does:
  - Automatically adjusts replica count of pods based on observed metrics
  - Watches CPU, memory or custom metrics
  - Maintains workload responsiveness
  - Runs continuously in the background
- How HPA works
  - Metrics collection
  - Threshold comparison
  - Scale up decision
  - Scale down decision
- minReplicas/maxReplicas are defined in the yaml
- Custom metrics for AI
  - GPU utilization
  - Request latency: scale when p95/p99 inference times exceed thresholds
    - p95: 5% of requests are slower than a criterion
    - p88: 1% of requests are slower than a criterion
  - QPS (Queries per second)
- Autoscaling in cloud AI services
  - AWS EKS
  - GCP GKE
  - Azure AKS
- Limitations of HPA
  - Scaling delay
  - Not for training
  - Stateless only
  - GPU complexity
  - Tuning required
- Best practices
  - Redundancy: minReplicas >1
  - Monitoring
  - Readiness Probes
  - Multi-level scaling

### 35. 34. Helm Charts for Simplified AI Deployments
- Helm: package manager for Kubernetes
  - Simplifies instgallation and updates
  - Enables templating and reuse
  - Versioning and rollbacks
- Raw yaml vs Helm
  - Raw YAML
    - Verbose and repetitive
    - Hard to maintain at scale
    - No built-in versioning
  - Helm
    - Parameterized, DRY approach
    - Reuse configs across environments
    - Built-in rollback for failed releases
    - Makes AI infrastructure modular

### 36. 35. Lab – Deploy a Model on Minikube

## Section 7: Week 6: Data storage for AI

### 37. 36. Data Lakes vs Data Warehouses vs Feature Stores
- Storage paradigms in AI
  - Data availability
  - Data quality
  - Purpose-built systems
- Data lake
  - Stores enormous volumes of raw data in its native format until needed
  - Stores unstructured and semi-structured data (logs, image, txt, video)
  - Implements schema-on-read
  - Utilizes cheap, scalable object storage (S3, GCS, Azure Data Lake)
  - Pros
    - High scalable
    - Format flexibility
    - AI-friendly
  - Cons
    - Query inefficient
    - Data swamp risk
    - Expertise required
- Data warehouse
  - A centralized repository for structured, processed data optimized for anlysis and business intelligence
  - Implements schema-on-write
  - Optimized for comples SQL queries
  - Snowflake, BigQuery, Redshift, and Synapse
  - Suited for business reporting and analytics
  - Pros
    - Fast analytical quereis with optimized engines
    - High data quality through validation and transformation
    - Mature governance and security capabilities
  - Cons
    - Expensive for storing large volumes of raw data
    - Poor fit for unstructured AI data - images, audios, text
    - Limited flexibility with strict schema requirements
    - Processing overhead for data loading and transformation
- Feature store
  - A specialized system for managing and serving ML features
    - Bridges the gap b/w data engineering and ML systems
    - ESuires feature consistency b/w training and serving
    - Manages feature versioning, metadata, and lineage
    - Feast, Tecton, AWS SageMaker Featur Store, Databricks Feature Store
    - Pros:
      - Feature consistency
      - Accelerated development
      - Prevents data leakage
      - Feature reusability
    - Cons:
      - Infrastructure complexity
      - Integration challenges
      - Learning curve
      - Operational overhead: requires dedicated maintenance

### 38. 37. Object Storage with AWS S3 and GCP GCS
- Why object storage for AI?
  - Massive scale
  - Cost effective
  - Parallel Access
- What is object storage?
  - Object structure
    - Each object contains data + metadata + unique identifier
    - No traditional file system hierarchy
  - Access Methods
    - RESTful API (no file path)
    - HTTP/HTTPS operations (GET, PUT, DELETE)
  - Key providers
    - AWS S3, Google cloud storage, Azure blob storage
- AWS S3 overview
  - Data organized in buckets
  - Rich feature set: versioning, replication, lifecycle rules
  - Native integration with AWS AI/ML services
  - Access methods: CLI, SDK, REST API
- GCP GCS overview
  - Global namespace with region/multi-regional options
  - Strong consistency model
  - Deep integration with BigQuery for AI analytics
  - Optimized for Google Cloud AI platform
  - Access via gsutil CLI, client libraries, or REST API
    - Perfect for TensorFlow and JAX workloads
- Tiered storage
  - Standard/Host storage: Highest cost per GB, no retrieval fees. Ideal for active AI training datasets
  - Infrequent acces/nearline: retrieval within seconds
  - Archive/coldline: may take minutes to ours
- Security and Access Control
  - Identity and access management (IAM)
  - Bucket policies and ACLS
  - Encryption
- AI use cases for object storage
  - Raw datasets
  - Model checkpoints
  - Inference artifacts
  - Distributed training

### 39. 38. Relational vs NoSQL Databases in AI
- Why databases matter for AI
  - Foundation of AI infrastructure: essential for building robust data pipelines and features stores
  - Performance bottleneck: database choice impacts scalability, latency, and consistency
- Relational databases: the structured approach
  - Table structure
  - Schema definition
  - SQL Queries
  - Ex: PostgreSQL, MySQL, Oracle
- Strength of Relational Databases for AI
  - Structured data excellence
  - SQL ecosystem
  - Transactional integrity
  - Feature engineering
- Limitations of relational databases for AI
  - Scaling challenges
  - Schema rigidity
  - Unstructured data struggles
  - Real-time limitations
- NoSQL databases: the flexible alternative
  - Key-value: Redis, DynamoDB
  - Document: MongoDB, Couchbase
  - Columnar: Cassandra, Bigtable
  - Graph: Neo4j, Amazon Neptune
- Strenghts of NoSQL databases for AI
  - Massive scale
  - Schema flexibility
  - Real-time performance
  - Specialized storage
- Limitations of NoSQL databases for AI
  - Consistency tradeoffs
  - Query complexity
  - API fragmentation
- Hybrid approaches for AI pipelines
  - Relational databases: curated historical data and offline feature engineering
  - NoSQL databases: real-time feature serving and high-volume inference

### 40. 39. Streaming Data with Kafka & Pub/Sub
- Handling real-time data
- Why Streaming data for AI?
  - Fraud detection systems
  - Recommendation engine
  - IoT netowrks generating continuous sensor data
- Traditional batch processing creates unacceptable latency for time-sensitive AI applications
- Apache Kafka
  - Distributed platform: millions of events per second with fault tolerance and horizontal scalability
  - Topic based architecture
  - Rich ecosystem
- Kakfa in AI workflows
  - IoT and sensory data
  - User interaction
  - Log analytics
  - Feature pipelines: continuously update feature stores with fresh data to improve model accuracy and relevance
- Google Pub/Sub
  - Fully managed messaging and streaming service
  - No infrastructure required
  - Flexible delivery modes (push or pull)
  - Native integration with BigQuery, Dataflow, and Vertex AI
  - Global scale with pay-per-use pricing model
  - Built-in security and compliance controls
- Kafka architecture
  - Producers: applications that send data to Kafka topics
  - Brokers: servers that sotre data in partitioned topics
  - Consumers: Applications that subscribe to and process events
  - Zookeeper (legacy): manages cluster coordination and metadata
  - Kafka Connect: integrates with databases, cloud services, and ML pipelines
- Pub/Sub architecture
  - Publisher: application that creates and sends messages to a topic
  - Topic: named resource that represents a feed of messages
  - Subscription: named resource representing the stream of messages from a topic to a specific consumer
  - Subscriber: application that receives and processes messages from a subscription
- Best practices for AI streaming
  - Optimize partitioning
  - Monitor consumer lag
  - Implement security controls
  - Use schema management

### 41. 40. Scaling Storage for AI Training Datasets
- Why storage scaling matters in AI
  - AI training requires terabytes to petabytes of data
  - Poor storage architecture leads to:
    - Slow training epochs
    - Wasted GPU computation hours
    - Higher operational costs
- Storage bottleneck in AI training
  - Slow I/O throughput
  - Network limitations: bandwidth constraints
  - Dataset duplication
  - Inefficient caching
- Storage scaling strategies
  - Scale up
    - Faster SSDs/NVME
    - Optimized filesystems
    - Vertical growth
  - Scale out
    - Distributed storage
    - Horizontal expansion
    - Multi-node access
  - Cloud object storage
    - Near-infinite capacity
    - Elasticity
    - Pay-as-you-go
  - Caching + sharding
    - Reduced IO strain
    - Parallel access
    - Locality benefits
- Distributed file systems for AI workloads
  - HDFS
  - CephFS/ClusterFS: popular in enterprise AI clusters
  - Lustre: HPC optimized parallel file system. Complex deployment
- Cloud object storage for AI
  - Key players
    - AWS S3
    - GCS
    - Azure blob storage
  - Benefits for AI workloads
    - Virtually infinite 
    - Cost-effective for raw dataset
    - API-driven access from training jobs
    - Integrated data versioning and lifecycle policies
- High-performance storage for training
  - NVMe/SSDs
  - Parallel File systems
  - Cloud premium tiers
- Caching and preprocessing strategies
  - Techniques for optimizing data flow
    - Local caching
    - Pre-sharding: split large datasets for parallel reads across nodes
    - Format optimization: Use binary formats optimized fro ML workloads
  - Key tooling
    - Webdataset: efficient loading of tar files for PyTorch
    - TFRecords: optimized binary format for Tensorflow
    - Hugging Face Datasets: memory-mapped access for transformers
- Scaling best practices
  - Implement hybrid tiered storage
  - Monitor IO throughput
  - Deduplicate datasets
  - Optimize dataset formats
  - Automate with orchestration

### 42. 41. Secure Data Access and Encryption
- Why security matters in AI infrastructure
  - High-value targets
  - Significant consequences
  - Regulatory mandates
- Secure data access basics
  - The foundation: least privileage
  - Configure Role-based Access Control (RBAC)
  - Implement Identity and Access Management (IAM) polices
  - Maintain comprehensive audit logs for all access attempts
  - Rotate access keys quarterly 
- Encryption at rest
  - Cloud provider solutions
    - AWS KMS
    - GCP CMEK
    - Azure Key Vault with Managed HSM
  - Infrastructure level protection
    - Linux Unified Key Setup (LUKS) for disk encryption
    - eCrptfs for file-level
    - TRansparent data encryption in databases
  - ML-specific assets
    - Always encrypt:
      - Model weights and parameters
      - Training checkpoints
      - Experiment log containing data samples
      - Hyperparameter configurations
- Encryption in transit
  - TLS/SSL everywhere
  - Secure SSH tunnels
  - VPC service controls
- Secrets management
  - The cardinal rule: no hardcode credentials in code, config files, or container images
  - Secure Storage solutions
  - Best practices
    - Implement security hygiene:
      - Rotate secrets on fixed schedules (30-90days)
      - Use temporary credentials
      - Maintain audit logs of all secret access
- Secure data sharing
  - Time limited access
  - Granular permissions
  - Data anonymization
- Security best practices
  - Zero trust architecture: never trust, always verify
  - Defense in depth
  - Security culture

### 43. 42. Lab – Build a Data Ingestion Pipeline

## Section 8: Week 7: GPU Hardware Deep Dive

### 44. 43. Anatomy of a GPU for AI
- CPU architecture
  - Latency-focused design
  - Optimized for sequential tasks
- GPU architecture
  - Throughput-optimized design
  - Designed for massive parallelism
- GPU cores and Streaming multiprocessors
  - Each SM executes multiple threads concurrently
  - Warps execute in lockstep
  - Designed for SIMD
  - Optimized for matrix/vector operations that power AI
- Tensor cores
  - Accelerates FP16, BF16, and INT8 matrix operations
- GPU memory hierarchy
  - Registers
  - Shared memory
  - L1/L2 cache
  - Global device memory (HBM/GDDR)
- High-bandwidth memory (HBM)
  - Memory throughput is often the limiting factor in AI workloads
- GPU interconnects
  - PCIe
  - NVLink/NVSwitch
  - Impact on AI

### 45. 44. CUDA Basics and GPU Programming
- CUDA programming model
  - Threads: smallest unit of execution
  - Blocks: Groups of threads that share resources
  - Grids: Collections of blocks tat execute a kernel
- CUDA memory hierarchy
  - Registers: ~1cycle
  - Shared memory: ~30 cycles
  - L1/L2 cache: ~300 cycles
  - Global memroy: ~600 cycles
- Best practices for CUDA programming
  - Maximize thread occupancy
  - Optimize memory access
  - Profile continuously
  - Avoid thread divergence    

### 46. 45. GPU Memory Hierarchy – Optimizing Usage
- Memory bandwidth and access patterns often create critical bottlenecks in AI workloads
- Registers: provide the fastest possible memory access with just 1 cycle latency
  - Allocated per CUDA thread for local variables
  - Managed automatically by the CUDA compiler
  - Limited resource - typically 255 max per thread on modern GPUs
  - Register spilling occurs when demand exceeds availability, forcing variables into slower memory
- Shared Memory
  - On chip memory with access speeds similar to L1 cache
  - Shared across all threads within the same thread block
  - Optimal use case
    - Matrix multiplication tiles that are re-used by multiple threads
    - Convolution operations with overlapping input regions
    - Reduction operations requiring inter-thread communication
- Cache hierarchy
  - L1 Cache
    - Per SM
    - Not programmable
  - L2 Cache    
    - Shared across all SMs
    - Cache friendly access patterns improve hit rates
- Global memory
  - Bandwidth utilization is critical
  - Coalesced access patterns essential
  - Minimizing unnecessary transfers
- Memory optimization techniques
  - Coalesced access
  - Shared memory tiling
  - Mixed precision
  - Gradient checkpointing: trade computation for memory by recomputing activations during backpropagation - computing is cheaper than memory consumption

### 47. 46. Multi-GPU Scaling and Interconnects (NVLink)
- Data parallelism
  - Mini batches are split across multiple GPUs
  - Each GPU has a comlete model copy
  - GRaidents computed independently on each device
  - Periodic sync
- Model parallelism
  - Beyond memory constraints
  - Pipeline parallelism: GPipe, PipeDream, DeepSpeed
  - Tensor parallelism: splitting individual layers
  - Sequence parallelism: distributing sequence dimensions
- NVLink architecture
  - Direct memory access b/w GPUs
  - Elimination of GPU as communication middleman
  - Support for unified memory address
- NVSwitch
  - For multiple nodes
- Best practices for multi-GPU scaling
  - Prioritize NVLink/NVSwitch enabled HW when budge allows
  - Optimize batch size
  - Implement mixed precision
  - Leverage NCCL for topology-aware collective operations
  - Benchmark at small scale before deploying to hundreds of GPUs

### 48. 47. Multi-Instance GPU (MIG) Configurations
- Oversubscription of GPUs
- Why multi-instance GPUs?
  - Small workloads waste GPU resources
  - Multi-tenant requirements
- Multi-Instance GPU (MIG)
  - HW partitioning: divides a physical GPU into isolated slices
    - Provided by HW, not SW
  - Dedicated resources: each slice has its own memroy and compute resource
  - Complete isolation: 
- Example of A100 MIG profiles
  - 1g.5gb: 1 compute slice with 5GB memory. ~1/7 of full GPU
  - 2g.10gb: 2 compute slices with 10GB memory. ~2/7 of full GPU
  - 3g.20gb: 3 compute slices with 20GB memory. ~3/7 of full GPU
  - 7g.40gb: 7 compute slices with 40GB memory. Full GP
- Benefits of MIG
  - Cost efficiency
  - Multi-tenancy
  - Higher utilization
  - Fault isolation
- MIG use cases
  - Inference mircoservices
  - Small-scale training
  - Multi-tenant GPU clusters
  - AI SaaS applications
- Enabling MIG
   - sudo nvidia-smi -mig 1 # enable MIG mode
   - sudo nvidia-smi -cgi 19,19 -C # create GPU instances
   - nvidia-smi -L # list MIG devices
- MIG + Kubernetes
  - Kubernetes scheduler views MIG instances as distinct GPUs
  - Automates MIG instance lifecycle management
  - Compatible with Slurm and other HPC schedulers
- Limitations of MIG
  - Limited HW support (A100, H100, L40S)
  - No NVLink b/w instances
  - Not for large models

### 49. 48. Benchmarking AI Workloads on GPUs
- Why benchmark?
  - Performance varies by task
  - Informed decision making
  - Deployment strategy
  - Future planning
- Types of GPU benchmarks
  - Synthetic: raw FLOPS, memory bandwidth
  - AI framework: Tensorflow/PyTorch training speed on standard datasets
  - Model-specific: ResNet, BERT, GPT benchmarks with real architecture
  - Inference: latency & throughput tests under production conditions
- Key metrics to track
  - Flops
  - Throughput: samples/sec or tokens/sec procssed during training or inference
  - Latency
  - Memory: vram utilization per batch
  - Energy efficiency
- Benchmarking tools
  - MLPerf
  - Nvidia Nsight
  - Pytorch/tensorflow profilers
  - nvidia-smi
- GPU memory benchmarks
  - Bandwidth tests: copy b/w host/device or device/device
  - VRAM capacity: maximum batch size
  - Memory fragmentation
  - Profiling tools
- Benchmarking multi-GPU scaling
  - Linear vs actual scaling
  - communication fabric
  - Distributed training: DDP(DistributedData Parallel) and model parallelism efficiencies  

### 50. 49. Lab – Run a Model on GPU with CUDA

## Section 9: Week 8: Distributed Training Basics

### 51. 50. Why Distributed Training Is Needed
- The scale of Modern AI
  - Requires peta bytes of training data across text, images, and video
  - A single GPU will take months or years to train them
- Why scale training across GPUs?
  - Accelerated training
  - Model capacity: fit larger models
  - Fault tolerance  
- Parallelism in training
  - Data parallelism: split batches across GPUs, combine gradients
  - Model parallelism: split layers across GPUs
  - Pipeline parallelism: stage execution like assembly line
- Compute vs communication tradeoff
  - Training speed = compute throughput - communication overhead
  - High speed interconnects are critical for minimizing overhead
- Challenges of distributed training
  - Communication bottlenecks
  - Straggler problem: the slowest node decides the training speed
  - Debugging complexity  

### 52. 51. Data Parallelism vs Model Parallelism
- Why parallelism matters in AI
  - Too large and too computationally expensive
- Two fundamental approaches
  - Data parallelism (DP)
  - Model parallelism (MP)
- Data parallelism
  - Complete model copy on each GPU
  - Different mini-batches processed in parallel
  - Synchronized gradient updates via AllReduce  
  - PyTorch DDP, TensorFlow MirroredStrategy
- Data parallelism workflow
  - Split and distribute
  - Parallel computation
  - Gradient synchronization
  - Model update
  - Repeat
- Strengths of data parallelism
  - Implementation simplicity
  - Linear scaling
  - Model compatibility
  - HW optimization
- Weakness of data parallelism
  - Memory constraint: each model must fit in a single GPU memory
  - Communication bottleneck: gradient sync creates network overhead
  - Straggler problem: the training speed is determined by the slowest node
  - Diminishing returns: communication overhead leads to efficiency drops at extreme scale (>1000 GPUs)
  - Batch size challenges: Larger batch size may harm model generalization
- Model parallelism
  - Each GPU holds specific layers or components  
  - Forward/backward passes flow sequentially across devices
  - Required when model size is larger than a single GPU memory
- Model parallelism workflow
  - Layer distribution: partition model layers across available GPUs
  - Sequential forward pass
  - Continue chain
  - Backward pass
  - Memory optimization
- Strength of model parallelism
  - Enables ultra-large models
  - Memory efficiency
  - Pipeline integration
  - Advanced techniques: supports memory optimization strategies like Zero Redundancy Optimizer and Fuilly Sharded Data Parallel
- Weakness of model parallelism
  - Implementation complexity
  - Partitioning challenges
  - Communication overhead
  - Load balancing
  - Training time increase
- Modern best practice
  - Hybrid approaches combining data parallelism, model parallelism and pipeline parallelism to maximize efficiency

### 53. 52. PyTorch Distributed Training
- PyTorch distributed architecture
  - torch.distributed
  - DistrubtedDataParallel (DDP)
  - RPC framework
- Communication backends
  - NCCL
  - Gloo: meta only. Both CPU and GPU  
  - MPI
- Distributed Data Parallel (DDP)
  - PyTorch's flagship distributed training paradigm
  - Replicates the entire model across all GPUs
  - Signficantly more efficent than legacy DataParallel
- DDP workflow
  - Initialize process group
  - Wrap model in DDP
  - Configure data sampling
  - Train with gradient sync
  - Save distributed checkpoints
- Common challenges
  - Process synchronization issues
  - Communication overhead
  - Hyperparameter adaptation
  - Distributed debugging complexity
- Best practices
  - Backend selection: Use NCCL
  - GPU bidning: pin each process to specific GPU using `device_ids=[rank]`
  - Monitoring
  - CUDA optimization: `torch.backends.cudnn.benchmark=True`
  - Mixed precision: use automatic mixed precision (AMP)to reduce communication volume

### 54. 53. TensorFlow Multi-GPU Training
- Why Tensorflow multi-GPU training?
  - Resource efficiency
  - Built-in solutions
  - Scalability: single GPU-> multi GPUs -> multinodes environment scales seamlessly
- Distribution strategies in TensorFlow
  - MirroredStrategy
    - Single node multi-GPUs
    - Creates an exact model replica on each available GPU
    - Automatically slits input batch across devices
    - Graidents are reduced and averaged across replicas
    - `tf.distribute.MirroredStrategy()`
  - MultiWorkerMirroredStrategy
    - Extends MirroredStrategy across multiple servers
    - Uses collective communication operations for gradient synchronization
    - Each work runs identical model replica
    - Requires TF_CONFIG environment variable for cluster configuration
    - Best for on-premise cluster with 10-100 GPUs
  - ParameterServerStrategy
    - Asynchronous training across distributed workers
    - Dedicated parameter servers store and update model variables
    - Workers compute gradients an dpush updates to paramter servers
    - ~ 1000 GPUs/CPUs
  - TPUStrategy  

### 55. 54. Horovod and AllReduce Explained
- Why Horovod?
  - Seamless integration with TensorFlow, PyTorch, and MXNet
  - Built around AllReduce for efficient gradient synchronization
  - Scales smoothly up to 1000+ GPUs with minimal code changes
- AllReduce example
  - Step 1: Local computation
  - Step 2: Sum all gradient
  - Step 3: Broadcast results
  - Step 4: Weight update
- Ring AllReduce in Horovod
  - GPUs are arranged in a logical ring topology
  - EAch GPU communicates only with its neighbors
  - Partial gradients are exchanged in multiple steps
  - Communication overlaps with computation for efficiency
  - Bandwidth utilization is maximized while avoiding central bottlenecks
- Benefits of Horovod
  - Simplicity
  - Framework-agnostic
  - Efficiency
  - Scalability
  - Adoption
- Limitations of Horovod
  - Demands high speed interconnects like NVLink or Infiniband
  - Not suitable for model parallelism
  - PyTorch DDP may yield better results  

### 56. 55. Fault Tolerance in Distributed Training
- Why fault tolerance matters
  - Failures become inevitable
  - Without proper fault tolerances, entire jobs must restart from scratch
- Types of failures
  - HW
    - GPU overheating
    - VRAM corruption
    - Power fluctuation
    - Disk failures
  - SW
    - Memory leaks
    - CUDA driver bugs
    - Framework deadlocks
    - OOM (Out-Of-Memory) exception
  - Network
    - Node communication failures
    - Intermittent packet loss
    - Bandwidth saturation
    - NIC failures
  - Scheduler/cluster
    - Kubernetes pod evictions
    - Slurm job preemptions
    - Resource contention
    - Maintenance downtime
- Checkpointing
  - Periodically save model weights + optimizer state
  - Enables resuming trainig from last checkpoint
  - Frequency tradeoff: overhead vs potential data loss
- Elastic training
  - Next generation fault tolerance
  - Dynamic adaptation
  - Resizable jobs
  - Failure resilience
  - PyTorch elastic, TorchElastic, Horovod Elastic, Ray Train, DeepSpeed Elastic
- Infrastructure level fault tolerance
  - Kubernetes: pod auto-restarts +  rescheduling policies
  - Slurm: job retries + checkpoint continuation
  - Cluster Managers: automatic task reassignment + GPU health monitoring
- Challenges in fault tolerance
  - Storage bottlenecks
  - Model parallelism complexity
  - Restart overhead
  - Performance safety tradeoff
- Best practices
  - Automate checkpointing
  - Incremental checkpoints
  - Test recovery flows
  - Multi-level resilience

### 57. 56. Lab – Train ResNet Across Multiple GPUs

## Section 10: Week 9: Workflow Automation & Experiment Tracking

### 58. 57. Why Tracking ML Experiments Matters
- The challenge of ML development
  - Hundreds of experiments with varying:
    - Hyperparameters
    - Dataset compositions and preprocessing steps
    - Model architecture and feature engineering
- Experiment tracking
  - Systematic logging
  - Dataset versioning
  - Run comparison
- Experiment tracking forms the core foundation of MLOps best practices, making your AI work truly reproducible and scientific
- Key things to track
  - Code version
  - Data version
  - Hyperparameters: batch size, learning rate, optimizer choice, regularization values
  - Metrics: accuracy, loss, precision/recall, inference latency
  - Artifacts: trained model weights, logs, visualization plots
- Benefits of experiment tracking
  - Reproducibility
  - Transparency
  - Scalability
  - Knowledge sharing
- Popular tools for tracking
  - MLflow
  - Weights and biases
  - comet.ml
  - TensorBoard
  - Auzre ML, AWS SageMaker, Google Vertex AI

### 59. 58. Introduction to MLflow
- Why MLflow?
  - Framework-agnostic
    - Compatible with PyTorch, Tensorflow, Scikit-learn, ...
  - Industry standard
  - Flexible deployment
- Core components of MLFlow
  - Tracking
    - Automatically logs everything you need
      - Hyperparametres
      - Performance metrics
      - Artifacts: model files, visualiztion, feature importance
      - Environment details: python version, library dependencies
  - Projects
    - Defines reproducible ML workflows that can be shared and executed
    - Environment management through conda.yaml or requirements.txt
    - Standardized entry points
    - Git integration
  - Models
    - A standard format for packaging ML models that works with diverse serving tools
    - Supports flavors: TensorFlow, PyTorch, scikit-learn, ONNX
  - Model registry
    - ML governance layer
    - Version control
    - STage transitions
    - Metadata tracking
```py
import mlflow
mlflow.start_run()
mlflow.log_param("lr", 0.01)
mlflow.log_metric("accuracy",0.92)
mlflow.log_artifact("model.pth")
...
mlflow.end_run()
```
- MLflow UI
  - Browsing all experiment runs
  - Comparing metrics and visualizations across multiple experiments
  - Searching and filtering by parameters
  - Examining artifacts and model details

### 60. 59. Logging Metrics and Artifacts
- Why logging matters
  - Lost results
  - Performance tracking
  - Reproducibility
- Logging creates the essential foundation for **scientific reproducibility** in AI/ML development
- Metrics
  - Quality metrics: Accuracy, F1, Precision, Recall, AUC
  - Training metrics: Loss, perplexity, gradient norms
  - Performance metrics: Latency, throughput, memory usage  
- Artifacts: Files generated during experiments that provide context and reproducibility  
  - Model artifacts: trained weights, checkpoints, serialized models
  - Visualization artifacts: loss curves, confusion matrices, embedding projections
  - Data artifacts: preprocssed datasets, feature statistics, sample predictions
- Logging metrics
```py
import mlflow
with mlflow.start_run():
    mlflow.log_param("lr",0.01)
    mlflow.log_metric("loss",0.345)
    mlflow.log_metric("accuracy",0.89)
```
  - Tracks metrics for each experiment run
  - Supports multiple metrics per run
  - Automatic experiment organization
- Logging artifacts
```py
import mlflow
with mlflow.start_run():
    mlflow.log_artifact("confusion_matrix.png")      
    mlflow.log_artifact("model.pth)
```
  - Artifacts are stored alongside run history
  - Downloadable for later comparison
  - Organized by experiment ID
- Common mistakes to avoid
  - Insufficient metrics
  - Missing checkpoints
  - Unversioned storage
  - Log overwriting
  - Poor tagging

### 61. 60. Versioning Data, Models, and Parameters
- Why versioning matters
  - Experiment complexity
  - Regulatory requirements
  - Team collaboration
- Three pillars of ML versioning
  - Data: raw and processed ataset
  - Models: trained weights, architectures, frameworks, and inference code
  - Parameters: configuration, hyperparameters, random seeds, and environment settings
- Versioning data
  - Track with unique identifiers
  - Version control solutions
    - DVC (Data Version Control)  
    - Git LFS
    - LakeFS
  - Log transformations: always document preprocessing steps and feature engineering trnasformations
- Versioning models
  - Save complete artifacts
  - Implement registry system
  - Enable model lifecycle
- Versioning parameters
  - Track hyperparameters: learning rates, batch sizes, loss functions, optimizer configurations
  - Use config files: JSON/YAML
  - Log random seeds
- Benefits of comprehensive versioning
  - Debugging
  - Reproducibility
  - Collaboration
  - Compliance
  - CI/CD for ML
- Common tools for versioning
  - MLflow Registry
  - DVC and Git LFS
  - Weight & Biases
  - LakeFS
- Challenges in versioning    
  - Storage costs
  - Syncing components
  - Environment consistency
  - Governance vs. Flexibility

### 62. 61. Weights & Biases vs MLflow Comparison
- MLflow
  - Created by Databricks
  - Four integrated components
    - Tracking, projects, models, and registry
  - End-to-end ML lifecycle focus
- Weights & Biases
  - SaaS-First approach: cloud native platform
  - Rich visual interfaces
  - Framework integration
  - Research optimization

### 63. 62. Automating Training Pipelines
- Why automate training pipeline?
  - Manual training processes lead to:
    - Inconsitent results:
    - Error-prone workflows
    - Significant time waste
    - Limited scalability
- What is a training pipeline?
  - Data ingestion and preprocessing
  - Training and validation
  - Evaluation and logging
  - Deployment (optional)
- Benefits of automation
  - Consistency
  - Scalability
  - Speed
  - Reliability
  - Reproducibility
- Core tools for automation
  -  Orchestration tools
    - Apache Airflow - DAG based orchestraiton with extensive connectors
    - Kubeflow Pipelines
    - Perfect/Dagster
  - ML-specific platforms
    - MLflow projects
    - SAgemaker pipelines
    - Vertex AI
- Example: training pipeline stages
  - Data ingestion
  - Feature engineering
   - Model training
   - Evalulation
   - Registry Update
   - Deployment
- Autmating retraining
  - Data-driven triggers
  - Performance-based triggers
  - Schedule-based triggers
- Automated retraining is essential for **real-time AI applications** such as fraud detection, recommendation systems, and programmatic advertising where data patterns evolve rapidly
- Best practices
  - Version everything
  - Build fault tolerance
  - Modularize components
  - Monitor everything
  - Secure your pipeline 

### 64. 63. Lab – Track Experiments with MLflow

## Section 11: Week 10: CI/CD for AI Models

### 65. 64. What Is MLOps and CI/CD for AI?
- Why MLOps matters?
  - ML models are dynamic
  - Traditional DevOps Falls short
  - MLOps bridges the gap from DevOps
- What is MLOps?
  - Combines ML with DevOps practices
  - Data engineering
  - Model development
  - Deployment
  - Monitoring
  - Automation: creating CI/CD pipelines for the entire process
- Core pillars of MLOps
  - Data management
  - Model training & tracking
  - Deployement and serving
  - Monitoring and Governance
  - Collaboration
- What is CI/CD in AI?
  - Continuous Integration (CI)
    - Automatically test and validate changes to:
      - ML code & algorithm
      - Data preprocessing
      - Model performance
      - Integration with other systems
  - Continuous Deployment (CD)
    - Automate the release process for:
      - Model artifacts and weights
      - Serving infrastructure
      - Configuration updates
      - A/B testing new models
- CI/CD workflow for ML
  - Code commit
  - Data pipeline
  - Model training
  - Testing
  - Registry
  - Deployment
- Tools for CI/CD in AI
  - Pipeline automation
    - Github actions
    - Gitlab CI
    - Jenkins
    - CircleCI
  - Experiment & versioning
    - MLflow
    - DVC 
    - Wandb
    - Neptune.ai
  - Workflow orchestration
    - Kubeflow pipelines
    - Apache Airflow
    - Prefect
    - Metaflow
  - Deployment Infrastructure
    - Docker
    - Kubernetes
    - Seldon core
    - KServe
- Cloud native options
  - AWS Sagemaker pipelines
  - Google Vertex AI
  - Azure ML
  - Databricks MLflow
- Benefits of MLOps + CI/CD
  - Accelerated innovation
  - Reliability at scale
  - Cross-functional collaboration
  - Continuous value delivery
- Challenges in AI CI/CD
  - Data dependency complexity
  - Resource intenstive training
  - Testing complexity
  - Organizational alignment

### 66. 65. GitHub Actions Basics for ML Projects
- Why github actions for ML?
  - Native automation
  - ML pipeline ready
  - Accessible
- What are github actions?
  - An event driven automation platform
  - Triggered by
    - Code pushes or pull requests
    - Scheduled jobs using cron syntax
    - Manual triggers vial GitHub UI
  - All workflows defined in `.github/workflows/` folder as YAML files
- GitHub Actions for ML lifecycle
  - Data validation: verify schema consistency, check for missing values
  - Unit tests: preprocessing pipelines 
  - Model training
  - Artifact logging
  - Deployment
- Runners for ML workload
  - GitHub-hosted runners: cpu only with 2 core processors
  - Self-hosted runners: use your own GPU servers for training. Requires infrastructure management
  - Cloud integration: critical for deep learning CI/CD
- Benefits of GitHub Actions
  - Integrated
  - Reusable
  - Scalable
  - Secure
  - Extensible
- Challenges and limitations
  - HW constraints
  - Time limits
  - Cost considerations
  - Workflow complexity
  - Infrastructure needs
      
### 67. 66. GitLab CI and Jenkins Pipelines
- Why CI/CD matters for ML
  - Frequent retraining and redeployment cycles as data or requirements evolve
  - Complex testing requirements for both code and model performance
  - Resource-intensive workflow requiring specialied HW
- GitLab CI
  - Native integration
  - YAML based workflows
  - Runners Architecture
  - Security Focus
- GitLab CI for ML
  - GPU support
  - ML pipeline stages
  - Container registry
  - Artifact management  
- Jenkins
  - Open-source legacy
  - Extensive plugin ecosystem
  - Groovy-based pipelines
  - Enterprise adaptability
- Jenkins for ML projects
  - Source integration
  - Compute orchestration
  - Deployement automation
- Jenkins is popular in regulated industries like finance, healthcare, and telecommunications

| Feature | GitLab CI |  Jenkins |
|----------|-------------|---------|
| Hosting options | SaaS + self-hosted | self-hosted only|
| Configuration| YAML (declarative) | Groovy DSL(programmatic) |
| GPU support | Via GitLab runners |  Via custom agents|
|Ease of Setup | Simple, built-in | Flexible, but complex |
| ML Integration| Container registry, artifacts | Rich plugin ecosystem |
| Best suited for | Cloud-native ML teams | Enterprise hybrid infrastructure |

### 68. 67. Testing Models Before Deployment
- Why test ML models?
  - Compared to SW development, ML needs validation stage
- What to test in ML models
  - Data
  - Training process
  - Performance metrics
  - Robustness
  - Fairness & bias
- Unit tesing for ML
  - Data processing functions verification
  - Feature engineering steps validation
  - Random seed consistency checks
  - Small dataset mocks for rapid iteration
- Integration testing for ML
  - Data ingestion
  - Preprocessing
  - Training
  - Inference
- Model validation metrics
  - Classification metrics: Accuracy, precision, recall, F1-score, ROC-AUC
  - Regression & system metrics: RMSE/MAE, R2 Latency, throughput, memory usage
- Robustness testing
  - Noisy input testing
  - Adversarial testing
  - Load testing
  - Drift simulation
- Fairness and bias testing
  - Use specialized tools: AIF360, Fairlearn, What-If Tool
  - Apply multiple fairness metrics
  - This is mandatory in regulated business
- Automation in CI/CD
  - Commit code
  - Run unit tests
  - Train model
  - Evaluate metrics
  - Log results
  - Deploy or Fail
- Best practices
  - Data management
    - Version control your test datasets
    - Maintain separate validation and test sets
    - Create specialized test cases for known edge cases
    - Generate synthetic data for rare scenarios
  - Testing strategy
    - Set minimum performance thresholds based on business impact
    - Test under simulated realworld conditions
    - Implement A/B testing for production validation
    - Monitor continuously post-deployment

### 69. 68. Automating Docker Builds for AI Apps
- Why docker for AI applications?
  - Consistent environments
  - Complete packaging
  - Scalable deployment

### 70. 69. Canary Deployments and Rollbacks
- Canary deployment: a new version is rolled out into a small subet of users before a full launch
- Why Canary deployments?
  - Reduce risk
  - Detect issues early
  - Minimize downtime
- Canary workflow for AI models
  - Deploy canary
  - Route limited traffic: 5-10% of production traffic
  - Monitor key metrics
  - Increase traffic gradually
  - Rollback if necessary
- Example with Kubernetes
```yaml
spec:
  traffic:
  - revisionName: model-v1
    percent: 90
  - revisionName: model-v2
    percent: 10
```
- Benefits of Canary for ML
  - Safety first
  - Real-world validation
  - Shadow testing
  - A/B testing support
  - Accelerated feedback      
- Monitoring Canary deployments
  - System metrics: latency, throughput, error rate & exceptions, resource utilization
  - ML-specific metrics: prediction quality, inference confidence, data drift detection, model bias indicators
  - Tooling: Prometheus + Grafana, EvidentlyAI, Seldon Alibi Detect, Custom dashboards
- Rollbacks in ML CI/CD
  - Detect issue
  - Initiate rollback
  - Restore Previous version
  - Analyze failure
- Immediate rollback vs gradual rollback
  - Issues are critical vs significant

### 71. 70. Lab – CI/CD Pipeline for Model Deployment

## Section 12: Week 11: Advanced Kubernetes for AI

### 72. 71. GPU Scheduling in Kubernetes
- Kubernetes and GPU support
  - Nvidia device plugin: enables Kubernetes to recognize GPUs as schedulable resources
  - Resource requests: Pods request GPUs like CPUs or memory
  - Cluster support
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```
- Nvidia Device plugin
  - Deploys as a Daemonset across all GPU nodes
  - Advertises available GPUs to the Kubernetes scheduler
  - Available in A100, H100 and other datacenter GPUs
  - Multi-instance GPU (MIG) technology
- Scheduling across nodes
  - Node selectors
  - Tains and tolerations
  - Affinity rules  
- Monitoring GPU usage in Kubernetes
  - Basic monitoring
    - `kubectl describe node`: node configuration and status
    - `kubecetl describe pod`: pod configuration and status
  - Advanced monitoring
    - Nvidia DCGM exporter -> Prometheus metrics
    - Grafana dashboards
    - Custom alert

### 73. 72. StatefulSets for Data-Heavy Workloads
- A statefulSet is a Kubernetes API object used to manage stateful applications, which require stable network identities, persistent storage, and ordered deployment and scaling
  - Stable identity
  - Persistent storage
  - Ordered operations
- Benefits for Data-heavy workloads
  - Consistent identity
  - Data sharding
  - High availability
  - Long-running jobs

| Feature | Deployment | StatefulSet |
|-----------|-----------|------------|
| Pod identity |  Random, replaceable | stable, sequential |
| storage: shared or ephemeral | persistent per pod|
| Use case | stateless application | data-heavy ML infrastructure |
| Scaling behavior | any order, parallel | ordered, predicatable |
| Pod replacement|  Completely new instance |  Same identity, storage retained |
| Network Requirements | Basic | Headless Service recommended |

- Challenges with StatefulSets
  - Increased complexity
  - Recovery complexity
  - Storage cost scaling
  - Network requirements
  - Sequential operations
  - Cloud-provider limitations
- Best practices
  - Use when appropriate
  - Headless services
  - Storage management
  - Backup strategies
  - Pod disruption budgets  

### 74. 73. Kubernetes Operators for ML
- What is Kubernetes operator?
  - Extensions of the Kubernetes control plane
  - Uses Custom Resource Definitions (CRDs)
  - Reconciliation loop
  - Operational knowledge as code
- Why ML needs operators
  - Automates distributed training
  - Simplifies model serving
  - Manages ML infrastructure
  - Reduces DevOps overhead
- Example of ML operators
  - Kubeflow TFJob
  - PyTorchJob Operator
  - KFServing/KServe
  - MLflow Operator
  - Ray Operator
- Benefits of ML operators
  - Declarative ML workloads
  - Seamless scaling
  - Automated fault recovery
  - Infrastructure-as-code
- Operators in model serving
  - KFServing/Kserve: Deploy models with YAML
    - Advanced deployment strategies: A/B testing, canary rollouts, and autoscaling
    - Framework agnostic
    - Integrated monitoring
- Challenges with operators
  - Learning curve
  - Resource overhead
  - Debugging complexity
  - Framework coverage
  - Kubernetes expertise
- Best practices
  - Start with community operators
  - Keep configs in Git
  - Monitor operator health
  - Focus on scalable infrastructure  

### 75. 74. Helm Advanced Templates for AI Apps
- Why Helm?
  - K8 is too complex

### 76. 75. Scaling AI Training with Kubeflow
- Kubeflow provides a specialized platform for ML that addresses the unique challenges of AI workloads
  - Native Kubernetes integration
  - Simplified distributed training
  - Enterprise-scal MLOps
- Kubeflow
  - Opensource ML toolkit
  - Modular components
  - MLOps simplification
- Kubeflow training operators
  - TFJob
  - PyTorchJob
  - MXNetJob
  - XGBoostJob
- Benefits of Kubeflow training
  - Declarative ML workloads
  - Automatic resource scheduling
  - Seamless scaling
  - Fault tolerance
  - Multi-Cloud compatibility
- Katib: Kubeflow's AutoML component for hyperparameter optimization
  - Runs parallel experiments across multiple GPUs/clusters

### 77. 76. Kubernetes Security Best Practices
- Why security in Kubernetes
  - High value targets
  - Complex attack surface
  - Constly consequences
- Common security vulnerabilities in Kubernetes
  - Critical weaknesses
    - Exposed API servers
    - Container escapes
    - Plaintext secrets
  - Operational gaps
    - Privilege escalation
    - Unvetted images
    - Misconfigured network polices
- Implementing the Principle of Least Privilege (PoLP)
  - Key practices
    - Define granular roles with specific verbs (get, list, watch)
    - Avoid wildcard permissions and cluster-wide access
    - Regularly audit role bindings and clean up unused acounts
- Implement pod security admission with the restricted profile 
- Read-only filesystems prevent attackers from modifying inference logic
- Network security for AI clusters
  - Default deny
  - Workload segmentation
  - Encryption in transit
- Container image security
  - Secure base images
    - Use official, minimal images from trusted sources
    - Nvidia NGC containers for GPU workload
    - Official PyTorch/Tensorflow images
    - Distroless containers where possible
  - Continuous scanning
    - Implement automated vulnerability scanning
- Monitoring and detection for AI workloads
  - API server audit logs
  - Runtime security
  - Resource anomalies
- Multi-tenancy security for shared AI platforms
  - Isolation mechanisms
    - Namespace boundaries
    - Pod security admission
    - Network segmentation
    - Custom admission controllers

### 78. 77. Lab – Deploy MLflow on Kubernetes

## Section 13: Week 12: Resource & Cost optimization

### 79. 78. Why AI Infrastructure Costs Spike
- Compute cost
  - GPU/TPU expenses scale dramaically
  - Training complexity grows exponentially: doubling parameters can increase compute needs by 4-8x
  - Idle GPU time
  - Cloud over-provisioning: too many reserved nodes
- Data & storage: petabyte challenge
  - Typical cost for petabyte-scale ML data storage with high availability: $150K monthly
  - Data transfer b/w regions: $20K monthly
- Networking: the hidden cost multiplier
  - Distributed training: high-bandwidth, low latency interconnection costs 3-5x more thant standard networking tiers
  - Cross-region traffic
  - Inference scaling: As users grow, API calls and network usage scale linearly with volume
  - Latency requirements: low-latency networks cost more
- Operational inefficiencies: wasted resources
  - Poor workload scheduling
  - Lack of granual monitoring
  - Duplicate experiments
  - Optimization oppportunities missed
- Beyond HW: human and organization costs
  - Higher salary for ML engineers
  - FRagmented workflows
  - Vendor lock-in
  - Compliance and Governance

### 80. 79. Using Spot Instances for AI Training
- Cutting costs with cloud's hidden discounts
- Spot instances
  - Unused capacity
  - Massive savings: 70-90% cheaper than on-demand instances
  - Training focus: training job can handle occasional disruption
- How spot instances work
  - The cloud provider can reclaim these resources at any time when demand increases
  - Preemption reality: your workload might be stopped with minimal notice  
- Challenge and risks
  - Interruption management
  - Checkpoint discpline
  - Regional variability
  - Inference mismatch: generally unsuitable for latency-sensitive inference workloads
- Best practices for spot success
  - Checkpoint everything
  - Hybrid deployment
  - Use management tools
  - Monitor availability
- A well-architected spot strategy is required

### 81. 80. Autoscaling AI Clusters in Cloud
- Static provisioning may lead to:
  - Over-provisioning
  - Under-provisioning
- Elasticity is the cornerstone of cloud efficiency
- Types of autoscaling for AI clusters
  - Horizontal scaling: adds or removes nodes
  - Veritical scaling: Adjust resources (CPU, memory, GPU) allocated to existing nodes
  - Cluster autoscaler
  - Pod autoscaler
- Challenges and tradeoffs in AI autoscaling
  - Cold start latency
  - GPU scarcity
  - Training complexity
  - Configuration risks
- Best practices for AI cluster autoscaling
  - Define clear scaling metrics
  - Implement buffer capacity
  - Optimize instance mix
  - Monitor and refine

### 82. 81. Monitoring Resource Utilization

### 83. 82. Storage Cost Optimization Strategies

### 84. 83. Multi-Tenant Cost Allocation in Teams

### 85. 84. Lab – Optimize Cloud AI Workload Costs

## Section 14: Week 13: Networking for AI Systems

### 86. 85. Fundamentals of Data Center Networking
### 87. 86. Software Defined Networking (SDN) for AI
### 88. 87. Infiniband and High-Speed Interconnects
### 89. 88. Load Balancing for AI Inference
### 90. 89. Network Bottlenecks in Distributed Training
### 91. 90. Security in Networked AI Systems
### 92. 91. Lab – Configure Load Balancer for AI API

    3min

### 93. 92. From Training to Serving – The Deployment Gap
### 94. 93. REST vs gRPC for Model APIs
### 95. 94. TensorFlow Serving for AI Models
### 96. 95. TorchServe for PyTorch Models
### 97. 96. Deploying Models with FastAPI
### 98. 97. Scaling Model Serving with Kubernetes
### 99. 98. Lab – Serve an Image Classifier with FastAPI

    2min

### 100. 99. NVIDIA TensorRT Optimization
### 101. 100. Triton Inference Server Basics
### 102. 101. Batch Inference vs Online Inference
### 103. 102. Caching for Fast Inference
### 104. 103. Multi-Model Serving Strategies
### 105. 104. A/B Testing Model Endpoints
### 106. 105. Lab – Deploy a Model on Triton Server

    3min

### 107. 106. Why Monitoring AI Systems Matters
### 108. 107. GPU Monitoring with DCGM
### 109. 108. Metrics Collection with Prometheus
### 110. 109. Visualization Dashboards with Grafana
### 111. 110. Tracing AI Requests with OpenTelemetry
### 112. 111. Building Alerts for AI System Failures
### 113. 112. Lab – Monitor GPU Cluster with Prometheus

    5min

### 114. 113. What Is Concept Drift vs Data Drift?
### 115. 114. Why Drift Destroys AI Performance
### 116. 115. Tools for Drift Detection (EvidentlyAI)
### 117. 116. Real-Time Drift Monitoring Pipelines
### 118. 117. Human-in-the-Loop Drift Evaluation
### 119. 118. Mitigation Strategies – Retraining & Rebalancing
### 120. 119. Lab – Build a Drift Detection Pipeline

    1min

### 121. 120. Security Risks in AI Infrastructure
### 122. 121. Identity and Access Management (IAM)
### 123. 122. Secrets Management for AI Systems
### 124. 123. Data Encryption at Rest and In Transit
### 125. 124. Model Theft and Adversarial Attacks
### 126. 125. Compliance Standards (GDPR, HIPAA, SOC2)
### 127. 126. Lab – Secure a Model Endpoint with Authentication

    3min

### 128. 127. Why AI Systems Fail in Production
### 129. 128. Fault Tolerance in AI Inference
### 130. 129. Redundancy and Failover for AI APIs
### 131. 130. Designing High-Availability AI Clusters
### 132. 131. Auto-Healing Infrastructure with Kubernetes
### 133. 132. Chaos Engineering for AI Systems
### 134. 133. Lab – Build a HA AI Inference Cluster

