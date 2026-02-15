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
- Goal: Launch a cloud VM with a GPU, connect via SSH, install a basic ML stack, and verify the GPU is usable from Python.
- What you’ll build: A Ubuntu-based VM with CUDA drivers + PyTorch ready to run GPU workloads.
- Estimated time: 60–90 minutes (including account setup).
- Cost guardrails: Use entry-level GPU flavors; stop the VM when not in use.
- Prerequisites (once per cloud account)
  - A credit-card verified AWS or Google Cloud account.
  - SSH client (macOS/Linux have it; on Windows use PowerShell or Windows Terminal).
  - A stable network; allow outbound HTTPS.
- Path A — AWS EC2 (Deep Learning AMI, GPU)
  - Why this path? AWS’s Deep Learning AMIs come pre-loaded with GPU drivers and popular frameworks, so you minimize setup time.
```
A1) Create key pair (for SSH)

    Go to EC2 → Key pairs → Create key pair.

    Type a name (e.g., ai-lab-key), type = RSA, file format = .pem (for macOS/Linux) or .ppk (for PuTTY).

    Save the file securely.

        Why: This private key is how you authenticate via SSH.

A2) Check GPU instance quotas

    In EC2 → Limits/Quotas, ensure you have capacity for g5 or g4dn.

    If not, Request limit increase (choose a small size like g5.xlarge).

        Why: Many accounts start with zero GPU quota.

A3) Launch the instance

    EC2 → Instances → Launch instances.

    Name: ai-lab-aws.

    AMI: search and select “Deep Learning AMI (Ubuntu)” (GPU version).

    Instance type: g5.xlarge (entry-level GPU with 1x NVIDIA A10G).

    Key pair: choose ai-lab-key.

    Network settings: Create/choose a security group:

        Allow SSH (22) from My IP only.

    Storage: set 100 GB gp3 (room for datasets).

    Click Launch instance.

        Why: The DLAMI includes CUDA/NVIDIA drivers and Conda envs out of the box.

A4) Connect via SSH

    From Instances list, copy Public IPv4 address.

    In terminal (macOS/Linux):

        chmod 400 ~/Downloads/ai-lab-key.pem
        ssh -i ~/Downloads/ai-lab-key.pem ubuntu@<PUBLIC_IP>

        Why: chmod 400 protects your key; AWS requires it.

A5) Verify GPU & drivers

    nvidia-smi

    You should see GPU model, driver, memory, and processes.

    If not present: Your AMI likely isn’t the GPU DLAMI or quota gave you a CPU instance—terminate and relaunch correctly.

A6) Update and prepare Conda

DLAMI usually includes Conda; check with:

    conda --version || echo "Conda not found"

    If present:

        conda create -n ai python=3.10 -y
        conda activate ai

    If not present (rare on DLAMI): install Miniconda:

        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        conda create -n ai python=3.10 -y && conda activate ai

A7) Install PyTorch with CUDA and verify

    python -V
    pip install --upgrade pip
    # Example install; adjust to current CUDA wheel per PyTorch docs if needed:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    python - <<'PY'
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.rand(1024,1024, device='cuda')
        y = torch.mm(x, x)
        print("Matmul OK, y shape:", y.shape)
    PY

    Why: Confirms PyTorch sees the GPU and runs a tiny kernel.
```
- Path B — Google Cloud (Compute Engine, GPU)
  - Why this path? GCP offers a one-click GPU driver install and modern GPU types like L4 (g2) and A100 (a2).
```
B1) Project & API

    In Google Cloud Console, create/select a Project.

    Enable the Compute Engine API.

        Why: VMs require this API.

B2) Request GPU quota

    IAM & Admin → Quotas. Filter by NVIDIA L4 or A100 in your region (e.g., us-central1).

    EDIT QUOTAS → request 1 GPU for g2 (L4) or a2 (A100).

        Why: GPU quota is not enabled by default.

B3) Create the VM

    Compute Engine → VM instances → Create instance.

    Name: ai-lab-gcp, Region/Zone: pick a zone with your GPU quota.

    Machine family:

        g2-standard-8 (1×L4) for cost-effective start, or

        a2-highgpu-1g (1×A100) if available.

    CPU platform: default.

    GPU: click CPU platform and GPU → Add GPU → select NVIDIA L4 (g2) or A100 (a2).

    Boot disk: Ubuntu 22.04 LTS, 100 GB.

    Firewall: allow SSH (HTTP/HTTPS optional).

    Click Create.

B4) Install NVIDIA drivers (one-click)

    After the VM is running, open it in the console; click “Install GPU driver” (if visible on the VM details page).

        OR SSH in and install manually:

            sudo apt-get update
            sudo apt-get install -y ubuntu-drivers-common
            sudo ubuntu-drivers autoinstall
            sudo reboot

        Why: Proper driver + CUDA install exposes the GPU to frameworks.

B5) Verify GPU & set up Conda

    SSH back in (from console SSH or your terminal):

        nvidia-smi

        You should see L4 or A100 listed.

    Install Miniconda and environment (if you prefer Conda):

        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        conda create -n ai python=3.10 -y && conda activate ai

B6) Install PyTorch with CUDA and verify

    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    python - <<'PY'
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.rand(1024,1024, device='cuda')
        y = torch.mm(x, x)
        print("Matmul OK, y shape:", y.shape)
    PY

Optional: Jupyter via SSH tunnel (either cloud)

    Use this to get a browser notebook without opening public ports.

    # On the VM:
    conda activate ai
    pip install jupyter
    jupyter notebook --no-browser --port 8888

From your local machine (replace host/IP accordingly):

    ssh -i ~/Downloads/ai-lab-key.pem -N -L 8888:localhost:8888 ubuntu@<AWS_PUBLIC_IP>
    # or for GCP:
    ssh -N -L 8888:localhost:8888 <gcp-username>@<GCP_EXTERNAL_IP>

Then open http://localhost:8888 in your browser and paste the token shown in the VM terminal.
Troubleshooting Cheatsheet

    Permission denied (publickey): wrong key, wrong user, or key file permissions. Use ubuntu@host for Ubuntu images and chmod 400 the key.

    nvidia-smi not found: wrong image (CPU-only) or drivers not installed; reinstall drivers or recreate instance with DLAMI (AWS) or the GPU driver tool (GCP).

    PyTorch says CUDA not available: version mismatch (driver/CUDA/toolkit). Reinstall PyTorch for the CUDA version supported by your driver, or update the driver.

    Slow downloads or installs: run sudo apt-get update && sudo apt-get upgrade -y first; try a closer region next time.

Cleanup (avoid surprise charges)

    Stop the VM when not in use (you can restart later).

    If done with the lab, Delete the VM and any attached disks or static IPs.

    Remove any snapshots you created.

    Keep your SSH key safe; rotate if it leaked.
```

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
- Goal: A clean Ubuntu box with Python envs, GPU driver (if available), PyTorch, JupyterLab, and Docker (+ NVIDIA runtime) — verified end-to-end.
```
0) Check OS, user, and hardware

    lsb_release -a           # confirm Ubuntu 22.04 (Jammy)
    whoami                   # your user
    lspci | grep -i nvidia || echo "No NVIDIA GPU detected"

    If you don’t see NVIDIA above, you can still proceed (CPU setup).

    If you’re on a cloud GPU VM, prefer a GPU image (e.g., AWS DLAMI) and skip driver steps below.

1) Update & base system hygiene

    sudo apt update && sudo apt -y upgrade
    sudo apt -y install build-essential git curl wget unzip zip tar ca-certificates \
      htop iotop iftop tree tmux pkg-config software-properties-common \
      nano vim neovim apt-transport-https gnupg lsb-release
    sudo reboot

Why: brings packages current, installs compilers/tools you’ll use constantly.
2) (Optional but recommended) Secure basics

    # Enable uncomplicated firewall (keeps SSH open)
    sudo ufw allow OpenSSH
    sudo ufw enable
    sudo ufw status

Why: sensible default protection, especially on cloud VMs.
3) Git + SSH keys + global config

    git --version
    git config --global user.name "Your Name"
    git config --global user.email "you@example.com"
     
    # Create an SSH key (press Enter to accept defaults, set a passphrase if you like)
    ssh-keygen -t ed25519 -C "you@example.com"
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    cat ~/.ssh/id_ed25519.pub   # add this to GitHub/GitLab settings

Why: seamless cloning/pushing and signed access.
4) Python environment manager (Miniconda)

    Fast, reliable, isolated environments for ML.

    cd ~
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    conda init bash
    exec $SHELL -l
    conda --version

Create your main env:

    conda create -n ai python=3.10 -y
    conda activate ai
    python -V

(Alternative: python3 -m venv .venv && source .venv/bin/activate if you prefer venv.)
5) Core Python stack

    pip install --upgrade pip wheel setuptools
    pip install numpy pandas scipy scikit-learn matplotlib jupyterlab ipywidgets \
      black isort flake8 pre-commit rich tqdm

Why: essentials for analysis, notebooks, and code quality.
6) NVIDIA GPU driver (only if you have an NVIDIA GPU & you’re not on a DLAMI)

    Do not install full CUDA toolkit unless you know you need it; PyTorch wheels ship with the CUDA runtime.

    # Detect & install the recommended proprietary driver
    sudo apt -y install ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    sudo reboot

Verify after reboot:

    nvidia-smi

You should see driver version and your GPU. If not, recheck hardware/VM type.
7) PyTorch install (GPU or CPU)

    GPU path (CUDA 12.1 runtime wheel):

    conda activate ai
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    CPU-only path:

    conda activate ai
    pip install torch torchvision torchaudio

Verify GPU access (works for CPU too):

    python - <<'PY'
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.rand((2048,2048), device='cuda')
        y = x @ x
        print("Matmul OK:", y.shape)
    PY

8) JupyterLab setup & quick test

    conda activate ai
    jupyter lab --version
    jupyter lab --no-browser --port 8888

    From your local machine, use an SSH tunnel if this is a remote server:

        ssh -N -L 8888:localhost:8888 <user>@<server-ip>

    Open http://localhost:8888, paste token, create a notebook, and run:

        import torch, pandas as pd
        torch.cuda.is_available(), pd.__version__

9) (Optional) OpenCV, Pillow, & extras

    pip install opencv-python pillow seaborn plotly ipykernel
    python -m ipykernel install --user --name ai --display-name "Python (ai)"

Why: common CV/visualization tools; kernel makes the env selectable in Jupyter.
10) Docker Engine (for reproducible runs)

    # Add Docker’s official GPG key & repo
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
      sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
     
    sudo apt update
    sudo apt -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker $USER
    newgrp docker
    docker --version
    docker run hello-world

Why: containers = portable, reproducible environments.
11) NVIDIA Container Toolkit (GPU in Docker)

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt update
    sudo apt -y install nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

Test GPU inside a container:

    docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

You should see your GPU in the container output.
12) Quality-of-life setup (nice to have)

    # tmux quickstart
    tmux new -s work
    # split: Ctrl-b then "
    # detach: Ctrl-b then d
     
    # Pre-commit hooks in repos
    cd /path/to/your/repo
    pre-commit install

Why: tmux keeps long trainings alive; pre-commit enforces consistent code.
13) Create a project template

    mkdir -p ~/projects/ai-starter/{data,notebooks,scripts,models,logs}
    cd ~/projects/ai-starter
    echo "numpy\npandas\nscikit-learn\ntorch\ntorchvision\n" > requirements.txt

Why: a standard layout speeds up every new project.
14) Verification checklist (what to confirm)

    nvidia-smi shows your GPU (if present).

    python test shows CUDA available: True (if GPU) and successful matmul.

    jupyter lab opens and can run a Python cell.

    docker run hello-world works.

    docker run --gpus all ... nvidia-smi shows GPU inside Docker.

15) Troubleshooting quick hits

    Driver mismatch / CUDA not available: ensure you installed proprietary NVIDIA driver and rebooted; reinstall PyTorch wheel for cu121.

    Conda not found after install: ensure PATH export in ~/.bashrc, then source ~/.bashrc.

    Docker needs sudo: you didn’t add your user to docker group; run sudo usermod -aG docker $USER && newgrp docker.

    Jupyter inaccessible on remote: use SSH port forward (-L 8888:localhost:8888) and keep Jupyter on --no-browser.
```

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
- Goal: Launch an AWS EC2 GPU instance, install drivers/frameworks, run a quick PyTorch GPU test.
- Time: ~60–90 minutes
- Cost: A few dollars depending on instance type (be sure to stop/terminate after use).
```
1️⃣ Prerequisites

    An AWS account (credit card verified).

    Basic knowledge of Linux shell (Day 9–13 content).

    AWS CLI installed locally (optional but recommended).

2️⃣ Create a Key Pair (SSH Access)

    In AWS Console, go to EC2 → Key Pairs → Create key pair.

    Select: Name: ai-keypair, Type: RSA, File format: .pem (Linux/Mac).

    Save the .pem file to ~/.ssh/ai-keypair.pem.

    Run locally:

        chmod 400 ~/.ssh/ai-keypair.pem

    ✅ Why: Protects your key file so SSH will accept it.

3️⃣ Check GPU Quotas

    Navigate to Service Quotas → EC2 → Running On-Demand G and P instances.

    If quota is 0, request a limit increase (e.g., 1 for g5.xlarge).
    ✅ Why: By default, new AWS accounts often have no GPU quota.

4️⃣ Launch a GPU Instance

    In AWS Console: EC2 → Launch Instance.

    Name: ai-ec2-lab.

    AMI: choose Deep Learning AMI (Ubuntu 22.04) GPU Optimized.

        Preinstalled: CUDA, PyTorch, TensorFlow.

    Instance type: g5.xlarge (NVIDIA A10G GPU, 4 vCPUs, 16 GB RAM).

    Key Pair: choose ai-keypair.

    Network Settings:

        Security Group → allow SSH (22) from My IP.

        (Optional) allow HTTP/HTTPS if serving models.

    Storage: set to 100 GB (default 8 GB too small for datasets).

    Click Launch Instance.

5️⃣ Connect to Your Instance

    From Instances list, copy Public IPv4 address.

    SSH in:

        ssh -i ~/.ssh/ai-keypair.pem ubuntu@<INSTANCE_PUBLIC_IP>

    ✅ Why: You’re now inside a cloud GPU server.

6️⃣ Verify GPU

Inside the VM, run:

    nvidia-smi

Expected: a table showing NVIDIA A10G GPU, driver version, memory usage.

    If you don’t see it → you launched CPU-only instance (terminate & retry).

7️⃣ Activate a Conda Environment

The DLAMI comes with Conda preinstalled.

    conda env list               # list prebuilt envs
    conda activate pytorch       # activate PyTorch-ready env

✅ Why: Saves you from manually installing CUDA & drivers.
8️⃣ Run a PyTorch GPU Test

    python - <<'PY'
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.rand((2048,2048), device='cuda')
        y = x @ x
        print("Matmul OK:", y.shape)
    PY

✅ Expected output:

    CUDA available: True

    GPU name (e.g., NVIDIA A10G)

    Matmul OK: torch.Size([2048, 2048])

9️⃣ (Optional) JupyterLab on EC2

    Install if missing:

        pip install jupyterlab

    Launch on VM:

        jupyter lab --no-browser --port 8888

    On local machine, forward port:

        ssh -i ~/.ssh/ai-keypair.pem -L 8888:localhost:8888 ubuntu@<INSTANCE_PUBLIC_IP>

    Open http://localhost:8888 → run GPU-enabled notebooks.

🔟 Cleanup (to avoid charges)

    In AWS Console, stop instance if you’ll reuse it.

    Or terminate if finished.

    Delete unused volumes & security groups.
```

### 21. 20. Hands-On with Google Cloud GPU Instances
- Goal: Launch a Google Cloud VM with a GPU, install drivers + PyTorch, and verify GPU access.
- Time: ~60–90 minutes
- Cost: A few dollars depending on GPU type (stop VM after use).
```
1️⃣ Prerequisites

    A Google Cloud account with billing enabled.

    A new project created in the GCP Console.

    gcloud CLI installed locally (optional, for advanced users).

2️⃣ Enable Compute Engine API

    In the Console, go to APIs & Services → Enable APIs and Services.

    Search and enable Compute Engine API.
    ✅ Why: You can’t create VMs without it.

3️⃣ Request GPU Quotas

    In Console → IAM & Admin → Quotas.

    Filter by GPUs in your target region (e.g., us-central1).

    Request at least 1 GPU (L4 = g2 instances, A100 = a2 instances).
    ✅ Why: GPU usage is disabled by default in new accounts.

4️⃣ Create a VM with GPU

    Console → Compute Engine → VM instances → Create Instance.

    Name: ai-gcp-lab.

    Region/Zone: pick one with available GPU quota.

    Machine type:

        g2-standard-8 (1×L4 GPU, cost-effective), OR

        a2-highgpu-1g (1×A100 GPU, high performance).

    CPU platform: leave default.

    GPU: click CPU platform and GPU → Add GPU → NVIDIA L4 or A100.

    Boot disk: Ubuntu 22.04 LTS, at least 100 GB.

    Firewall: allow SSH; (HTTP/HTTPS optional if serving APIs).

    Click Create.

5️⃣ Install NVIDIA Drivers

    Once VM is running, SSH in from Console:

        gcloud compute ssh ai-gcp-lab --zone=us-central1-a

    Run one-click GPU driver installer:

        sudo apt-get update
        sudo apt-get install -y ubuntu-drivers-common
        sudo ubuntu-drivers autoinstall
        sudo reboot

✅ Why: Ensures GPU is visible to frameworks like PyTorch.
6️⃣ Verify GPU

After reboot, reconnect via SSH:

    nvidia-smi

Expected: Table showing L4 or A100, driver version, and usage.

    If missing → driver not installed or wrong VM type.

7️⃣ Install Conda & Python Environment

    # Download and install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
     
    # Create and activate environment
    conda create -n ai python=3.10 -y
    conda activate ai

✅ Why: Isolated env for ML dependencies.
8️⃣ Install PyTorch with GPU Support

    For CUDA 12.1 runtime (works with L4/A100):

    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

9️⃣ Run a PyTorch GPU Test

    python - <<'PY'
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.rand((2048,2048), device='cuda')
        y = x @ x
        print("Matmul OK:", y.shape)
    PY

✅ Expected:

    CUDA available: True

    GPU name (e.g., “NVIDIA L4” / “A100”)

    Matmul OK: torch.Size([2048, 2048])

🔟 (Optional) JupyterLab Setup

    pip install jupyterlab
    jupyter lab --no-browser --port 8888

On your local machine, tunnel to the VM:

    gcloud compute ssh ai-gcp-lab --zone=us-central1-a -- -L 8888:localhost:8888

Then open → http://localhost:8888.
1️⃣1️⃣ Cleanup (Avoid Charges)

    In Console → Stop the VM if you plan to reuse.

    Or Delete VM, attached disk, and static IP.

    Always shut down when not in use.
```

### 22. 21. Lab – Compare Cost & Performance Across Clouds
- Goal: Deploy GPU instances across AWS, Google Cloud, and Azure, run a simple PyTorch benchmark, and compare cost vs performance.
- Time: 90–120 minutes
- Cost: A few dollars per provider (use spot/preemptible instances where possible, terminate after use).
```
1️⃣ Prerequisites

    Active accounts on AWS, GCP, and Azure (all with billing enabled).

    Quotas for at least 1 GPU VM in each cloud (AWS g5.xlarge, GCP g2-standard-8, Azure NCas T4_v3).

    SSH access working (previous labs cover setup).

    A shared dataset or script for consistency.

2️⃣ Define the Benchmark Task

We’ll use a matrix multiplication stress test in PyTorch.

    import torch, time
     
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand((10000, 10000), device=device)
     
    # Warmup
    y = x @ x
     
    # Benchmark
    start = time.time()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    end = time.time()
     
    print(f"Device: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")
    print("Time taken:", round(end-start, 3), "seconds")

✅ Why: Matrix multiplication is core to deep learning training (dense linear algebra).
3️⃣ Launch Instances Across Clouds

A) AWS (g5.xlarge)

    AMI: Deep Learning AMI (Ubuntu 22.04).

    GPU: 1×A10G.

    Setup: conda activate pytorch → run benchmark.

B) Google Cloud (g2-standard-8)

    GPU: 1×L4 GPU.

    OS: Ubuntu 22.04, install drivers + PyTorch (pip install torch --index-url https://download.pytorch.org/whl/cu121).

    Run benchmark.

C) Azure (NCas T4_v3, Standard_NC4as_T4_v3)

    GPU: 1×T4 GPU.

    Image: Azure ML DLVM Ubuntu (includes CUDA + PyTorch).

    Run benchmark.

4️⃣ Record Performance Results

For each provider, capture:

    GPU model (from nvidia-smi)

    Benchmark runtime (seconds)

    PyTorch version used

Example results table:

Cloud GPU Time (10 matmuls) Cost/hr (on-demand) AWS A10G 12.4 sec $1.20/hr GCP L4 10.8 sec $0.95/hr Azure T4 17.6 sec $0.90/hr
5️⃣ Compare Costs

Check on-demand pricing from each provider:

    AWS g5.xlarge: ~$1.20/hr

    GCP g2-standard-8: ~$0.95/hr

    Azure NCas T4_v3: ~$0.90/hr

✅ Cost varies by region and discount type (spot/preemptible).
6️⃣ Analysis

    Performance per dollar = (benchmark runtime ÷ cost/hr).

    Which GPU gives best absolute performance?

    Which gives best value for money?

    Consider reliability of spot vs on-demand.

7️⃣ Deliverables

    Screenshots of nvidia-smi for each cloud instance.

    Benchmark script output (time taken).

    A completed comparison table.

    Short written reflection:

        “Which provider gave me the best performance?”

        “Which provider gave me the best value?”

        “Which one would I choose for a large AI project and why?”

8️⃣ Cleanup (Critical ⚠️)

    Stop or terminate all 3 GPU instances.

    Delete any persistent disks or IPs.

    Verify billing dashboards show no active charges.

✅ By completing this lab, learners will:

    Understand real differences in GPU performance across clouds.

    Learn how to evaluate performance vs cost tradeoffs.

    Build intuition for choosing infra in real-world AI projects.
```

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
- Goal: Package a simple PyTorch model inside a Docker container, expose it as an API, and run predictions.
- Time: ~60–90 minutes
- Cost: Free (local Docker) or minimal (cloud VM with Docker + GPU).
```
1️⃣ Prerequisites

    Docker installed (docker --version to confirm).

    NVIDIA Container Toolkit (if using GPU).

    PyTorch installed locally (for saving model).

2️⃣ Step 1 – Train and Save a Model

Create train_model.py:

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
     
    # Simple pretrained model (ResNet18)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes
     
    # Fake training: just save weights for demo
    torch.save(model, "model.pt")
    print("✅ Model saved as model.pt")

Run once locally:

    python train_model.py

✅ Creates model.pt for deployment.
3️⃣ Step 2 – Build the FastAPI Inference App

Create app.py:

    from fastapi import FastAPI, UploadFile
    from PIL import Image
    import torch
    import torchvision.transforms as T
     
    app = FastAPI()
     
    # Load model
    model = torch.load("model.pt", map_location="cpu")
    model.eval()
     
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])
     
    @app.post("/predict")
    async def predict(file: UploadFile):
        img = Image.open(file.file).convert("RGB")
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            y = model(x)
        pred = y.argmax().item()
        return {"prediction": int(pred)}

✅ Minimal API for predictions.
4️⃣ Step 3 – Define Dependencies

Create requirements.txt:

    fastapi==0.103.0
    uvicorn==0.23.2
    torch==2.1.0
    torchvision==0.16.0
    pillow==10.0.0

5️⃣ Step 4 – Create Dockerfile

    FROM python:3.10-slim
     
    # Set working directory
    WORKDIR /app
     
    # Install system deps
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
     
    # Copy requirements and install
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
     
    # Copy model + code
    COPY model.pt app.py ./
     
    # Expose port
    EXPOSE 8000
     
    # Run FastAPI app
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

6️⃣ Step 5 – Build Docker Image

    docker build -t pytorch-api:latest .

✅ Creates image pytorch-api:latest.

Check with:

    docker images

7️⃣ Step 6 – Run the Container

    docker run -it -p 8000:8000 pytorch-api:latest

✅ Starts FastAPI server at http://localhost:8000/docs.
8️⃣ Step 7 – Test API

Open browser:

    Go to http://localhost:8000/docs

    Try /predict endpoint → upload an image (e.g., cat.jpg).

    Response:

    {"prediction": 3}

Or test via curl:

    curl -X POST "http://localhost:8000/predict" \
      -F "file=@cat.jpg"

9️⃣ (Optional) GPU Support

Run container with GPU:

    docker run --gpus all -it -p 8000:8000 pytorch-api:latest

✅ Requires NVIDIA toolkit installed.
✅ Learning Outcomes

    Build + save PyTorch model for inference

    Package model + API inside a container

    Run reproducible AI inference anywhere

    Understand containerized AI app workflow
```

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
- Goal: Run a PyTorch model API on Kubernetes using Minikube.
- Time: ~90 minutes
- Requirements: Docker, Minikube, kubectl installed.
```
1️⃣ Start Minikube

    minikube start --memory=4096 --cpus=4

    Allocates resources for the cluster

    Check cluster status:

    kubectl get nodes

✅ You should see 1 ready node.
2️⃣ Build a Model API Container

Re-use the FastAPI PyTorch app from Day 28 (app.py, model.pt, requirements.txt, Dockerfile).

Build image inside Minikube’s Docker:

    eval $(minikube docker-env)
    docker build -t ai-model:latest .
    docker images | grep ai-model

✅ Confirms the image is available to Minikube.
3️⃣ Create Deployment YAML

deployment.yaml:

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: ai-model-deployment
    spec:
      replicas: 2
      selector:
        matchLabels:
          app: ai-model
      template:
        metadata:
          labels:
            app: ai-model
        spec:
          containers:
          - name: ai-model
            image: ai-model:latest
            ports:
            - containerPort: 8000

    Runs 2 replicas of the model API

    Ensures self-healing + scalability

Apply it:

    kubectl apply -f deployment.yaml
    kubectl get pods

✅ Pods should transition to Running.
4️⃣ Expose Service

service.yaml:

    apiVersion: v1
    kind: Service
    metadata:
      name: ai-model-service
    spec:
      selector:
        app: ai-model
      ports:
      - protocol: TCP
        port: 80
        targetPort: 8000
      type: NodePort

Apply it:

    kubectl apply -f service.yaml
    kubectl get svc

✅ You’ll see a NodePort assigned (e.g., :30080).
5️⃣ Access the API

Get service URL:

    minikube service ai-model-service --url

Example: http://127.0.0.1:30080

Open in browser → /docs to see FastAPI Swagger UI.
✅ Try /predict with an image file.
6️⃣ Scale Deployment

    kubectl scale deployment ai-model-deployment --replicas=4
    kubectl get pods

✅ You should see 4 pods running your API.
7️⃣ Inspect Logs

    kubectl logs <pod-name>

    Shows FastAPI server logs

    Useful for debugging inference requests

8️⃣ Cleanup

    kubectl delete -f service.yaml
    kubectl delete -f deployment.yaml
    minikube stop

✅ Frees resources after the lab.
🎯 Learning Outcomes

    Deploy a containerized PyTorch model on Kubernetes

    Expose it via a Service for user access

    Scale replicas with kubectl scale

    Gain first hands-on experience with AI on Kubernetes
```

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
- Goal: Create a pipeline that ingests raw data (CSV), stores it in object storage, streams updates through Kafka (or Pub/Sub), and lands curated data into a database for AI use.
- Time: ~90–120 minutes
- Tools: Python, Docker, Kafka (or GCP Pub/Sub), PostgreSQL, S3/MinIO
```
1️⃣ Set Up the Environment

    Install Docker & Docker Compose

    Clone starter repo (or create working dir):

    mkdir ai-ingestion-lab && cd ai-ingestion-lab

    Services we’ll run:

        MinIO (S3-compatible storage)

        Kafka broker

        PostgreSQL database

2️⃣ Run Infra Services with Docker Compose

docker-compose.yml:

    version: "3.9"
    services:
      minio:
        image: minio/minio
        command: server /data
        environment:
          MINIO_ROOT_USER: admin
          MINIO_ROOT_PASSWORD: password
        ports:
          - "9000:9000"
          - "9001:9001"
     
      kafka:
        image: bitnami/kafka:latest
        environment:
          KAFKA_ENABLE_KRAFT: yes
          KAFKA_CFG_LISTENERS: PLAINTEXT://:9092
          KAFKA_CFG_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
          KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE: "true"
        ports:
          - "9092:9092"
     
      postgres:
        image: postgres:14
        environment:
          POSTGRES_USER: aiuser
          POSTGRES_PASSWORD: aipass
          POSTGRES_DB: aidb
        ports:
          - "5432:5432"

Start services:

    docker-compose up -d

✅ Now you have local object storage, a Kafka broker, and a database running.
3️⃣ Ingest Raw Data into Object Storage

Python script ingest_to_minio.py:

    from minio import Minio
    import os
     
    client = Minio(
        "localhost:9000",
        access_key="admin",
        secret_key="password",
        secure=False
    )
     
    # Ensure bucket exists
    bucket = "rawdata"
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
     
    # Upload file
    client.fput_object(bucket, "transactions.csv", "transactions.csv")
    print("✅ Uploaded transactions.csv to MinIO")

Run:

    pip install minio
    python ingest_to_minio.py

✅ Raw CSV now stored in MinIO (simulating S3).
4️⃣ Stream Updates via Kafka

Python producer kafka_producer.py:

    from kafka import KafkaProducer
    import json, time, random
     
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
     
    for i in range(10):
        record = {"user_id": i, "amount": random.randint(10,1000)}
        producer.send("transactions", record)
        print("Produced:", record)
        time.sleep(1)
     
    producer.flush()

Run:

    pip install kafka-python
    python kafka_producer.py

✅ Messages streaming into topic transactions.
5️⃣ Consume and Store in Database

Consumer kafka_consumer.py:

    from kafka import KafkaConsumer
    import json, psycopg2
     
    consumer = KafkaConsumer(
        "transactions",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )
     
    conn = psycopg2.connect(
        dbname="aidb", user="aiuser", password="aipass", host="localhost", port=5432
    )
    cur = conn.cursor()
     
    cur.execute("CREATE TABLE IF NOT EXISTS transactions (user_id INT, amount INT);")
     
    for msg in consumer:
        record = msg.value
        cur.execute("INSERT INTO transactions (user_id, amount) VALUES (%s, %s)",
                    (record["user_id"], record["amount"]))
        conn.commit()
        print("Inserted:", record)

Run:

    pip install psycopg2 kafka-python
    python kafka_consumer.py

✅ Streaming messages now land into PostgreSQL.
6️⃣ Verify Pipeline End-to-End

    Check object storage: open http://localhost:9001 (MinIO console).

    Inspect Kafka messages: logs from producer/consumer.

    Query database:

    docker exec -it <postgres-container> psql -U aiuser -d aidb -c "SELECT * FROM transactions;"

✅ You should see streamed data in the DB table.
7️⃣ Cleanup

    docker-compose down -v

✅ Stops services & removes volumes.
🎯 Learning Outcomes

    Store raw datasets in object storage (MinIO/S3)

    Stream data into pipelines with Kafka

    Land curated data into a relational DB

    Understand data ingestion architecture for AI pipelines
```

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
- Goal: Train and run a simple deep learning model on a GPU using CUDA.
- Time: ~60–90 minutes
- Tools: Python, PyTorch, CUDA-enabled GPU, nvidia-smi
```
1️⃣ Verify GPU and CUDA Availability

Check GPU devices:

    nvidia-smi

    Shows GPU type (e.g., A100, RTX 3090) and memory usage.

    Confirms drivers are working.

Check CUDA in PyTorch:

    import torch
    print(torch.cuda.is_available())   # Should be True
    print(torch.cuda.device_count())   # Number of GPUs
    print(torch.cuda.get_device_name(0))  # GPU name

✅ Output should confirm CUDA-enabled GPU is available.
2️⃣ Load a Dataset

We’ll use MNIST (handwritten digits) for speed.

    import torch
    import torchvision
    import torchvision.transforms as transforms
     
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

✅ Dataset prepared for training.
3️⃣ Define a Simple Neural Network

    import torch.nn as nn
    import torch.nn.functional as F
     
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
     
        def forward(self, x):
            x = x.view(-1, 28*28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
     
    model = Net()

✅ Basic 3-layer MLP defined.
4️⃣ Move Model to GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

✅ Model is now using GPU if available.
5️⃣ Define Optimizer & Loss Function

    import torch.optim as optim
     
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

6️⃣ Training Loop on GPU

    for epoch in range(2):  # Run 2 epochs
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
     
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
     
            running_loss += loss.item()
     
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

✅ Model trains on GPU with CUDA acceleration.
7️⃣ Monitor GPU Usage During Training

Run in a separate terminal:

    watch -n 1 nvidia-smi

    Check GPU utilization %, memory usage, and processes.

    Confirms training jobs are actively using CUDA.

8️⃣ Test Model Inference on GPU

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
     
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
     
    print(f"✅ Accuracy on test data: {100 * correct / total:.2f}%")

✅ Model makes predictions on GPU, improving speed.
9️⃣ Experiment: CPU vs GPU Speed

    Run training with device = "cpu" and note time.

    Run training with device = "cuda" and compare.

    Expect 2–10x speedup depending on GPU.

🔟 Cleanup

If using cloud (AWS, GCP, Colab):

    Stop instances to save cost.

    Clear dataset if needed.

🎯 Learning Outcomes

    Verify CUDA + GPU setup for AI workloads

    Train and test a PyTorch model on GPU

    Compare CPU vs GPU training performance

    Monitor GPU utilization in real time
```

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
- Goal: Train ResNet-18 on the CIFAR-10 dataset using PyTorch Distributed Data Parallel (DDP) across multiple GPUs.
- Time: ~90 minutes
- Tools: Python, PyTorch, CUDA-enabled system with 2+ GPUs
```
1️⃣ Verify Environment

Check GPU availability:

    nvidia-smi

    Confirm at least 2 GPUs.

Check PyTorch with CUDA:

    import torch
    print(torch.cuda.device_count())   # should be >= 2
    print(torch.cuda.is_available())   # should be True

2️⃣ Setup CIFAR-10 Dataset

    import torch
    import torchvision
    import torchvision.transforms as transforms
     
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
     
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

3️⃣ Define ResNet-18 Model

    import torchvision.models as models
    import torch.nn as nn
     
    def build_model():
        model = models.resnet18(weights=None, num_classes=10)
        return model

4️⃣ DDP Training Script

Create file train_ddp.py:

    import os, torch, torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn, torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models
    from torch.nn.parallel import DistributedDataParallel as DDP
     
    def setup(rank, world_size):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
     
    def cleanup():
        dist.destroy_process_group()
     
    def train(rank, world_size):
        setup(rank, world_size)
     
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=train_sampler)
     
        model = models.resnet18(weights=None, num_classes=10).to(rank)
        model = DDP(model, device_ids=[rank])
     
        criterion = nn.CrossEntropyLoss().to(rank)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
     
        for epoch in range(2):  # keep short for demo
            train_sampler.set_epoch(epoch)
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(rank), labels.to(rank)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"[GPU {rank}] Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")
     
        cleanup()
     
    def main():
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
     
    if __name__ == "__main__":
        main()

5️⃣ Launch Distributed Training

    python -m torch.distributed.run --nproc_per_node=2 train_ddp.py

    --nproc_per_node=2 → runs across 2 GPUs (adjust as needed).

    Each GPU runs its own process.

✅ Output should show parallel loss logs per GPU.
6️⃣ Monitor GPU Utilization

Run in separate terminal:

    watch -n 1 nvidia-smi

    See multiple processes using GPUs.

    Confirms GPUs are busy training in parallel.

7️⃣ Evaluate Trained Model

(Optional) Save checkpoint in script:

    if rank == 0:  # only one process saves
        torch.save(model.state_dict(), "resnet_ddp.pth")

Load for testing:

    model = models.resnet18(weights=None, num_classes=10)
    model.load_state_dict(torch.load("resnet_ddp.pth"))
    model.eval()

8️⃣ Cleanup

    Kill all training processes if needed:

    pkill -f train_ddp.py

    Free GPU memory after run.

🎯 Learning Outcomes

    Understand how to run DDP with PyTorch

    Train ResNet across multiple GPUs efficiently

    Monitor GPU scaling & utilization

    Learn practical multi-GPU AI infra skills
```

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
- Goal: Learn how to log parameters, metrics, and artifacts of a PyTorch model training run using MLflow Tracking.
- Time: ~60–90 minutes
- Tools: Python, PyTorch, MLflow, CUDA-enabled GPU (optional)
```
1️⃣ Install and Launch MLflow

Install MLflow:

    pip install mlflow[extras]

Run MLflow tracking server locally:

    mlflow ui

    Default: http://127.0.0.1:5000

    Web UI → experiment dashboard

2️⃣ Verify Environment

    import mlflow
    print("MLflow version:", mlflow.__version__)

✅ Confirms MLflow installed and available.
3️⃣ Load Dataset (MNIST Example)

    import torch, torchvision
    import torchvision.transforms as transforms
     
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

4️⃣ Define Model

    import torch.nn as nn
    import torch.nn.functional as F
     
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
     
        def forward(self, x):
            x = x.view(-1, 28*28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

5️⃣ Train with MLflow Logging

    import torch.optim as optim
    import mlflow.pytorch
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
     
    with mlflow.start_run():
        mlflow.log_param("lr", 0.001)
        mlflow.log_param("batch_size", 64)
     
        for epoch in range(2):  
            running_loss = 0.0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
     
            avg_loss = running_loss / len(trainloader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")
     
        mlflow.pytorch.log_model(model, "model")

✅ Logs:

    Params → learning rate, batch size

    Metrics → loss per epoch

    Artifacts → trained model

6️⃣ Check Results in MLflow UI

    Open: http://127.0.0.1:5000

    Explore experiment → see params, metrics, model files

    Compare multiple runs side by side

7️⃣ Log Artifacts (Optional)

    import matplotlib.pyplot as plt
     
    # Example artifact: loss curve
    losses = [0.9, 0.5, 0.3]
    plt.plot(losses)
    plt.savefig("loss_curve.png")
    mlflow.log_artifact("loss_curve.png")

✅ Artifact uploaded to MLflow → viewable in UI.
8️⃣ Compare Multiple Runs

    Change hyperparams (e.g., learning rate = 0.01 vs 0.001)

    Run training again

    Compare results in MLflow dashboard

    Identify best-performing config

9️⃣ Save and Load Model from MLflow

    model_uri = "runs:/{run_id}/model"
    loaded_model = mlflow.pytorch.load_model(model_uri)

    Replace {run_id} with your experiment’s run ID

    Test loaded model → confirms reproducibility

🔟 Cleanup

Stop MLflow server:

    CTRL+C

(Optional) Clear logs:

    rm -rf mlruns/


🎯 Learning Outcomes

    Set up & run MLflow tracking server

    Log params, metrics, and artifacts during training

    Compare runs to pick best configs

    Save and load models with MLflow
```

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
- Goal: Create a CI/CD pipeline that trains a PyTorch model, containerizes it with Docker, and deploys it via GitHub Actions (or GitLab/Jenkins alternative).
- Time: ~90–120 minutes
- Tools: GitHub Actions (or GitLab/Jenkins), Docker Hub, Kubernetes/Heroku (for deployment)
```
1️⃣ Prepare Repository

    Create new repo (GitHub or GitLab).

    Add files:

        train.py → trains & saves model

        app.py → Flask/FastAPI inference service

        Dockerfile → container definition

        .github/workflows/cicd.yml → pipeline config

✅ Repo structured for ML + deployment.
2️⃣ Train a Simple Model (train.py)

    import torch, torchvision, torchvision.transforms as transforms
    import torch.nn as nn, torch.optim as optim
     
    trainset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
     
    class Net(nn.Module):
        def __init__(self): super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(28*28,128), nn.Linear(128,64), nn.Linear(64,10)
        def forward(self,x): x=x.view(-1,28*28); return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
     
    model = Net()
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001)
     
    for epoch in range(1):
        for images, labels in trainloader:
            optimizer.zero_grad(); outputs=model(images)
            loss=criterion(outputs, labels); loss.backward(); optimizer.step()
     
    torch.save(model.state_dict(), "model.pth")
    print("✅ Model trained and saved")

3️⃣ Inference API (app.py)

    from flask import Flask, request, jsonify
    import torch, torch.nn as nn
     
    app = Flask(__name__)
     
    class Net(nn.Module):
        def __init__(self): super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(28*28,128), nn.Linear(128,64), nn.Linear(64,10)
        def forward(self,x): x=x.view(-1,28*28); return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
     
    model = Net(); model.load_state_dict(torch.load("model.pth")); model.eval()
     
    @app.route("/predict", methods=["POST"])
    def predict():
        data = torch.tensor(request.json["input"])
        output = model(data.float())
        _, predicted = torch.max(output, 1)
        return jsonify({"prediction": predicted.item()})

4️⃣ Dockerfile

    FROM python:3.9-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["python", "app.py"]

✅ Containerizes training + inference service.
5️⃣ GitHub Actions Workflow (.github/workflows/cicd.yml)

    name: ML CI/CD Pipeline
    on: [push]
     
    jobs:
      build-deploy:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v3
     
          - name: Set up Python
            uses: actions/setup-python@v4
            with:
              python-version: "3.9"
     
          - name: Install dependencies
            run: pip install -r requirements.txt
     
          - name: Train Model
            run: python train.py
     
          - name: Build Docker image
            run: docker build -t myrepo/ai-model:latest .
     
          - name: Login to Docker Hub
            uses: docker/login-action@v2
            with:
              username: ${{ secrets.DOCKER_USER }}
              password: ${{ secrets.DOCKER_PASS }}
     
          - name: Push Docker Image
            run: docker push myrepo/ai-model:latest

6️⃣ Deploy to Kubernetes (Optional)

Add deploy step to workflow:

          - name: Deploy to K8s
            run: |
              kubectl set image deployment/ai-app ai-app=myrepo/ai-model:latest

✅ Updates live deployment with new model.
7️⃣ Validate CI/CD Pipeline

    Push changes to repo → pipeline triggers automatically

    Training, Docker build, and push to registry

    (Optional) Auto-deploys to Kubernetes or Heroku

8️⃣ Monitor Pipeline Runs

    GitHub Actions tab → see logs of each stage

    Validate model artifacts + image build success

    Check deployed service endpoint /predict

9️⃣ Rollback if Needed

    Keep older Docker image tags

    Roll back in K8s:

    kubectl rollout undo deployment ai-app

✅ Ensures safe recovery from bad deployment.
🎯 Learning Outcomes

    Automate training + deployment pipeline with CI/CD

    Use Docker + GitHub Actions for ML apps

    Deploy trained models seamlessly to production

    Learn rollback strategy for safe ML infra
```

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
| storage | shared or ephemeral | persistent per pod|
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
- Goal: Stand up MLflow Tracking Server on Kubernetes with:
  - PostgreSQL as the backend store (metrics/params/runs)
  - MinIO (S3-compatible) as the artifact store
  - Persistent volumes, Service, and Ingress for access
- Time: ~90–120 minutes
- You need: A K8s cluster (kind/Minikube/cloud), kubectl, helm, and docker (if you build images).
- Namespace: mlops
```
1) Create Namespace & StorageClass (if needed)

    kubectl create namespace mlops
    # If your cluster lacks a default StorageClass, set or create one (example for kind/minikube usually not needed).
    kubectl get storageclass

2) Install MinIO (Artifact Store)

Option A – Helm (recommended):

    helm repo add minio https://charts.min.io/
    helm repo update
     
    helm install minio minio/minio \
      --namespace mlops \
      --set rootUser=admin \
      --set rootPassword=admin12345 \
      --set resources.requests.memory=256Mi \
      --set mode=standalone \
      --set replicas=1

Port-forward to access MinIO console (temporarily):

    kubectl -n mlops port-forward svc/minio 9000:9000 9001:9001

    Console: http://localhost:9001 (user: admin, pass: admin12345)

Create a bucket for MLflow artifacts (e.g., mlflow-artifacts). You can also do this from CLI later.
3) Install PostgreSQL (Backend Store)

Option A – Helm (Bitnami):

    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm install pg bitnami/postgresql \
      --namespace mlops \
      --set global.postgresql.auth.postgresPassword=pgpass \
      --set global.postgresql.auth.username=mlflow \
      --set global.postgresql.auth.password=mlflowpass \
      --set global.postgresql.auth.database=mlflowdb \
      --set primary.persistence.size=5Gi

Fetch connection info:

    PG_HOST=$(kubectl -n mlops get svc pg-postgresql -o jsonpath='{.spec.clusterIP}')
    echo $PG_HOST

4) Create Secrets for MLflow

We’ll store DB creds and MinIO keys in a single secret.

    kubectl -n mlops create secret generic mlflow-secrets \
      --from-literal=BACKEND_URI="postgresql://mlflow:mlflowpass@pg-postgresql.mlops.svc.cluster.local:5432/mlflowdb" \
      --from-literal=ARTIFACT_URI="s3://mlflow-artifacts" \
      --from-literal=AWS_ACCESS_KEY_ID="admin" \
      --from-literal=AWS_SECRET_ACCESS_KEY="admin12345" \
      --from-literal=MLFLOW_TRACKING_USERNAME="admin" \
      --from-literal=MLFLOW_TRACKING_PASSWORD="changeme"

    Note: We’ll use MinIO’s S3 endpoint env var to point MLflow’s boto client.

5) (Optional) Create the Bucket via Job (if not created in console)

    apiVersion: batch/v1
    kind: Job
    metadata:
      name: mkbucket
      namespace: mlops
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
          - name: mc
            image: minio/mc:latest
            env:
            - name: MC_HOST_minio
              value: http://admin:admin12345@minio.mlops.svc.cluster.local:9000
            command: ["sh","-c"]
            args:
              - |
                mc ls minio || true
                mc mb -p minio/mlflow-artifacts || true

    kubectl apply -f mkbucket.yaml
    kubectl -n mlops logs job/mkbucket

6) Persistent Volume Claim for MLflow (optional but nice)

    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: mlflow-pvc
      namespace: mlops
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 5Gi

    kubectl apply -f mlflow-pvc.yaml

7) Deploy MLflow Tracking Server

Deployment + Service:

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: mlflow
      namespace: mlops
    spec:
      replicas: 1
      selector:
        matchLabels: { app: mlflow }
      template:
        metadata:
          labels: { app: mlflow }
        spec:
          containers:
          - name: mlflow
            image: ghcr.io/mlflow/mlflow:v2.14.1
            imagePullPolicy: IfNotPresent
            ports:
            - containerPort: 5000
            envFrom:
            - secretRef:
                name: mlflow-secrets
            env:
            - name: MLFLOW_S3_ENDPOINT_URL
              value: http://minio.mlops.svc.cluster.local:9000
            - name: AWS_DEFAULT_REGION
              value: us-east-1
            - name: MLFLOW_TRACKING_USERNAME
              valueFrom: { secretKeyRef: { name: mlflow-secrets, key: MLFLOW_TRACKING_USERNAME } }
            - name: MLFLOW_TRACKING_PASSWORD
              valueFrom: { secretKeyRef: { name: mlflow-secrets, key: MLFLOW_TRACKING_PASSWORD } }
            command: ["mlflow"]
            args:
              - server
              - "--host=0.0.0.0"
              - "--port=5000"
              - "--backend-store-uri=$(BACKEND_URI)"
              - "--default-artifact-root=$(ARTIFACT_URI)"
            volumeMounts:
            - name: mlflow-data
              mountPath: /mlflow
          volumes:
          - name: mlflow-data
            persistentVolumeClaim:
              claimName: mlflow-pvc
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: mlflow
      namespace: mlops
    spec:
      type: ClusterIP
      selector: { app: mlflow }
      ports:
      - name: http
        port: 5000
        targetPort: 5000

    kubectl apply -f mlflow-deploy.yaml
    kubectl -n mlops get pods,svc

8) Expose with Ingress (or Port-Forward)

Quick test via port-forward:

    kubectl -n mlops port-forward svc/mlflow 5000:5000
    # Visit: http://localhost:5000

Ingress (requires an ingress controller like NGINX):

    apiVersion: networking.k8s.io/v1
    kind: Ingress
    metadata:
      name: mlflow
      namespace: mlops
      annotations:
        nginx.ingress.kubernetes.io/auth-type: basic
        nginx.ingress.kubernetes.io/auth-secret: mlflow-basic-auth
    spec:
      rules:
      - host: mlflow.localtest.me
        http:
          paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: mlflow
                port:
                  number: 5000

Create a basic auth secret (optional hardening):

    # Create htpasswd file (requires apache2-utils or busybox)
    htpasswd -bc htpasswd admin changeme
    kubectl -n mlops create secret generic mlflow-basic-auth --from-file=auth=htpasswd
    kubectl apply -f mlflow-ingress.yaml
    # Update /etc/hosts if needed for mlflow.localtest.me -> ingress IP

9) Verify End-to-End

    Open MLflow UI (port-forward or Ingress URL).

    Create a quick client run (from laptop or a cluster pod):

    import mlflow, os
    os.environ["MLFLOW_TRACKING_USERNAME"]="admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"]="changeme"
    mlflow.set_tracking_uri("http://localhost:5000")  # or Ingress URL
     
    with mlflow.start_run():
        mlflow.log_param("lr", 0.001)
        mlflow.log_metric("loss", 0.42, step=1)
        with open("hello.txt","w") as f: f.write("artifact test")
        mlflow.log_artifact("hello.txt")

    In UI, confirm:

        Run visible with param/metric

        Artifact hello.txt stored in MinIO bucket (mlflow-artifacts)

10) (Optional) Secure the DB and MinIO

    Restrict MinIO via NetworkPolicy to only MLflow pod.

    Rotate MinIO and DB credentials; store in an external secrets manager.

    Enable TLS on Ingress and MinIO (cert-manager).

Troubleshooting

    Artifacts not saving? Check MLFLOW_S3_ENDPOINT_URL, bucket exists, and creds match.

    DB connection errors? Confirm BACKEND_URI host/port & service DNS.

    UI blank/timeout? Verify Service, port-forward/Ingress, and pod logs:

        kubectl -n mlops logs deploy/mlflow

    Permissions denied on bucket? Verify MinIO policies and access keys.

Cleanup

    helm -n mlops uninstall minio
    helm -n mlops uninstall pg
    kubectl delete ns mlops


🎯 Learning Outcomes

    Deploy MLflow Tracking with persistent stores on Kubernetes.

    Wire up PostgreSQL backend + S3/MinIO artifact store.

    Expose the service via Ingress and secure with basic auth.

    Validate end-to-end experiment logging from a client.
```

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
- Why resource monitoring?
  - Preventing resource waste
  - Early bottleneck detection
  - Efficient scaling
- Critical resources to monitor
  - Compute usage
    - Core utilization
    - CUDA memory allocation
    - Temperature and power draw
  - Memory usage
    - System RAM
    - VRAM usage
    - Memory leak detection
  - Storage & IO
    - Disk IOPS & throughput
    - Storage capacity usage
    - Cache hit ratios
  - Network traffic  
    - Inter-node bandwidth
    - Packet loss rates
    - Data transfer latency
- Essential monitoring tools
  - Prometheus
  - Grafana
  - Nvidia DCGM: Deep GPU health monitoring
- Common monitoring challenges
  - Metrics overlad
  - GPU complexity
  - Fragment visibility
  - Alert fatigue
- Monitoring Best Practices
  - Define clear KPIs
  - Implement tiered alerting
  - Build purpose built dashboards
  - Integrate with MLOps workflow

### 83. 82. Storage Cost Optimization Strategies
- Storage: the hidden cost driver in AI
  - Training datasets grow exponentially
  - Model checkpoints accumulate rapidly
  - Logs and debugging artifacts
  - Finished models require versioning and redundancy
- Common storage cost drivers
  - Excessive hot storage
  - Dataset redundancy
  - Inefficient versioning
  - Transfer fees
- Implement a tiered storage approach
  - Hot storage
    - Active datasets, current model training
    - High performance, low latency
    - NVME/SSD, high IOPS block storage
  - Warm stroage
    - Recent projects, validated datasets
    - Balanced performance
    - Standard SSD, object storage with frequent access tiers
  - Cold storage
    - Archival, compliance
    - Slow retrieval
    - Glacier, archive storage, tape systems
- Space-saving techniques
  - Lossless compression
  - Format-specific compression (jpeg->WEBP)
  - Delta storage (stores change only)
  - Content-based deduplication
- Best practices
  - Establish regular storage audits
  - Automate lifecycle management
  - Implement shared feature stores
  - Factor all storage costs into planning

### 84. 83. Multi-Tenant Cost Allocation in Teams
- The challenge of shared infrastructure
  - Opaque costs
  - Limited accountability
  - Resource contention
- Why cost allocation matters
  - Prevents Tragedy of the commons
  - Financial visibilty
  - Resource optimization
  - Business alignment
- Methods of cost allocation
  - Resource tagging & labeling
  - Quotas and resource limits
  - Usage-based metering
  - Cost transparency dashboards
- Tools and platforms for implementation
  - Kubernetes native controls
  - Kubecost
  - Cloud provider tools: AWS Cost Explorer, GCP Billing, Azure Cost Management
  - Custom BI solutions      
- Implementation challenges
  - Attributing shared resources
  - Workload variation
  - Multi-cloud complexity
  - Cost-benefit balance
- Best practices
  - Align with organization structure
  - Start with showback, then chargeback
  - Automate from day one
  - Create shared incentives

### 85. 84. Lab – Optimize Cloud AI Workload Costs
- Goal: Cut end-to-end training/inference costs by 40–80% while maintaining (or improving) throughput and accuracy.
- You’ll do: baseline → instrument → optimize (compute, storage, network, scheduling) → measure → report.
```
0) Prerequisites

    A cloud account (AWS or GCP or Azure).

    One GPU instance (on-demand) and permission to launch spot/preemptible.

    Docker + Kubernetes cluster (managed or self-hosted) or VM-only path.

    CLI: aws or gcloud or az; kubectl (if using K8s).

    Sample project: PyTorch ResNet-50 on CIFAR-10 (training) + FastAPI/Triton (inference).

    Tip: Keep your dataset/model fixed across runs so cost deltas are attributable to infra changes.

1) Design the Experiment

Define the baseline (row 2 in the Excel):

    Cloud/Region: (e.g., AWS us-east-1)

    Instance: 1× GPU (A10/T4/V100 class) on-demand

    Storage: local NVMe + object store (S3/GCS/Blob) in same region

    No autoscaling, no spot, FP32, batch size 128

KPIs to record (per run):

    Throughput/s, Latency (p50/p95), GPU util %, Time to train, Accuracy, Cost (compute/storage/network), TotalCostUSD, Savings vs baseline %.

Use the worksheet to log each run: one change per row.
2) Instrumentation & Visibility (Baseline + All Runs)
A. VM-level quick check

    # GPU + memory + power
    nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.total,pstate,power.draw --format=csv -l 5
    # Disk & network
    iostat -xz 5
    ifstat 5

B. Kubernetes (recommended)

    Prometheus + Grafana for metrics, NVIDIA DCGM Exporter for GPU.

    # Add DCGM exporter (Helm)
    helm repo add nvidia https://nvidia.github.io/dcgm-exporter
    helm install dcgm nvidia/dcgm-exporter -n monitoring --create-namespace

    Optional: Kubecost for $ visibility.

    helm repo add kubecost https://kubecost.github.io/cost-analyzer/
    helm install kubecost kubecost/cost-analyzer -n kubecost --create-namespace

C. Tag everything for cost attribution

    AWS: --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=Lab84},{Key=Owner,Value=Vivian}]'

    GCP: --labels=project=lab84,owner=vivian

    Azure: --tags Project=Lab84 Owner=Vivian

3) Baseline Run (On-Demand)
A. Launch (pick your cloud)

AWS (GPU on-demand example)

    aws ec2 run-instances \
      --image-id ami-... \
      --instance-type g5.2xlarge \
      --key-name yourkey \
      --count 1 \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200}}]' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=Lab84}]'

GCP (on-demand)

    gcloud compute instances create lab84-ondemand \
      --zone=us-central1-a --machine-type=a2-highgpu-1g \
      --accelerator=count=1,type=nvidia-tesla-a100 \
      --boot-disk-size=200GB --labels=project=lab84

Azure (on-demand)

    az vm create -g rg-lab84 -n lab84-ondemand \
      --image Ubuntu2204 --size Standard_NC4as_T4_v3 \
      --storage-sku Premium_LRS --tags Project=Lab84

B. Train (FP32, no optimizations yet)

    Use your standard ResNet-50 script, e.g., PyTorch CIFAR-10.

    Record: Throughput/s, GPU util %, Training time, Accuracy, Costs.

Fill row 2 (baseline) in the Excel.
4) Optimization Levers (Run one lever per row)
4.1 Spot/Preemptible Instances (+ Checkpointing)

Enable resilient training first:

PyTorch (automatic mixed precision optional later):

    # train.py (snippet)
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    start_step = load_checkpoint_if_exists(model, optimizer, scaler)  # your function
     
    for step, (x, y) in enumerate(loader, start=start_step):
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=False):  # keep False for pure baseline; set True in AMP step
            yhat = model(x.cuda(non_blocking=True))
            loss = criterion(yhat, y.cuda(non_blocking=True))
        loss.backward()
        optimizer.step()
        if step % CKPT_EVERY == 0:
            save_checkpoint(model, optimizer, scaler, step)  # lightweight, atomic

Launch Spot/Preemptible:

    AWS Spot:

    aws ec2 run-instances \
      --instance-market-options 'MarketType=spot' \
      --instance-type g5.2xlarge --image-id ami-... \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=Lab84}]'

    GCP Spot / Preemptible:

    gcloud compute instances create lab84-spot \
      --zone=us-central1-a --machine-type=a2-highgpu-1g \
      --provisioning-model=SPOT --boot-disk-size=200GB \
      --labels=project=lab84

    Azure Spot:

    az vm create -g rg-lab84 -n lab84-spot \
      --image Ubuntu2204 --size Standard_NC4as_T4_v3 \
      --priority Spot --max-price -1 --eviction-policy Deallocate \
      --tags Project=Lab84

Measure & log a new row (expect large compute $ drop).
4.2 Autoscaling (K8s)

    Cluster Autoscaler enabled on your managed cluster.

    HPA for inference pods:

    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata: { name: inference-hpa, namespace: ai }
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: inference-api
      minReplicas: 2
      maxReplicas: 50
      metrics:
      - type: Resource
        resource:
          name: cpu
          target: { type: Utilization, averageUtilization: 70 }

Log costs before/after a load test.
4.3 Mixed Precision (AMP) for Training

Turn autocast(enabled=True) + GradScaler:

    with autocast(True):
        yhat = model(x.cuda(non_blocking=True))
        loss = criterion(yhat, y.cuda(non_blocking=True))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

Usually boosts throughput and reduces time → lower compute cost.
4.4 Batch Size & Gradient Accumulation

Increase BatchSize and add GradAccumSteps to keep memory in check.
Record throughput & accuracy; keep the best trade-off.
4.5 Right-Sizing & Utilization

    If GPU util < 50%, you’re probably I/O-bound.

        Cache dataset on local NVMe.

        Pin dataloader workers, enable async I/O.

    loader = DataLoader(ds, batch_size=B, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

4.6 Storage Lifecycle Policies

Move old checkpoints/logs to cold storage automatically.

AWS S3 (lifecycle JSON example):

    {
      "Rules": [{
        "ID": "MoveOlderArtifacts",
        "Filter": { "Prefix": "experiments/" },
        "Status": "Enabled",
        "Transitions": [{ "Days": 14, "StorageClass": "STANDARD_IA" },
                        { "Days": 45, "StorageClass": "GLACIER" }],
        "NoncurrentVersionTransitions": [{ "NoncurrentDays": 30, "StorageClass": "GLACIER" }]
      }]
    }

GCP and Azure offer similar policies—configure equivalent rules.
Log storage and egress cost changes in the sheet.
4.7 Data Locality (Avoid Cross-Region Egress)

Keep compute and object storage in the same region.
If you were cross-region, redo the run same-region and record the network $ drop.
4.8 Container & Runtime Optimizations

    Base images: use slim CUDA runtimes, remove build tools from runtime image.

    Enable torch.compile() (PyTorch 2) or TensorRT for inference if supported.

    # training (safe when model is stable)
    model = torch.compile(model)  # often improves throughput

4.9 Inference Cost Cuts (Triton)

Enable dynamic batching and model-level optimization.

config.pbtxt (snippet):

    max_batch_size: 64
    dynamic_batching { preferred_batch_size: [ 4, 8, 16, 32 ] max_queue_delay_microseconds: 1000 }
    instance_group [{ kind: KIND_GPU, count: 1 }]

Add a Redis cache layer for idempotent requests (e.g., embeddings).
4.10 Reservations/Commitments (Analysis Step)

Not a live change for the lab, but simulate:

    If a fixed baseline footprint runs 24/7 (e.g., inference), estimate Reserved Instances / Savings Plans (AWS) or Committed Use Discounts (GCP) and log the hypothetical monthly savings in the sheet’s notes for planning.

5) Kubernetes: Spot-Aware Scheduling (Optional but Powerful)

    Create node pools: on-demand and spot.

    Taint spot nodes: spot=true:NoSchedule.

    Tolerate in your training job and add PodDisruptionBudget.

    apiVersion: apps/v1
    kind: Deployment
    metadata: { name: trainer, namespace: ai }
    spec:
      replicas: 1
      template:
        metadata: { labels: { app: trainer } }
        spec:
          tolerations:
          - key: "spot" operator: "Equal" value: "true" effect: "NoSchedule"
          nodeSelector: { lifecycle: spot }
          containers:
          - name: trainer
            image: yourrepo/trainer:latest
            resources:
              limits: { nvidia.com/gpu: "1" }

This keeps stateful services on on-demand and ephemeral training on spot.
6) Run, Measure, Record (Repeat)

For each lever:

    Apply exactly one change.

    Run training/inference.

    Capture metrics & costs.

    Fill a new row in the Excel (it auto-sums TotalCost; Savings% compares to baseline).

    Keep short notes on what changed.

    Target sequence: Spot → AMP → Batch/GA → Data Locality → Lifecycle → Autoscale → Runtime/Compile → Inference batching/cache.

7) Analyze & Decide

    Plot TotalCostUSD vs Throughput/s to spot Pareto-optimal settings.

    Confirm that accuracy remains within your tolerance window.

    Promote the top configuration to your default pipeline.

What “Good” Looks Like (Typical Wins)

    Spot/Preemptible + checkpointing: 60–80% compute cost cut

    AMP + larger batches: 20–40% time cut at same accuracy

    Same-region storage + lifecycle: double-digit % storage/network savings

    Triton dynamic batching + cache: big inference $ drop with stable latency
```

## Section 14: Week 13: Networking for AI Systems

### 86. 85. Fundamentals of Data Center Networking
- Clos/Spine-Leaf Underlay: a scalable high-performance data center architecture, using two layers of switches (spines and leaves)
  - Core architecture
    - Leaf switches: connect directly to servers
    - Spine switches: interconnect all leaf switches
    - Layer-3 fabric with ECMP (Equal-Cost Multi-Path) routing
    - Typically 2-3 hops maximum b/w any two endpoints
  - Implementation considerations
    - Plan oversubscription ratio based on workload (1:1 for AI training)
    - Achieve modular scale without redesigning the fabric
    - Isolate faults to minimize blast radius
    - Ensure deterministic performance with consistent hop count
- L2/L3 segmentation & overlays
  - Layer 2: VLAns
  - Layer 3: VRFs
  - VXLAN + EVPN
  - Micro-segmentation    
- Transport and congestion control
  - Transport protocols
    - TCP
    - RDMA/RoCEv2
  - Congestion management
    - PFC (Priority Flow control): creates lossless traffic classes
    - ECN/RED: early congestion notification
    - DCQCN: data center quantized congestion notification
  - Optimization techniques
    - Jumbo frames: Reduce CPU/Interrupt overhead (9000 MTU)
    - Buffer tuning
    - End-to-end QoS
- Links & Interconnect Choices
  - Ethernet 
  - Infiniband
  - Cabling considerations
    - DAC: < 5m, lowest cost/power
    - AOC: < 30m, moderate cost
    - Optics: up to 10km_, highest cost
- AI workload patterns
  - Dominant communication pattern: All-Reduce
  - Toplogy considerations
    - Ring algorithm scale linearly (N)
    - Tree algorithms scale logarithmically (log N)
    - Mesh/all-to-all operations stress bisection
  - Framework implementation
    - NCCL
    - GLOO
    - MPI
    - HOROVOD
- Service networking & load balancing
  - Load balancing approaches
    - L4 (Transport)
    - L7 (Application)
  - In cluster networking
    - CNI: Container Network Interface
    - eBPF: Extended Berkeley Packet Filter for high-performance datapaths
    - Service Mesh: optional mTLS, traffic management, observability
  - Global traffic management
    - Anycast: same IP advertised from multiple locations
    - Ingress/Egress: smart traffic routing based an application semantics
    - Protection: rate limiters, circuit breakers for backend
- Observability and troubleshooting
  - Data collection methods
    - Streaming telemetry
    - sFlow/NetFlow/IPFIX
    - SNMP
  - Key performance indicators
    - Latency heatmaps
    - Packet drops
    - Queue occupancy
    - Buffer utilization
  - Troubleshooting toolkit
    - Ping/tracepath
    - iperf3
    - SPAN/ERSPAN
    - tcpdump/wireshark
- Security and resilience
  - Zero-Trust security
  - Link & Path resilience
  - Advanced Protection
- Design Best Practices
  - Keep the Underlay simple
  - Align end-to-end parameters
  - Separate traffic classes
  - Document everything

### 87. 86. Software Defined Networking (SDN) for AI
- Why SDN for AI?
  - AI traffic challenges
    - Many to many communicatins
    - Extremely latency-sensitive workloads
    - Constantly shifting resource demands
  - Static networks fall short
    - Can't adapt to rapid job churn
    - Unable to scale
    - Manual configuration create bottlenecks
  - SDN advantage
    - Centralized intent definition
    - Automated policy enforcement
    - Delivers higher throughput, lower jitter, better utilization
- SDN fundamentals
  - Architecture split
  - Southbound protocols
  - Network structure
  - Policy as code
- AI-aware traffic engineering
  - Flow intellgence
    - Distinguish high-volume "elephant" flows from frequent but small "mice" flows
    - Apply appropriate routing strategies based on flow classification
  - Optimization goal
    - Maximize bisection bandwidth for collective operations
    - Dynamically adjust path selection using ECMP baising or SRv6
    - Leverage congestion signals (ECN/DCQCN) to inform controller decisions
- QoS & class isolation
  - Training: high bandwidth, moderate latency
  - Inference: low bandwidth, extreme latency
  - Storage: high bandwidth, medium latency
  - Control: low bandwidth, high reliability
- Multi-tenant segmentation
  - Tenant isolation
  - Intent definition
  - Micro segmentation
- Kubernetes and SDN
  - CNI options
  - GitOps deployment
  - Topology awareness
- Tooling and ecosystem
  - Controllers: ACI, Apstra, CloudVision
  - Infrastructur as code: Terraform, ansible, and CI/CD pipelines
  - MLOps integration
  - Observability: Grafana dashboards, Hubble for visualization
- Security with SDN
  - Zero-trust architecture
  - Encryption capabilities
  - Automated protections
  - Change management
- Best practice
  - Build AI-ready networks
  - Architecturaly simplicity
  - Policy as code
  - Closed-loop operations
  - Business outcomes

### 88. 87. Infiniband and High-Speed Interconnects
- How to maximize bisection bandwidth and predicatability at scale
- IB vs ethernet
  - Infiniband 
    - Purpose built for HPC
    - HW based flow control
    - Deterministic low latency performance
    - Mature collective operations support
  - Ethernet + RoCEv2
    - Ubiquitous enterprise technology
    - Flexible deployment options
    - Requires tight QoS/congestion tuning
    - Broader ecosystem cokmpatibility
    - Typically lower HW costs
- IB generations & speeds
  - FDR: 56 Gbps
  - EDR: 100 Gbps
  - HDR: 200 Gbps
  - NDR: 400 Gpbs
  - XDR: 800 Gpbs
- RDMA fundamentals
  - Remote direct Memory Access
  - Zero copy data transfer b/w newtwork connected systems
  - Kernel bypass eliminates OS overhead
  - Minimal CPU utilization during transfers
  - Verbs API model with send/receive, RDMA read/write operations
- AI accelerants on IB
  - NCCL over IB
  - GPUDirect RDMA
  - In-network comkpute: SHARP style offlads
- Fabric topology and routing
  - Network topologies
    - Fat-TREE/Clos: non-blocking fabrics with full bisection bandwidth
    - Dragonfly: optimized for scale & cost with controlled oversubscription
  - Traffic management
    - Adaptive routing to dynamically dodge hot spots
    - ECMP-like path diversity for traffic distribution
    - QoS/Service Levels (SLs) to separate traffic classes
- HW building blocks
  - HCAs/NICs & DPUs
  - Switches
  - Cabling infrastructure
- Kubernetes integration
  - RDMA device plugin + SR-IOV
  - Container network interface with eBPF datapaths
- IB diagnotics
  - `ibdiagnet` for topology validation
  - `perfquery` for performance statistics
  - `ibstat` for port status
- Workload tests
  - `nccl-tests` (all_reduce_pef) for collective operations
  - `ib_write_bw` and `ib_send_lat` for baseline performance verification
- Best practices
  - Technology selection: IB or RoCE
  - Co-designed architecture
  - Quality of Service
  - Continuous validation

### 89. 88. Load Balancing for AI Inference
- Why load balancing for AI?
  - Inference traffic exhibits spiky, unpredictable patterns
  - GPU backends have complex batching and concurrency
- Load balancing layers and options
  - L4: TCP/UDP - Google Maglev, Linux IPVS, Anycast routing
  - L7: HTTP/HTTP2/gRPC - envoy proxy, NGINX, HAProxy
  - Global: cross-region
  - In-cluster: Kubernetes
- Core load balancing algorithms
  - Round robin/Weighted RR
  - Least request/EWMA: routes to the least busy servers
  - Consistent hashing: mapping based on key
  - Priority/weighted pools: direct traffic to preferred backends
- GPU-aware considerations
  - Expose concurrency metrics: route based on available GPU slots and queue depth
  - Coordinate with batching
  - Handle GPU heterogeneity
- Sticky vs stateless load balancing
  - Prefer statelss when possible
  - Use session affinity judiciously
  - Smart key selection
  - Plan for failures
- Multi-region patterns
  - Active/active: run full capacity in multple regions using anycast or latency-based DNS
  - Smart traffic steering: client latency, regional capacity, and infrastructure cost
  - Regional circuit breakers: implement regional isolation to contain failures, preventing cascading issues across global footprint
  - Failover planning

### 90. 89. Network Bottlenecks in Distributed Training
- Symptoms of network bottleneck
  - Low GPU utilization
  - Scaling plateaus
  - Step-time variance
  - Collective delays
- Topology and placement matter
  - Keep worker nodes within the same rack/reaf when possible
  - `nvidia-smi topo -m` to verify and align GPU-NIC with NUMA domains
  - Set CPU affinity for
    - GPU NUMA locality
    - NIC PCIe attachment
    - Memory controller access
- Transport tuning
  - Enable RDMA
  - Set MTU 9000 jumbo frame
  - Congestion control: configure ECN/DCQCN and use PFC on lossless traffic classes
  - Watch buffers
- NCCL/Horovod knobs  
  - Framework settings
    - PyTorch DDP: tune `bucket_cap_mb` (25-100MB), set `static_graph=True`
    - Horovd: Adjust `HOROVOD_FUSION_THRESHOLD`, `HOROVOD_CYCLE_TIME`
    - Use `no_sync()` for gradient accumulation
    - Overlap compute/communication (DDP default)
  - Environment variables
```bash
# interface selection
NCCL_SOCKET_IFNAME=ib0,eth1
NCCL_IB_HCA=mlx5_0,mlx5_1
# protocol settings
NCCL_IB_GID_INDEX=3 # RoCEv2
NCCL_PROTO=LL128 # small msgs
NCCL_NET_GDR_LEVEL=2 # enable GDR
```

### 91. 90. Security in Networked AI Systems
- AI security threat landscape
  - Data exfiltration & training data theft
  - Model endpoint abuse & scraping
  - Lateral movement in clusters
  - Supply-chain risks
  - DDoS & cost-exhaustion attacks
- Zero-trust principles
  - Never trust: always verify
  - Strong identity everywhere
  - Continuous policy enforcement
  - Assume breach
- Identity, authentication & authorization
  - Identity standards
    - OIDC/OAuth2 for human users
    - SPIFFE/PSIRE for workloads
    - Federated identity across clouds
  - Access controls
    - Short-lived credentials
    - Just-in-time (JIT) access
    - Break-glass emergency procedures
  - Permission models
    - RBAC for role-based permissions
    - ABAC for attribute-based controls
    - Fine-grained policy enforcement
- Best practices
  - Default-deny architecture
  - Response automation
  - Encryption and authentication
  - Tenant isolation
  - Continuous monitoring

### 92. 91. Lab – Configure Load Balancer for AI API
- Goal: Expose an AI inference API through a resilient, scalable load balancer with health checks, rate limits, canary rollout, and observability.
- Outcome: A working L4/L7 LB in Kubernetes, plus configs you can adapt to EKS/GKE/AKS or bare metal.
- Download the ready-made files:
  - Kubernetes Manifests (FastAPI app + Services + HPA + Ingress + Canary)
  - Envoy Proxy (L7 LB with retries, outlier detection, circuit breaking)
  - README – quick commands
```
0) Prerequisites

    A Kubernetes cluster (EKS/GKE/AKS or local with a LoadBalancer solution).

    kubectl configured, and an Ingress controller if you want L7 (e.g., ingress-nginx).

    Optional: a DNS name (e.g., api.example.com) and TLS certs for HTTPS.

1) What You’ll Deploy (Architecture)

    Inference API (FastAPI) with /healthz, /readyz, /v1/echo.

    Service (ClusterIP) for in-cluster access.

    Service (LoadBalancer, L4) for a cloud LB (NLB/ELB on AWS, similar on GKE/AKS).

    Ingress (L7) via NGINX Ingress for TLS, routing, basic rate limits.

    HPA to autoscale pods by CPU (you can adapt to custom metrics).

    Canary Ingress (10% traffic) to validate new versions safely.

    (Optional) Envoy config as an alternative L7 LB with retries and outlier detection.

2) Deploy the Stack

    # Apply everything (Namespace, ConfigMap, Deployment, Services, HPA, Ingress, Canary)
    kubectl apply -f lab91_k8s_inference_fastapi.yaml
     
    # If using Ingress + TLS, create the TLS secret (replace paths)
    kubectl -n ai create secret tls inference-tls --cert=server.crt --key=server.key

Verify:

    kubectl -n ai get pods,svc,ingress,hpa

3) Test Locally First

If your LB isn’t provisioned yet, port-forward to the ClusterIP:

    kubectl -n ai port-forward svc/inference-svc 8080:80
    curl -s http://localhost:8080/healthz
    curl -s -X POST http://localhost:8080/v1/echo \
      -H "Content-Type: application/json" \
      -d '{"text":"hello"}'

4) Test the Cloud Load Balancer (L4)

When inference-lb shows an external IP/DNS:

    kubectl -n ai get svc inference-lb
    curl -s http://<EXTERNAL_LB> /healthz

This is straight L4 (TCP) balancing to your pods via K8s Service.
5) Add L7 Features with Ingress (NGINX)

    TLS termination, per-path routing, and rate limits are pre-wired in the manifest.

    Point api.example.com DNS to the L4 LB hostname/IP.

    # After DNS and TLS secret are ready:
    curl -s https://api.example.com/healthz
    curl -s -X POST https://api.example.com/v1/echo \
      -H "Content-Type: application/json" -d '{"text":"gamma"}'

What’s included (already in the YAML):

    nginx.ingress.kubernetes.io/limit-rps: "100" + burst multiplier

    60s proxy timeouts for long-running inference

    TLS via inference-tls secret

6) Canary Release (10% Traffic)

The inference-ingress-canary routes 10% of requests to inference-api-canary.
Test by issuing many requests and watching some hit faster canary pods:

    for i in {1..50}; do
      curl -s https://api.example.com/healthz | jq .
    done

Adjust canary weight by changing nginx.ingress.kubernetes.io/canary-weight.
7) Autoscaling (HPA)

HPA is set to CPU 70%, min 2 pods, max 50. Generate load and watch scaling:

    kubectl -n ai get hpa -w

You can switch to GPU utilization or custom metrics in advanced labs.
8) Optional: Envoy as Smart L7 LB

Use the provided lab91_envoy.yaml if you want an Envoy front proxy with:

    LEAST_REQUEST LB policy

    Retries and per-try timeouts

    Outlier detection (auto-eject unhealthy backends)

    Circuit breakers (caps connections/requests)

Typical deployment patterns:

    Run Envoy as a DaemonSet or sidecar, or as a standalone Deployment fronting inference-svc.

    Point your public LB at Envoy (port 8080), which then routes to the service.

9) Health Checks, Backpressure & Resilience

    Readiness/Liveness are wired (/readyz, /healthz) → LB only routes to ready pods.

    Rate limiting: protects backends during spikes.

    Canary: de-risk releases and enable quick rollback.

    Retries + outlier detection (Envoy path): eject flaking pods to reduce tail latency.

10) Observability (Minimum Viable)

    App exposes /metrics placeholder; in real infra, expose Prometheus metrics.

    Capture p50/p95/p99 latency, error rate, RPS, and backend queue depth.

    For L7: use NGINX or Envoy metrics dashboards; add OpenTelemetry tracing across LB → app.

11) Load & Failure Testing

Load tests (choose one on your laptop/runner):

    # hey
    hey -z 60s -q 50 -c 100 -m POST -H "Content-Type: application/json" \
      -d '{"text":"load"}' https://api.example.com/v1/echo
     
    # wrk
    wrk -t4 -c100 -d60s -s post.lua https://api.example.com/v1/echo

Failure drills:

    # Kill a pod and watch the LB keep SLOs
    kubectl -n ai delete pod -l app=inference-api --wait=false

12) Clean Up

    kubectl delete -f lab91_k8s_inference_fastapi.yaml

What to Capture for Your Lab Report

    External LB endpoint (or domain) and screenshots of success responses.

    HPA scaling events during load test.

    p95/p99 latency before/after enabling rate limits or Envoy.

    Canary rollout notes (weight used, observed behavior).

    Any resilience behavior during pod kill tests.

Pro Tips / Extensions

    Add JWT auth at the Ingress (or Envoy ext_authz) before exposing public endpoints.

    Make LB GPU-aware by exporting backend queue depth/concurrency and feeding it to a smarter router.

    Use Gateway API (K8s) as a more future-proof alternative to classic Ingress.
```

## Section 15: Week 14: Model Serving Basics

### 93. 92. From Training to Serving – The Deployment Gap
- Deployment gap
  - Environment mismatch
  - Missing requirements
  - Undefined process

### 94. 93. REST vs gRPC for Model APIs
- Protocol fundamentals
  - REST
    - HTTP/1.1 or HTTP/2 + JSON payloads
    - Resource oriented 
    - Human readable data format
  - gRPC
    - Built on HTTP/2 + Protocol buffers
    - Contract-first development approach
    - Streaming capabilities built-in
    - Designed for service-to-service comms
    - Binary protocol buffers prodcue smaller payloads
- When REST shines
  - Public API compatibility
  - Rapid development
  - Rich ecosystem
  - Debugging ease
- When gRPC shines
  - Performance-critical systems
  - Advanced streaming capabilities
  - Strong schema enforcement
  - Polygot development
- Model serving patterns
  - Synchronous inference: Either REST or gRPC
  - Batch/streaming inference: gRPC preferred
  - Large embedding vectors: gRPC
  - Browser-facing applications: REST frontend -> internal gRPC hop
  - Multi-model inference pipelines: gRPC
  
### 95. 94. TensorFlow Serving for AI Models
- Endpoints and ports
  - gRPC endpoint
    - Default port: 8500
    - API methods: predict, classify, regress
    - Optimized for internal service-to-service
    - Higher performance with binary protocol
  - REST endpoint
    - Default port: 8501
    - API path: /v1/models/;predict
    - Better for external/edge services
    - JSON based for easier debugging
- Model versioning and policies
  - Directory based version control
  - Make all versions available through diffferent endpoints
- Multi-model config
  - Ex: text model + image classification
  - May jeopardize resource
- Batching & throughput
  - `--enable-batching=true`
- Performance tips
  - Protocol optimization: gRPC + protobuf 
  - Warm requests: eliminate cold-start latency
  - Precision otimization: float16/bfloat16
  - Resource management: Pinning CPUs  

### 96. 95. TorchServe for PyTorch Models
- Why Torchserve?
  - Official PyTorch solution
  - Built-in APIs: REST and gRPC
  - Multi-model support: hosts multiple models simultaneously
  - Production features
    - Logging, metrics collection, batch inference, and A/B testing support
- Core components
  - Model Archive (MAR)
  - TorchServe Process
  - Management API
  - Inference API
- Creating a Model Archive (MAR)  
  - Export model to .pt or a scripted/traced format
  - Bundle with a handler
  - Use torch-model-archiver to create MAR fiel
- Starting TorchServe
  - Launch the server: `torchserve --start --model-store model_store --models resnet50=resnmet50.mar`
  - Access Inference API:
    - REST endpoint: `POST /predictions/resnet50`
    - gRPC service: `service InferenceAPIsService { rpc Predictions(PredictionsRequest)} returns (PredictionResponse); }`
- Configurations and options
  - Config.properties
  - Worker scaling
  - Batch inference
  - Multi-model hosting
- Common pitfalls
  - Oversized model archives
  - Resource contention
  - Hidden latency
  - Batch size neglect

### 97. 96. Deploying Models with FastAPI
- Why FastAPI?
  - Modern Python Framework
  - High Performance
  - ML Library Integration
  - Built-in OpenAPI docs
- Core workflow
  - Load your trained model
  - Create FastAPI app
  - Add Pre/Post processing
  - Deploy
- Minimal example:
```py
from fastapi imprt FastAPI
import joblib
app = FastAPI()
model = joblib.load("model.pkl")
@app.post("/predict")
def predict(features:dict):
    return {"prediction":model.predict([list(features.values())])[0]}
```
  - Run `uvicorn main:app --host 0.0.0.0 --port 8000`
- Adding pydantic models
  - Use `BaseModel` for request/response schemas to ensure validation and enhance API docs
```py
from fastapi import FastAPI
from pydantic import BaseModel
class Input(BaseModel):
    age: int
    income: float
    credit_score: int
    has_debt: bool
class Output(BaseModel):
    prediction: float
    probability: float
  @app.post("/predict",response_model=Output)
  def predict(data: input):
      # process with model
      return Output(...)
```
- Containerization
  - Consistent environments
  - Dependency isolation
  - Easy deployment
  - Versioning support
- Scaling & Production
  - Concurrency: Use Gunicorn workers to handle multiple requests simultaneously
  - Orchestration: deploy on kubernetes with horizontal pod autoscaling
  - Load balancing
  - Caching: Add Redis layer for caching frequent inference requests
- Advanced patterns
  - Async endpoints
```py
@app.post("/predict")
async def predict(data: input):
    result = await run_inference(data)
    return result
```  
  - Streaming responses
```py
@app.post("/generate")
async def generate(prompt: str):
    async def token_generator():
        for token in model.generate(prompt):
            yield {"token":token}
    return StreamingResponse(token_generator())
```

### 98. 97. Scaling Model Serving with Kubernetes
- Autoscaling options
  - Horizontal Pod Autoscaler (HPA)
    - Scales number of inference pods based on CPU/GPU utilization or custom metrics
  - Veritcal Pod Autoscaler (VPA)
    - Dynamically adjusts CPU/memory resources allocated to containers
  - Cluster Autoscaler
    - Adds or removes worker nodes when pods can't be scheduled
  - Custom Metric Scaling
    - Scale on business metrics: queries per second, inference latency, queue depth
- Cost optimization
  - Right size resources
  - Leverage spot/preemptible instances
  - Efficient autoscaling
  - Separate training and inference
- Common pitfalls
  - Resource misconfiguration   
  - Incomplete scaling metrics
  - Poor GPU utilization
  - Cold start latency surprises
- Best practice
  - Scalable infrastructure
  - Multi-layer scaling
  - SLO-driven operations
  - Separation of concerns

### 99. 98. Lab – Serve an Image Classifier with FastAPI
- 🎯 Goal: Deploy a pretrained image classification model (ResNet18 by default) behind a FastAPI REST API. You’ll run it locally, send images for prediction, and optionally containerize it with Docker or deploy it on Kubernetes.
```
Step 0 – Prerequisites

    Python 3.10+

    pip or conda

    uvicorn for serving

    GPU optional (CUDA used automatically if available)

Step 1 – Project Setup

Create a folder lab98_fastapi_classifier/ and inside it add:

    lab98_fastapi_classifier/
    ├── app/
    │   └── main.py
    ├── scripts/
    │   └── client.py
    ├── requirements.txt
    ├── Dockerfile
    └── README.md

Step 2 – Install Dependencies

requirements.txt:

    fastapi==0.112.2
    uvicorn[standard]==0.30.6
    pillow==10.4.0
    torch==2.3.1
    torchvision==0.18.1
    pydantic==2.8.2
    python-multipart==0.0.9

Install them:

    pip install -r requirements.txt

Step 3 – FastAPI App (main.py)

app/main.py:

    import io, os, time
    from typing import List
     
    import torch, torchvision
    from torchvision import transforms
    import torch.nn.functional as F
    from PIL import Image
     
    from fastapi import FastAPI, UploadFile, File, HTTPException, Query
    from pydantic import BaseModel
     
    # ---- App ----
    app = FastAPI(title="Image Classifier API")
     
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = torchvision.models.resnet18(weights="DEFAULT").to(DEVICE).eval()
    PREPROCESS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    LABELS = [l.strip() for l in open("imagenet_classes.txt").readlines()]
     
    class Prediction(BaseModel):
        index: int
        label: str
        probability: float
     
    class PredictResponse(BaseModel):
        model: str
        device: str
        top_k: int
        time_ms: float
        predictions: List[Prediction]
     
    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "device": DEVICE}
     
    @app.post("/predict", response_model=PredictResponse)
    async def predict(image: UploadFile = File(...), top_k: int = Query(5, ge=1, le=20)):
        if image.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(415, f"Unsupported file type: {image.content_type}")
     
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = PREPROCESS(img).unsqueeze(0).to(DEVICE)
     
        t0 = time.perf_counter()
        with torch.inference_mode():
            probs = F.softmax(MODEL(tensor), dim=1)
            top_p, top_i = probs.topk(top_k, dim=1)
        elapsed = (time.perf_counter() - t0) * 1000
     
        preds = [Prediction(index=int(i), label=LABELS[i], probability=float(p))
                 for p, i in zip(top_p[0], top_i[0])]
        return PredictResponse(model="resnet18", device=DEVICE,
                               top_k=top_k, time_ms=elapsed, predictions=preds)

Download ImageNet labels:

    wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

Step 4 – Run the API

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Test health:

    curl http://localhost:8000/healthz

Step 5 – Test Prediction

scripts/client.py:

    import sys, requests
     
    url = "http://localhost:8000/predict"
    img_path = sys.argv[1]
     
    with open(img_path, "rb") as f:
        files = {"image": (img_path, f, "image/jpeg")}
        r = requests.post(url, files=files, params={"top_k": 5})
        print(r.json())

Run:

    python scripts/client.py path/to/image.jpg

Step 6 – Dockerize (Optional)

Dockerfile:

    FROM python:3.11-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY app/ ./app/
    COPY imagenet_classes.txt .
    EXPOSE 8000
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

Build & run:

    docker build -t fastapi-cls .
    docker run -p 8000:8000 fastapi-cls

✅ Learning Outcomes

    Built a FastAPI model server for image classification

    Served predictions via /predict endpoint

    Integrated PyTorch + Torchvision models

    Tested locally and containerized for deployment
```

## Section 16: Week 15: Advanced Model Serving

### 100. 99. NVIDIA TensorRT Optimization
- TensorRT: Nvidia's specialized SDK for high-performance inference
  - Lower latency
  - Reduced cost
  - Higher throughput
- Core optimization
  - Layer fusion
  - Precision calibration: automatically converts FP32 into FP16/INT8
  - Kernel auto-tuning
  - Memory optimization
- Supported model formats
  - TensorFlow
  - PyTorch
  - ONNX
    - `trtexec --onnx=model.onnx --saveEngine=model.engine`
  - TensorRT Engine
- Precision Modes
  - FP32: 1x
  - FP16: 2x
  - INT8: 4x
  - FP8: 6x
- TensorRT workflow overview
  - Export model
  - Optimize with TensorRT
  - Select precision
  - Deploy engine
  - Benchmark Performance    
- Deployment options
  - Standalone TensorRT
  - Nvidia Triton server
  - Kubernetes integration
  - Edge deployment
    - Jetson devices with TensorRT runtime
- Performance optimization tips
  - Static vs dynamic shapes
  - Engine warm-up
  - GPU friendly dimensions: align input shapes with multiple sof 8/16/32 to maximize tensor core utilization
  - Performance metrics: monitor both p95/p99 latency and throughput
- Observability and debugging
  - TensorRT profiling tools
    - TensorRT verbose logs
    - Nsight systems
    - Nsight compute
  - Key metrics to monitor
    - SM utilization
    - Memory throughput
    - Kernel duration
    - Accuracy comparison
- Common pitfalls and solutions
  - Unsupported operations
    - Exotic ONNX ops cause fallback -> implement custom plugins or replace with supported alternatives
  - Calibration issues
    - Poor INT8 accuracy due to inadequate calibration dataset -> use representative dataset with 100+ samples covering the input distribution
  - Precision problems
    - Over-aggressive precision causing unstable predictions -> set precision per layer with TensorRT API for sensitive operations
  - Batch size neglect  
    - Default batch size leads to underutilized GPU -> profile different batch sizes to find optimal throughput/latency balance

### 101. 100. Triton Inference Server Basics
- Triton
  - An open-source model serving platform
  - Supports multiple ML framekworks: PyTorch, TensorFlow, ONNX, TensorRT, XGBoost, and more
  - Provides gRPC/REST APIs for easy integraiton
  - Enables dynamic bathcing
  - Facilitates multi-model hosting
- Why Triton?
  - Unified serving
  - GPU optimization
  - Advanced features
  - Monitoring
- Deployment options
  - Docker containers: official Nvidia container images
  - Kubernetes: Scale with Helm charts and NGC containers for orchestrated deployments across clusters
  - Edge devices
  - Integration options
- Model repository structure
```bash
models/
└──resnet50/
   └── v1/
       ├── model.onnx
       └── config.pbtxt
```       
  - One directory per model (resnet50)
  - Versioned subfolders (v1)
  - config.pbtxt defines critical model parameters  
    - Input/output specifications
    - Batching configuraiton
    - Backend selection
- Launching Triton server
  - `docker run --gpus all -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:24.05-py3 tritonserver --model-repository=/models`
  - HTTP/REST (8000)
  - gRPC (8001)
  - Metrics (8002): Prometheus-compatible endpoint
- Making client requests
  - REST API ex: `POST http://localhost:8000/v2/models/resnet50/infer`
- Dyanmic batching
  - Triton can automatically combine multiple inference requests into a single batch operation, significantly improving GPU utilization and throughput
  - Particularly effective for high-volume inference workloads where individual requests arrive rapidly but asynchronously
- Model ensembles
  - Chain multiple models into a unified inference pipeline, reducing IO overhead and simplifying orchestration
  - Steps
    - Preprocessing
    - Core inference
    - Post processing
- Common pitfalls to avoid
  - Configuration errors: incorrect config.pbtxt
  - Resource allocation: Oversized batch configuration
  - Version compatibility: Mismatching CUDA/cuDNN versions b/w training and inference environments
  - Oversubscription: Running too many models per GPU

### 102. 101. Batch Inference vs Online Inference
- Batch inference
  - Large volumes of predictions processed in bulk jobs
  - Typically run on a schedule
  - Optimized for throughput and cost efficiency
  - Ex: churn prediction, risk scoring, recommendation refresh
- Online inference
  - Predictions served on-demand in real time
  - Must meet strict p95/p99 latency SLOs
  - Optimized for responsiveness and availabilty
  - Ex: fraud detection, ads ranking, chatbot responses
- Batch inference characteristics
  - Hough throughput
  - Large clusters
  - Result storage
  - Cost efficiency
- Online inference characteristics
  - Low latency/jitter
  - Needs autoscaling
  - REST/gRPC APIs
  - Modern infrastructure: Kubernetes + GPU/CPU optimized nodes + sophisticated load balancing
- Infrastructure trade-offs
  - Batch  
    - Cost efficient for non-urgent jobs
    - Fault tolerance
    - Resource optimization
  - Online
    - Higher operational costs
    - Reliability engineering
    - Complex scaling
- Common pitfalls 
  - Using online inference when batch suffices: overspending
  - Ignoring feature skew b/w offline batch vs online service
  - Not monitoring tail latency in real-time APIs
  - Overloading online infra with requests better suited for batch
- Decision matrix
  - Choose batch when:
    - Workload tolerates latency of minutes to hours
    - Throughput + cost > real-time needs
    - Predict once, reuse many times
    - Processing window is predictable
    - Resource efficiency is priortized
  - Choose online when:
    - Requires resonse in ms to sec
    - Latency + SLA-driven workloads
    - Predictions tied to user interactions
    - Immediate decisions are needed
    - User experience depends on speed
- Best practices
  - Define SLOs before choosing mode
  - Use feature stores for consistency
  - Monitor both accuracy drift and infra metrics
  - Design clear ownership b/w batch +  online pipelines

### 103. 102. Caching for Fast Inference
- Why caching matters
  - Many inference requests are repeated or highly similar
  - Strategic caching delivers following benefits
    - Latency
    - Cost
    - Load
- Levels of caching
  - Client-side
  - Edge/CDN
  - API gateway: Cache REST/gRPC calls
  - Application layer: Redis/Memcached store
  - Feature/model cache: store embeddings
- What to cache?
  - Full API responses
  - Model outputs for identical inputs
  - Precomputed embeddings
  - Preprocessing results
- Cache keys and strategies
  - Deterministic keys
  - Input normalization
  - Version tagging
  - Time-to-Live (TTL): balance freshness vs cost savings by setting expriation periods
- Tools and implementation
  - In-Memory stores: Redis/Memcached
  - Edge solutions: Cloudflare, Akamai, ASWS CloudFront
  - Gateway Layers: Envoy, NGINX
  - Vector databases: FAISS, Pinecone, Weaviate
- Inference-specific patterns
  - LLMs: Cache prompts
  - Computer Vision: Cache image fingerprints
  - Recommendation: Cache top-N recommendation lists
  - Search: cache embedding
- Challenges and pitfalls
  - Staleness
  - Memory pressure
  - Cache invalidation
  - Metric masking
- Best practices
  - Version everything
  - Implement tiered caching
  - Monitor aggressively
  - Balance TTL settings
  - Quantify benefits
- Strategically implemented caching dramatically reduces latency, cost, and stress on GPU clusters
- Compbine multiple cache layers for maximum efficiency and resilience

### 104. 103. Multi-Model Serving Strategies
- Why multi-model serving?
  - Modern enterprises often manage hundreds or thousands of production ML models
  - Need to balance cost, latency, isolation and manageability
- Key approaches
  - Single model containers
    - One pod = one model
    - Pros: strong isolation, easy debugging, single deployment
    - Cons: Poor GPU/CPU utilization, higher overhead, resource waste
  - Multi-model servers
    - Serve multiple models in one process (e.g., Triton, TorchServe)
    - Pros: Resource sharing, dynamic/loading/unloading, better utilization
    - Cons: Noisy-neighbor risks, complex configurations, harder to debug
- Dynamic model loading
  - Load models on demand
  - Useful for long-tail models (rarely used but required)
- Scaling patterns
  - Horizontal scaling: replicate pods across nodes
  - Weighted routing: split traffic per model version
  - Popularity-based autoscaling: scale resources based on per-model traffic patterns
- Resource sharing strategies
  - Co-locate models with similar resource footprints to avoid imbalance
  - Use GPU partitioning (MIG) to isolate resource intensive models
  - Monitor per-model latency and QPS
  - Apply granual quota 
- Resource strategies
  - Static routing: fixed URL path per model
  - Dynamic routing: model ID in request path
  - Ensemble routing: chain models in pipelines
- Common pitfalls 
  - Memory thrashing
  - Startup bloat
  - Environment mixing
  - Missing SLOs
- Best practices
  - Criical model isolation: start with single-model per pod for high-value applications with strict SLAs
  - Scale with multi-model servers
  - Optimize for the long trail
  - Instrument everything

### 105. 104. A/B Testing Model Endpoints
- Why A/B test models?
  - Real-world validation
  - Risk reduction
  - Business impact measurement
- Core concepts
  - Control model (A)  
  - Candidate model (B)
  - Comparison: collect metrics from both models to determine promotion or rollback
- Traffic splitting methods
  - Weighted routing: 90%/10% for A/B, Best for initial testing with minimal risk
  - User/session-based: ensures users get consistent experience
  - Header-based: gradual rollout strategies
  - Progressive rollout: balances risk and data collection needs
- Infrastructure patterns
  - Traffic management
    - API Gateway
    - Service mesh
    - Proxy servers
  - Model serving
    - Inference servers (Triton, TorchServe)
    - Custom endpoints (FastAPI, Flask)
    - Load balancers for scaling
  - Observability and safety
    - Metrics collectoin (Prometheus)
    - Visualization (Grafana)
    - Automated canary analysis and rollback
- What to measure
  - Technical metrics: latency, error rate & exceptions, throughput, resource utilization
  - Model metrics: accuracy/precision/recall, confidence scores, feature drift indicators, fairness metrics
  - Business KPIs: conversion rates, click-through rates, revenue impact, user engagement
- Statistical considerations
  - Sample size determination
  - Statistical testing
  - Avoid premature conclusions
  - Monitor for bias
- Observability Practices
  - Implementation techniques
    - Tag everything with model version and request ID
    - Export per-model metrics
    - Create side-by-side dashboards
    - Set up automated alerts
    - Implement automatic rollback triggers
  - Essential dashboards
    - Traffic distribution
    - Latency comparison
    - Error rate
    - Business KPI
    - Statistical significance
    - Feature drift
- Security and governance
  - Authentication & rate limiting
  - Documentation & auditing
  - Data protection
- Common pitfalls
  - Insufficient traffic
  - Traffic mix mismatch
  - Inconsistent routing
  - Tunnel vision: focusing only on ML metrics while ignoring business KPIs
- Best practices
  - Start small, scale gradually
  - Implement sticky routing
  - Define success criteria upfront
  - Automate deployment pipeline

### 106. 105. Lab – Deploy a Model on Triton Server
- 🎯 Goal: Export a model to ONNX, create a Triton model repository, run Triton in Docker, and send real inference requests. Optional: TensorRT, batching, K8s, metrics.
```
0) Prerequisites

    Docker installed

    (Optional GPU) NVIDIA driver + nvidia-container-toolkit

    Python 3.10+ (pip install torch torchvision pillow numpy requests)

    Ports 8000/8001/8002 available (HTTP/gRPC/metrics)

1) Create a Triton model repository layout

    mkdir -p ~/lab105/models/resnet50_onnx/1
    cd ~/lab105

Triton expects:

    models/
     └─ resnet50_onnx/
         ├─ config.pbtxt
         └─ 1/
            └─ model.onnx      # you'll export this next

Create models/resnet50_onnx/config.pbtxt:

    name: "resnet50_onnx"
    platform: "onnxruntime_onnx"
    max_batch_size: 16
     
    input [
      { name: "input",  data_type: TYPE_FP32, dims: [3,224,224] }
    ]
    output [
      { name: "logits", data_type: TYPE_FP32, dims: [1000] }
    ]
     
    instance_group [ { kind: KIND_GPU, count: 1 } ]
     
    dynamic_batching {
      preferred_batch_size: [4, 8, 16]
      max_queue_delay_microseconds: 1000
    }

2) Export a pretrained ResNet50 to ONNX

Create export_onnx.py in ~/lab105:

    import torch, torchvision as tv
     
    def main():
        model = tv.models.resnet50(weights="DEFAULT").eval()
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model, dummy, "models/resnet50_onnx/1/model.onnx",
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17
        )
        print("Exported to models/resnet50_onnx/1/model.onnx")
     
    if __name__ == "__main__":
        main()

Run it:

    python export_onnx.py

3) Start Triton (Docker)

Pick a tag (e.g., latest or a specific 23.xx-py3):

    export TRITON_TAG=latest
    docker run --rm -it \
      --gpus all \
      -p8000:8000 -p8001:8001 -p8002:8002 \
      -v $PWD/models:/models \
      nvcr.io/nvidia/tritonserver:${TRITON_TAG} tritonserver \
        --model-repository=/models \
        --strict-model-config=false \
        --exit-on-error=false

No GPU? Omit --gpus all (slower but fine for testing).

Health/metadata checks (in another terminal):

    curl -s http://localhost:8000/v2/health/ready && echo
    curl -s http://localhost:8000/v2/models/resnet50_onnx | jq

Metrics (Prometheus format):

    curl -s http://localhost:8002/metrics | head

4) Send a real inference request (HTTP/JSON)

Quick Python client (client_http.py) in ~/lab105:

    import argparse, numpy as np, requests
    from PIL import Image
     
    def preprocess(img):
        img = img.convert("RGB").resize((256,256))
        o = (256-224)//2; img = img.crop((o,o,o+224,o+224))
        x = np.asarray(img).astype("float32")/255.0
        mean = np.array([0.485,0.456,0.406],dtype=np.float32)
        std  = np.array([0.229,0.224,0.225],dtype=np.float32)
        x = (x-mean)/std
        x = np.transpose(x,(2,0,1))[None, ...]  # NCHW
        return x
     
    if __name__ == "__main__":
        ap = argparse.ArgumentParser()
        ap.add_argument("image_path")
        ap.add_argument("--url", default="http://localhost:8000/v2/models/resnet50_onnx/infer")
        args = ap.parse_args()
     
        x = preprocess(Image.open(args.image_path))
        payload = {
          "inputs": [{
            "name": "input",
            "shape": list(x.shape),
            "datatype": "FP32",
            "data": x.flatten().tolist()
          }],
          "outputs": [{"name": "logits"}]
        }
     
        r = requests.post(args.url, json=payload, timeout=60)
        r.raise_for_status()
        out = r.json()["outputs"][0]["data"]
        logits = np.array(out, dtype=np.float32).reshape((x.shape[0],1000))
     
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exps / exps.sum(axis=1, keepdims=True)
        top5 = probs[0].argsort()[-5:][::-1]
        print("Top-5 indices:", top5.tolist())
        print("Top-5 probs:", probs[0][top5].tolist())

Run:

    pip install pillow requests numpy
    python client_http.py path/to/image.jpg

5) (Optional) Use TensorRT for speed

Build an FP16 TensorRT engine:

    trtexec --onnx=models/resnet50_onnx/1/model.onnx \
            --saveEngine=models/resnet50_trt/1/model.plan \
            --explicitBatch --fp16

Create models/resnet50_trt/config.pbtxt:

    name: "resnet50_trt"
    platform: "tensorrt_plan"
    max_batch_size: 16
    input:  { name: "input",  data_type: TYPE_FP32, dims: [3,224,224] }
    output: { name: "logits", data_type: TYPE_FP32, dims: [1000] }
    instance_group [ { kind: KIND_GPU, count: 1 } ]
    dynamic_batching { preferred_batch_size: [4,8,16], max_queue_delay_microseconds: 1000 }

Restart Triton (or run with repository polling) and infer against resnet50_trt.
6) Tune throughput & latency

    More copies per GPU:

        instance_group [ { kind: KIND_GPU, count: 2 } ]

    Adjust preferred_batch_size/max_queue_delay_microseconds to meet p95 latency targets.

    Drive load (e.g., hey, Locust) to validate batching benefits.

7) (Optional) Minimal Kubernetes deployment

Create k8s-triton.yaml:

    apiVersion: apps/v1
    kind: Deployment
    metadata: { name: triton }
    spec:
      replicas: 1
      selector: { matchLabels: { app: triton } }
      template:
        metadata: { labels: { app: triton } }
        spec:
          containers:
          - name: triton
            image: nvcr.io/nvidia/tritonserver:latest
            args: ["tritonserver","--model-repository=/models"]
            ports:
            - { containerPort: 8000 }
            - { containerPort: 8001 }
            - { containerPort: 8002 }
            volumeMounts:
            - { name: model-repo, mountPath: /models }
            resources:
              limits: { nvidia.com/gpu: 1 }
          volumes:
          - name: model-repo
            hostPath: { path: /path/on/node/models }  # or PVC
    ---
    apiVersion: v1
    kind: Service
    metadata: { name: triton-svc }
    spec:
      selector: { app: triton }
      type: NodePort
      ports:
      - { name: http,    port: 8000, targetPort: 8000 }
      - { name: grpc,    port: 8001, targetPort: 8001 }
      - { name: metrics, port: 8002, targetPort: 8002 }

Apply:

    kubectl apply -f k8s-triton.yaml

8) Troubleshooting

    Model fails to load: check Triton logs for input/output name/shape mismatches → fix config.pbtxt or ONNX export names.

    400 on infer: ensure request uses "input", shape [N,3,224,224], datatype:"FP32".

    Low throughput: enable dynamic batching, raise instance_group count, switch to TensorRT.

    CPU run slow: expected; use GPU/TensorRT for real perf.

    Labels: this lab prints top-5 indices; map indices to ImageNet labels if needed.

✅ You accomplished

    Exported a model → built a Triton model repo

    Served it over HTTP/gRPC with Prometheus metrics

    Sent real inference requests

    Learned batching, scaling, and TensorRT optimization paths
```

## Section 17: Week 16: Observability in AI infrastructure

### 107. 106. Why Monitoring AI Systems Matters
- Mindset
  - AI != traditional apps
  - Visibility into black-box models
  - Detect issues early
- Why AI is different
  - Non-determinism
  - Data dependency
  - Model drift: accuracy degrades over time as the world changes
  - Infrastructure sensitivity
- Dimensions to monitor
  - Infrastructure metrics: CPU/GPU utilization, memory consumption, network throughput
  - Model performance: accuracy and F1 score, drift detection metrics
  - Operational metrics: latency, request throughput, error and failure assets
  - Business KPIs: conversion rates, fraud detection accuracy, customer satisfaction
- Failure modes without monitoring
  - Silent performance drops
  - Infrastructure bottlenecks
  - Data pipeline errors
  - Compliance violations  
- Monitoring across the lifecycle
  - Training
    - Loss curves and convergence
    - Resource utilizawtion metrics
    - Reproducibilty checks
  - Deployment
    - Endpoint health and availability
    - Latency profiles under load
    - Autoscaling performance
  - Post-deployment
    - Concept drift detection
    - Bias and fairness metrics
    - Real-world accuracy tracking
  - Continuous learning
    - Feedback loop capture
    - Automated retraining triggers
    - Performance improvement tracking
- Tools & ecosystem
  - Infrastructure: Prometheus, Grafana, DCGM
  - Model operations: MLflow, wandb, EveidentlyAI
  - Observability: OpenTelemetry, ELK stack, Datadog
  - Business layer: Custom dashboards
- Best practices
  - Define clear SLOs
  - Monitor distribution, not averages
  - Combine signal types
  - Build alerts and playbooks  

### 108. 107. GPU Monitoring with DCGM
- Data Center GPU Management is Nvidia's low-level toolkit for comprehensive GPU telemetry and management
  - Natively integrates with Prometheus, Kubernetes, Slrum, and DCGM expoter
- Why DCGM?
  - GPUs are the bottleneck
  - CPU monitoring is insufficient
  - Comprehensive detection
- Key metrics
  - Utilization: GPU core percentage, memory utilization, SM occupancy
  - Memory: Used vs free memory, bandwidth throughput, memory controller utilization
  - Power & thermals: wattage consumption, throttling events, Fan speed and temperature
  - Errors and process stats: ECC memory errors, PCIe/NVLink connectivity issues, PID-to-GPU resource mapping
- DCGM deployment modes
  - Embedded mode: library integrated inside application with C and Python bindings for custom monitoring solutions
  - DCGM exporter: Prometheus-compatible metrics endpoint
  - Standalone mode: CLI (dcgmi) for system adminstrators to perform on-demand diagnotics and checks
  - Kubernetes integraiton
- CLI examples with dcgmi
```
dcgmi discovery # list all GPUs
dcgmi stats --gpu 0 # get stats of GPU 0
dcgmi diag -r 1 # run short diagnostics
dcgmi health --set 1 # enable health monitoring group
```
- DCGM exporter for Prometheus
  - Run a DaemonSet on every GPU node
  - Exposes `/metrics` endpoint for Prometheus scaping
  - Example metrics exposed:
    - `DCGM_FI_DEV_GPU_UTIL`: GPU utilizatino percentage
    - `DCGM_FI_DEV_MEM_COPY_UTIL`: memory controller activity
    - `DCGM_FI_DEV_POER_USAGE`: power consumption in watts
- Kubernetes integration
  - Nvidia GPU Operator includes DCGM exporter by default
  - Auto-labeling
  - Scaling integration
- Nvidia provides ready-to-use Grafana dashboards for DCGM metrics
- Best practices
  - Deploy DCGM exporter in all GPU clusters
  - Track utilization efficiency
  - Implement proactive alerting
  - Use metrics for capacity planning 

### 109. 108. Metrics Collection with Prometheus
- Prometheus
  - Metrics monitoring system
  - PromQL Query Language
  - Cloud-native architecture
- Why Prometheus for AI ?
  - Unified observability
  - High-Cardinality support
  - Rich ecosystem integraion: seamless works with DCGM, Triton, and Kubernetes
  - Actionable Insights  
- Core Prometheus Concepts
  - Metrics
    - Counter: cumulative metrics
    - Gauge: current value metrics
    - Histogram
    - Summary: calculates quantiles client-side
  - Labels
  - PromQL
  - Scrape Targets
- Scrape configuration example
  - prometheus.yml:
```yaml
scrape_configs:
 - job_name: 'gpu-metrics'
   static_configs:
    - targets: ["node1:9400","node2:9400"]
 - job_name: "triton"
   static_configs:
    - targets:["triton-svc:8002"]    
```
- PromQL in action
  - GPU utilization (avg per cluster): `avg(DCGM_FI_DEV_GPU_UTIL)`
  - Model inference QPS (per model): `rate(nv_inference_count[1m]) by (model)`
  - P95 latency: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
- Integration with Kubernetes  
  - Automated discovery and configuration
  - Pod annotations for auto-discovery
```
annotations:
  prometheus.io/scrape:"true"
  prometheus.io/port:"8002"
```
- Metrics storage and retention
  - Local TSDB
  - Remote storage: long term retention
  - Model performance history
- Best practices
  - Meaningful labels
  - Percentile monitoring
  - Cardinality management
  - Security controls

### 110. 109. Visualization Dashboards with Grafana
- Grafana
  - Open source analytics and visualization platform
  - Connects seamlessly to Prometheus, Loki, Elastic, CloudWatch and others
- Why Grafana?
  - Unified visualization
  - Faster anomaly detection
  - Cross-team visibility
  - Integrated alerting
- Core building blocks
  - Data sources
  - Panels
  - Dashboards
  - Alerts
- Prometheus + Grafana setup
  - Add Prometheus as Data source
  - Create PromQL queries
  - Configure auto-refresh
- Visualization types
  - Time-series graphs
  - Single stat gauges
  - Heatmaps
  - Tables
- Dashboards in Kubernetes
  - Deployment options
    - Run Grafana via Helm chart or Operator for K8 native management
    - Import Nvidia GPU + Triton dashboards from community templates
    - Configure persistent storage for dashboard definitions
  - Organization
    - Integrate with K8 RBAC for team-based access control
    - Use folders to group dashboards by domain
    - Implement namespace-based segregation for multi-tenant setups
- Advanced features
  - Annotations
  - Variables
  - Drilldowns
  - Alerting 2.0
- Best practices
  - Audience-focused design
  - Prioritize key metrics
  - Embrace templating
  - Implement GitOps: store dashboard JSON in version control and automate provisioning via CI/CD pipeline

### 111. 110. Tracing AI Requests with OpenTelemetry
- OpenTelemetry
  - Open-source observability framework
  - Unified standard: collects traces, metrics, logs
  - Ecosystem integration with Prometheus, Grafana, Jaeger, Datadog
  - Multi-language support: Python, Java, C++, Go, and many others
- Why tracing matters for AI
  - Multi-layer complexity
  - Hidden failure points
  - Root-cause visibility
  - Tail latency detection
- Core tracing concepts
  - Trace
  - Span
  - Context propagation
  - Attributes and events
- AI pipeline example trace
  - Request & FastAPI
  - TRiton server
  - Feature store
  - Post-processing
  - Response  
- Instrumenting AI services
  - Python SDK for AI Frameworks
    - Add opentelemetry API into FastAPI, Flask, TorchServe, etc
    - Auto-instruments HTTP endpoints, database calls, and external dependencies
  - Key implementation steps
    - Install OpenTelemetry SDK and instrumentation package
    - Configure trace exporter to your backend
    - Add auto-instrumentation for your framework
    - Create custom spans for model-specific operation
    - Attach trace IDs to logs for correlation
- Exports and backends
  - Open source tracking backends: Jaeger & Tempo
  - Unified Observability: Grafana + Temp + Loki
  - Cloud-native: AWS X-Ray, GCP Trace, Azure Monitor
- Tracing GPU & model layers
  - Model execution spans
  - Rich context attributes: model version, batch size, GPU ID, ...
- Sampling strategies
  - 100% always-on: Complete but expensive
  - 1-10% probabilistic: predictable overhead but may miss specific issues
  - p95+ tail-based: captures problementic requests but requires dynamic sampling implementation
- Best practices
  - End-to-end propagation
  - Consistent tagging
  - Strategic sampling
  - Integrated observability

### 112. 111. Building Alerts for AI System Failures
- Why alerts matter
  - AI systems fail in subtle & silent ways
  - Early detection prevents:
    - Customer-facing outages
    - Model drift going unnoticed
    - Expensive infrastructure waste
- Types of failures to alert on
  - Infrastructure
    - GPU out-of-memory errors
    - Thermal throttling
    - Node crashes or restart
    - Network connectivity issues
  - Operation
    - High p95/p99 latency spikes
    - Error rate increases
    - Request throughput anomalies
    - Batch job failures
  - Model-level
    - Accuracy/precision drops
    - Feature drift signals
    - Bias or fairness issues
    - Prediction confidence changes
  - Business-level
    - KPI degradation
    - Fraud rate increases
    - Customer churn signals
    - Revenue impact indicators
- Anatomy of a good alert
  - Signal
  - Threshold
  - Actionability
  - Context
- Routing alerts
  - Use Alertmanager for deduplication and intelligent routing
  - Use severity levels to determine urgency
    - Critical: immediate action (PagerDuty)
    - Warning: attention needed (Slack)
    - Info: Awareness only (email)
- Avoiding alert fatigue
  - Focus on SLOs, not every metric
  - Use multi-window thresholds
  - Group related alerts
  - Regular alert hygiene
  * Having 5-8 actionable alerts per day will make folks ignore them
- AI-specific alerting patterns
  - Model-drift detection
  - GPU underutilization
  - Data pipeline stalls
  - Shadow model disagreement: when canary/shadow models diverge signficantly from production model
- Incident playbooks
  - Each alert should link to a detailed runbook with step-by-step troubleshooting instructions
- Best practices
  - Define multi-layer coverage
  - Use SLO-based thresholds
  - Establish clear on-call practices
  - Test before production

### 113. 112. Lab – Monitor GPU Cluster with Prometheus
- Here’s a practical, end-to-end lab to monitor a Kubernetes GPU cluster with Prometheus (plus Grafana dashboards and basic alerts). You’ll deploy NVIDIA’s DCGM Exporter to expose GPU metrics, scrape them with Prometheus, visualize in Grafana, and add alert rules.
```
0) What you’ll build (mental model)

    GPU nodes run DCGM Exporter (DaemonSet) → exposes GPU metrics on port 9400.

    Prometheus scrapes those metrics + standard cluster metrics (node exporter, kube-state-metrics).

    Grafana visualizes everything (dashboards).

    Alertmanager (bundled) routes alerts (we’ll add a few sample rules).

1) Prerequisites

    A Kubernetes cluster with at least one NVIDIA GPU node (drivers installed).

    kubectl and Helm configured for the cluster.

    NVIDIA device plugin installed (commonly used on GPU nodes).

    Cluster admin privileges.

    Tip: If your cluster isn’t already GPU-ready, install the NVIDIA device plugin first:

    helm repo add nvidia https://nvidia.github.io/k8s-device-plugin
    helm repo update
    kubectl create ns gpu-operator --dry-run=client -o yaml | kubectl apply -f -
    helm install nvidia-device-plugin nvidia/k8s-device-plugin -n gpu-operator

2) Create a dedicated namespace

We’ll keep everything in one place.

    kubectl create namespace monitoring

3) Install Prometheus, Grafana, and friends (kube-prometheus-stack)

This Helm chart deploys Prometheus, Grafana, Alertmanager, node-exporter, and kube-state-metrics.

    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring

Why this route? It’s the fastest way to get a production-worthy Prometheus stack with CRDs (ServiceMonitor/PrometheusRule) that make scraping and alerting clean and declarative.
4) Label your GPU nodes (for clean scheduling)

We’ll schedule the DCGM exporter only on GPU nodes.

    # Replace <node-name> with each GPU node
    kubectl label nodes <node-name> gpu=true

Why? It ensures the exporter runs only where GPUs exist. If you already have useful labels (e.g., from NFD), adjust the selector in the YAML later.
5) Deploy NVIDIA DCGM Exporter (DaemonSet)

This exposes GPU metrics (utilization, memory, temperature, power, ECC, clocks, etc.) per node/GPU.

Create dcgm-exporter.yaml:

    apiVersion: apps/v1
    kind: DaemonSet
    metadata:
      name: dcgm-exporter
      namespace: monitoring
      labels:
        app: dcgm-exporter
    spec:
      selector:
        matchLabels:
          app: dcgm-exporter
      template:
        metadata:
          labels:
            app: dcgm-exporter
        spec:
          nodeSelector:
            gpu: "true"
          hostPID: false
          hostNetwork: false
          tolerations:
            - effect: NoSchedule
              operator: Exists
            - effect: NoExecute
              operator: Exists
          containers:
            - name: dcgm-exporter
              # Pin to a known-good tag in your environment; 'latest' shown for simplicity
              image: nvcr.io/nvidia/k8s/dcgm-exporter:latest
              imagePullPolicy: IfNotPresent
              ports:
                - name: metrics
                  containerPort: 9400
              securityContext:
                privileged: true
              env:
                # Optional: adjust sampling interval if needed
                - name: DCGM_EXPORTER_KUBERNETES
                  value: "true"
              volumeMounts:
                # Provides per-pod GPU accounting when available (useful with MIG)
                - name: pod-resources
                  mountPath: /var/lib/kubelet/pod-resources
                  readOnly: true
          volumes:
            - name: pod-resources
              hostPath:
                path: /var/lib/kubelet/pod-resources
                type: Directory
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: dcgm-exporter
      namespace: monitoring
      labels:
        app: dcgm-exporter
    spec:
      clusterIP: None  # headless so Prometheus scrapes each pod endpoint
      selector:
        app: dcgm-exporter
      ports:
        - name: metrics
          port: 9400
          targetPort: metrics

Apply it:

    kubectl apply -f dcgm-exporter.yaml
    kubectl -n monitoring get pods -l app=dcgm-exporter -o wide

Why headless Service? Prometheus will discover each DaemonSet pod and scrape metrics per node, not just round-robin through a single service IP.
6) Tell Prometheus to scrape DCGM Exporter (ServiceMonitor)

Create dcgm-servicemonitor.yaml:

    apiVersion: monitoring.coreos.com/v1
    kind: ServiceMonitor
    metadata:
      name: dcgm-exporter
      namespace: monitoring
      labels:
        release: monitoring    # must match your Helm release name
    spec:
      selector:
        matchLabels:
          app: dcgm-exporter
      namespaceSelector:
        matchNames: ["monitoring"]
      endpoints:
        - port: metrics
          interval: 15s
          path: /metrics

Apply it:

    kubectl apply -f dcgm-servicemonitor.yaml

Why ServiceMonitor? kube-prometheus-stack watches these CRDs and dynamically updates Prometheus scraping config.
7) Sanity-check: see the raw GPU metrics

Port-forward to any DCGM exporter pod and curl /metrics:

    POD=$(kubectl -n monitoring get pod -l app=dcgm-exporter -o jsonpath='{.items[0].metadata.name}')
    kubectl -n monitoring port-forward pod/$POD 9400:9400
    # in another terminal:
    curl -s localhost:9400/metrics | head -n 40

You should see metrics like DCGM_FI_DEV_GPU_UTIL, DCGM_FI_DEV_GPU_TEMP, DCGM_FI_DEV_FB_USED, etc., with labels for GPU index, UUID, instance (MIG), and node.
8) Open Prometheus UI and run a few PromQL queries

Find the Prometheus service name:

    kubectl -n monitoring get svc | grep prometheus

Port-forward (replace with your service name if different):

    kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090

Open http://localhost:9090 → Graph.

Try queries (adjust label names if needed in your environment):

    GPU Utilization (%)

        avg by (instance, gpu) (DCGM_FI_DEV_GPU_UTIL)

    Memory Utilization (%)

        100 * (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL)

    GPU Temperature (°C)

        max by (instance, gpu) (DCGM_FI_DEV_GPU_TEMP)

    Power (W)

        avg by (instance, gpu) (DCGM_FI_DEV_POWER_USAGE)

If no results, check:

    ServiceMonitor metadata.labels.release matches the Helm release (monitoring above).

    DCGM Service/labels/port names match the ServiceMonitor selector and endpoints.port.

9) Access Grafana and import a GPU dashboard

Get the Grafana admin password and port-forward:

    # Password (decodes the secret created by the chart)
    kubectl -n monitoring get secret monitoring-grafana -o jsonpath="{.data.admin-password}" | base64 -d; echo
     
    kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80

Open http://localhost:3000 → login admin / (password above).

Create a GPU dashboard (quick start):

    Create a new Dashboard → Add Panel.

    Panel 1 (GPU Util %):

        avg by (instance, gpu) (DCGM_FI_DEV_GPU_UTIL)

    Panel 2 (FB Memory %):

        100 * (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL)

    Panel 3 (Temperature °C):

        max by (instance, gpu) (DCGM_FI_DEV_GPU_TEMP)

    Panel 4 (Power W):

        avg by (instance, gpu) (DCGM_FI_DEV_POWER_USAGE)

    Add a repeating variable for instance and gpu if you want per-node/per-GPU drilldown.

    Note: If your DCGM exporter uses different metric names/prefixes (some builds prefix with nvidia_dcgm_...), use Grafana’s query editor autocompletion to confirm names. The /metrics output is the source of truth.

10) Add alert rules (PrometheusRule)

Create gpu-alerts.yaml with a few pragmatic alerts:

    apiVersion: monitoring.coreos.com/v1
    kind: PrometheusRule
    metadata:
      name: gpu-alerts
      namespace: monitoring
      labels:
        release: monitoring
    spec:
      groups:
        - name: gpu.rules
          rules:
            - alert: GPUScrapeMissing
              expr: up{job="dcgm-exporter"} == 0
              for: 10m
              labels: { severity: warning }
              annotations:
                summary: "DCGM exporter scrape failing"
                description: "Prometheus cannot scrape DCGM exporter targets for 10m."
     
            - alert: GPUHighTemperature
              expr: max by (instance, gpu) (DCGM_FI_DEV_GPU_TEMP) > 80
              for: 5m
              labels: { severity: warning }
              annotations:
                summary: "GPU temperature high (>80°C)"
                description: "Instance {{ $labels.instance }} GPU {{ $labels.gpu }} too hot."
     
            - alert: GPUMemoryPressure
              expr: 100 * (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL) > 90
              for: 10m
              labels: { severity: warning }
              annotations:
                summary: "GPU memory > 90% for 10m"
                description: "Sustained memory pressure on {{ $labels.instance }} GPU {{ $labels.gpu }}."
     
            - alert: GPUECCErrorsSpike
              expr: rate(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL[5m]) > 0
              for: 5m
              labels: { severity: critical }
              annotations:
                summary: "ECC single-bit errors increasing"
                description: "ECC SBE rate > 0 on {{ $labels.instance }} GPU {{ $labels.gpu }}."

Apply:

    kubectl apply -f gpu-alerts.yaml

Wire up notifications: Edit the Alertmanager config in the chart values or via the alertmanager Secret to send to Slack, email, etc.
11) (Optional) Attribute GPU usage to Pods/Namespaces

If you mounted /var/lib/kubelet/pod-resources (we did), newer DCGM Exporter can label samples with pod, namespace, and container for per-tenant views (esp. with MIG). In Grafana, add panels grouped by namespace/pod to see who is using the GPUs.
12) Troubleshooting checklist

    No metrics in Prometheus: Verify ServiceMonitor label release: monitoring matches your Helm release; ensure the Service/port name is metrics and matches the ServiceMonitor endpoint.

    Exporter CrashLoopBackOff: Check GPU driver presence on the node: nvidia-smi (via a debug pod or SSH). Ensure privileged: true.

    MIG visibility issues: Confirm MIG mode and driver/DCGM support. If using MIG, ensure the pod-resources path is mounted as shown.

    Grafana empty: Select the Prometheus datasource (Prometheus from the stack) and check PromQL autocompletion to confirm metric names.

13) Clean up

    kubectl -n monitoring delete -f gpu-alerts.yaml
    kubectl -n monitoring delete -f dcgm-servicemonitor.yaml
    kubectl -n monitoring delete -f dcgm-exporter.yaml
    helm -n monitoring uninstall monitoring
    kubectl delete ns monitoring

What you should see in the end

    Prometheus targets for each GPU node’s DCGM exporter are UP.

    Grafana dashboards show GPU Util, Memory %, Temperature, Power per node/GPU.

    Alerts fire if temps exceed threshold, memory is saturated, or exporter scrapes fail.
```

## Section 18: Week 17: Model & Data Drift

### 114. 113. What Is Concept Drift vs Data Drift?
- Why drift matters
  - The assumption of training data == real-world data breaks down over time
- What is data drift
  - Statistical properties of your input features change over time
  - Visual inputs: image resolution, new camera sensors, lighting variations, updated mobile phone cameras
  - Text data: Vocabulary trends, shifts in language usage patterns
  - User demographics: distribution of age, location, and behavior profiles
- What is concept drift?
  - Relationship b/w inputs and outputs changes
  - Medical diagnosis: new disease
  - Fraud detection
  - Sentiment analysis: cultural context shifts
- Comparing data vs concept drift

Aspect | Data drift | Concept drift 
-------|------------|----------------
What changes | Input distribution | input-> label mapping
Detection methods | Statistical tests, histogram comparisons| Accuracy monitoring, error rate tracking
Fix approach | Update preprocessing, retrain on new data distribution | Retrain with newly labeled data, revise model logic
- Types of concept drift
  - Sudden drift: fraudsters' new tactics after a security patch
  - Increment drift: User preferences for content recommendation
  - Recurring drift: Seasonal shopping behavior
- Detecing drift
  - Data drift detection
    - Distribution monitoring with looks like EvidentlyAI
    - Statistical tests such as Kolmogorov-Smirnov test or Population Stability Index (PSI)
    - Feature histogram comparisons against training baseline
    - Dimensionality reduction to visualize high-dimensional drift
  - Concept drift detection
    - Accuracy tracking on labeled validation samples
    - Error rate monitoring with delayed ground truth
    - Prediction distribution analysis for shifts in output patterns
    - Model confidence metrics to identify uncertainty increases
- Real world examples
  - Data drift: location services by Covid19 lock-down
  - Concept drift: fraud detection
  - Both types: recommendation systems
- Why both matters
  - Data drift: your model sees "unfamiliar" inuts that it wasn't trained to handle
  - Concept drift: your model makes "wrong assumptions" about inputs
  - Both require:
    - Continuous monitoring systems
    - Alert thresholds and triggers
    - Retraining pipelines and strategies
    - Model versioning and rollback capabilities

### 115. 114. Why Drift Destroys AI Performance
- The nature of drift
  - ML model's fundamental assumption: training data and production data must follow similar patterns
  - When this assumption breaks, drift begins
- Performance degradation path
  - Training phase
  - Deployment
  - Drift accumulates
  - Business KPIs collapse
- Effects of data drift
  - Loss of predictive power
  - Real-world examples
    - New slang terms in NLP
    - Senior recalibration shifts numeric ranges
    - Image distributions change with camera upgrades
  - Consequences: higher error rates and algorithmic bias
- Effects of concept drift
  - Fraud detection
  - Consumer behavior
  - Financial markets: economic regime changes
- Hidden risks of drift
  - Silent failures damage your model's reputation
  - Compliance risk increases
  - Fairness issues as certain groups are disproportionally impacted
  - Computation waste
- Drift accelerators
  - Dynamic domains like finance, security, and healthcare domains
  - Cyclical patterns: seasonal or periodic data
  - User-generated content: evolving language, norms, and behaviors in social platforms
  - Feedback loops: chained actions
- Why drift is particularly dangerous
  - Gradual and invisible
  - Infrastructure resistant
  - Requires intervention
  - Erodes trust

### 116. 115. Tools for Drift Detection (EvidentlyAI)
- EvidentlyAI
  - An open-source Python library and dashboard
  - Data and concept drift
  - Model quality
  - Interactive reports
- Why EvidentlyAI?
  - Accessible and interpretable
  - Versatile deployment
  - Bridges teams
- Comprehensive drift dtection suite
  - Data drift report: comparison against baseline dataset
  - TArget drift report: monitors changes in the label distribution over time
  - Prediction drift report: tracks shifts in the model's output distributions
```py
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data = train_df,current_data=prod_df)
report.show(mode="inline")
```
- Powerful statistical foundation
  - Feature-level detection
  - Population stability index
  - Jensen-Shannon divergence
  - Kolmogorov-Smirnov test
- Seamless integration options
  - Data science notebooks
  - ML pipelines
  - Visualization dashboards
  - Alerting systems
- Beyond drift: comprehensive ML monitoring
  - Data quality checks
  - Target analysis: tracks class balanace shifts
  - Performance monitoring
  - Fairness & bias detection
- Strength and limitations
  - Strengths
    - Low adoption barrier
    -  Rich visual reports
    - Statistical rigor
    - Works with structured data
    - Open source flexibility
  - Limitations
    - Limited support for unstructured data like images and video
    - Requires external systems
    - Processing overhead
    - Some customization needs

### 117. 116. Real-Time Drift Monitoring Pipelines
- Drift is continuous, not occasional
- Realtime pipelines detect and alert as drift happens
- Critical for domains like fraud detection, financial trading, healthcare diagnostics
- Pipeline architecture
  - Data ingestion
  - Feature store/preprocessing
  - Drift detection engine
  - Metrics export
  - Dashboards and alerts
- Streaming data sources
  - User events
  - Sensor data
  - Inference logs
  - Embedding streams
- Drift detections in streaming
  - Statistical approach
    - Sliding window monitoring
    - Compare against reference baseline
    - Apply statistical tests on-the-fly to detect significant shifts
  - Common tests
    - KS test, PSI, chi-square for categorical features
    - Jensen-Shannon divergence for probability distributions
    - ADWIN, DDM for gradual vs sudden concept shift
- Prometheus/Grafana integration
  - Metric exposure: drift metrics exported as counters/gauges in Prometheus format
  - Visualization: Trend lines and heat maps
  - Alerting
- Scaling the pipeline
  - Enterprise-grade infrastructure
    - Use Flink/Spark streaming for large-scale drift checks across multiple models
    - Store drift logs in time-series databases for audit trails and retraining triggers
    - Integrate with CI/CD pipelines for automatically trigger retraining when thresholds are exceeded
- Best practices
  - Monitor both inputs + outputs
  - Window size selection
  - Automate feedback loops
  - Version control everything

### 118. 117. Human-in-the-Loop Drift Evaluation
- Why humans still matter
  - Fast and scalable
  - Context and judgement
  - Compliance and ethics
  - Not all drift requires action - humans provide the critical judgements
- The role of human review
  - Validates dift alerts before triggering costly retraining cycles
  - Distinguish real drift from natual data variability
  - Provides feedback loops for label updates
  - Approve retraining cycles in regulated industries
- Drift evaluation workflow
  - Automated detection
  - Flag for review
  - Human analysis
  - Decision
- Tooling for human-in-the-loop
  - Visualization dashboards
  - Feedback UI
  - Annotation tools
  - Ticketing integration
- Challenges in human-machine collaboration
  - Alert fatigue
  - Expertise gaps
  - Finding balance
  - Response time
- Best practices
  - Strategic automation: escalate only 20% of routine drift cases to human judgement
  - Actionable dashboards
  - Reviewer rotation
  - Decision documentation

### 119. 118. Mitigation Strategies – Retraining & Rebalancing
- Why mitigation is needed
  - Even the best models deteriorate over time
  - Drift is inevitable in production
  - Detection alone isn't enough. Must act
  - Goal: restore accuracy, fairness, and reliability
- Retraining basics
  - Collect new data that reflects current reality
  - Retrain model with updated distribution
  - Choose your approach
    - Full retraining
    - Incremental retraining
- When to retrain    
  - Performance metrics drop below SLA
  - Drift severity exceeds threshold
  - Major external events: pandemic, market crash, or regulatory changes
  - Periodic schedule: calendar based retraining as a preventive measure (daily, weekly, monthly)
- Rebalancing data
  - Oversampling
  - Undersampling
  - Synthetic data
- Active learning loop: when data is sparse or expensive to label, active learning (human in the loop) provides a targeted approach
  - Drift detection
  - Human labeling
  - Incremental retraining
  - Add to training: incorporate newly labeled data into training pool
- Trade-offs in mitigation strategies
  - Frequent retraining: better accuracy, faster adaptation but higher compute costs, resource intensive
  - Infrequent retraining: lower operation coss but risk drift build-up and performance degradation
  - Rebalancing: quick fix for imbalance issues but may not address fundamental distribution shifts
- Strategy selection must balance technical performance with business impact and budget
- Best practices for mitigation
  - Automate retraining triggers
  - Maintain baseline models
  - Monitor hoslistically
  - Log everything

### 120. 119. Lab – Build a Drift Detection Pipeline
- Goal: Detect data & concept drift in a simulated AI system using EvidentlyAI, Kafka (or mock stream), and export drift metrics to Prometheus/Grafana for monitoring.
```
0) Prereqs

    Python 3.9+

    pip install pandas scikit-learn evidently kafka-python prometheus_client

    Optional: Kafka cluster (local or Docker). If no Kafka → script falls back to mock batches.

    Prometheus + Grafana from Lab 112 (for metrics visualization).

1) Prepare Training (Reference) Data

    python scripts/prepare_reference.py

    Generates a reference dataset (synthetic classification data).

    Saves data/reference.csv (baseline distribution).

2) Simulate Streaming Data

Two options:

    With Kafka: produce drifting batches into topic inference-events.

    Without Kafka: script generates random batches with drift patterns (e.g., shifting mean, new categories).

    python scripts/stream_data.py

3) Drift Detection with EvidentlyAI

Run drift detection in a loop:

    python scripts/drift_monitor.py

    Consumes batches (Kafka or mock).

    Compares against reference dataset.

    Uses DataDriftPreset and PredictionDriftPreset.

    Exposes Prometheus metrics at http://localhost:8005/metrics.

4) Prometheus Integration

Edit prometheus.yml to scrape drift metrics:

    scrape_configs:
      - job_name: "drift-monitor"
        static_configs:
          - targets: ["drift-monitor:8005"]

Reload Prometheus → drift metrics now available.
5) Grafana Dashboard

Import grafana_dashboards/drift_dashboard.json.
Panels include:

    % Features Drifted

    Drift detected per feature

    Drift severity over time

    Alerts on sustained drift

6) Test the Pipeline

    Start with stable stream → dashboard shows low/no drift.

    Introduce synthetic drift (--drift-mode in stream_data.py) → Grafana shows feature drift rising.

    Alerts fire if thresholds breached.

7) Cleanup

    docker stop kafka zookeeper   # if using Docker Kafka
    pkill -f drift_monitor.py

📂 Folder Structure

    lab119_drift_pipeline/
     ├── data/
     │   └── reference.csv
     ├── scripts/
     │   ├── prepare_reference.py
     │   ├── stream_data.py
     │   └── drift_monitor.py
     ├── grafana_dashboards/
     │   └── drift_dashboard.json
     ├── README.md

✅ Next Steps

    Extend with concept drift detection using ground-truth labels.

    Connect retraining triggers → continuous training pipeline.

    Integrate with Lab 118 strategies for automated retraining & rebalancing.
```

## Section 19: Week 18: AI Security & Compliance

### 121. 120. Security Risks in AI Infrastructure
- Why AI security is different
  - Expanded attack surface: complex journey from data, features, models, inference, pipelines creates multiple vulnerable points
  - Infrastructure complexity: GPU clusters and specialized runtimes
  - Dual-nature threats: model can be both target and weapon (jailbreaks to bypass safety guardrails)
- Threat landscape
  - Data: theft, leakage, poisoning
  - Model: theft, inversion, membership inference attacks
  - Inference: prompt injection, adversarial inputs, Dos attacks
  - Infra: supply-chain vulnerabilities, secret exposure
- Data centric risks
  - Training data poisoning
  - Shadow datasets and PII (Personally Identifiable Information) Sprawl: untracked copies in S3/object storage lead to unmanaged sensitive data and potential regulatory violations
  - Weak access controls
  - Linkage attacks
- Model centric risks
  - Model extraction: response from API to create proprietary models, stealing intellectual property
  - Model inversion: Recover sensitive training samples by exploiting the model's memory of training data
  - Membership inference: detect user data in the training set, violating privacy expectations
  - Watermakr/backdoor triggers: Hidden patterns embedded in weights can be activated by specific inputs to force malicious behaviors
- Inference-time risks
  - Prompt injection/jailbreaks: crafted inputs that bypass safety guardrails in LLMs, enabling policy violations or system manipulation
  - Adversarial examples: subtly modified inputs that cause vision/speech models to make dangerous misclassifications
  - Tool/plugin abuse: exploting connected tools to accss unauthorized resources or exfiltrate secrets
  - DoS Attacks: Hot-path overload causing latency SLO breaches and service degradation
- Pipeline and supply chain vulnerabilities
  - Malicious model artifacts
  - Dependency attacks
  - Compromised training scripts
  - SBOM (Software Bill of Materials) Gaps
- Cloud & cluster risks
  - IAM vulnerabilty
  - Cloud metadata exposure: instance metadata theft
  - Kubernetes weaknesses
  - GPU security gaps
- Multi-tenancy and data isolation
  - Cross-tenant data access via shared feature stores/vector DBs exposing confidential information
  - Inference servers hosting many models creating noisy neighbors that can leak information
  - Inadequate namespace/RBAC controls allowing tenant boundary violations
  - Missing per-tenant resource quotas enabling DoS against shared services
  - Side-channel attacks against shared GPUs extracting model parameters
- Governance and compliance gaps
  - Missing audit trails
  - No data lineage
  - Key management issues
  - Sovereignty violations
- Indicators of compromise (IoCs)
  - Unusual inputs: spikes in atypical prompts or token usage patterns indicating probing attacks
  - Performance anomalies: unexpected latency/throughput changes on inference endpoints
  - Container/model integrity: unrecognized container digests or model hashes suggesting tampering
  - Access patterns: requests from unexpected principals/regions or outside normal hours
- Mitigation playbook (preview)
  - Data protection
    - Encryption, DLP scanning data minimization strategies
    - Cryptographically signed datasets with provenance
    - Senstive data detection in pre-processing pipeline
  - Model safeguards
    - Watermarking models to detect theft and misuse
    - Rate-limits on API usage to prevent extraction
    - Differential privacy
  - Inference protection
    - Content filters detecting malicious inputs
    - Input validation and sanitization frameworks
    - Real-time threat scoring for unusual requests
  - Infrastructure hardening
    - IAM least-privilege, mTLS b/w services
    - Signed container images with verified SBOMs
    - WAF protection for model APIs and admin interfaces
  - Operational excellence
    - Comprehensive logging, alerting, incident runbooks
    - Regular red-team exercises targeting AI components
- Design pricinples
  - Zero trust by default
  - Secure-by-default
  - Defense in depth
  - Measure and test

### 122. 121. Identity and Access Management (IAM)
- Why IAM matters
  - Protects sensitive data
  - Restricts model access
  - Prevents resource misuse
  - Ensures compliance
- IAM core concepts
  - Identity: users, service accounts, and roles
  - Authentication: verifying identity through SSO, API keys, certificates, and tokens
  - Authorization: enforcing what an authenticated identity can access or modify
  - Principles of least privilege: granting only the minimum permissions necessary for the task
- Cloud IAM examples
  - AWS IAM
  - GCP IAM
  - Azure RBAC: role-based access control
- Service accounts & workload identity
  - Use service accounts
  - Implement workload identity federation
  - Automate credential rotation
  - Never embed credentials
- Secrets management integration
  - IAM Authentication
  - Secret store
  - Protected assets
- Monitoring & auditing IAM
  - Access logging
  - Anomaly detection
  - Regular audits
- IAM threat scenarios
  - Accidental data exposure
  - Privilege escalation
  - Compromised service account
  - Pipeline tampering
- IAM Best practices
  - Policy design
    - Enforce least privilege + separation of duties
    - Use groups and roles over individual permissions
    - Break down monolithic policies into specific scopes
    - Document policy decisions and exceptions
  - Technical implementation
    - Rotate & expire credentials automatically
    - Use short-lived tokens instead of static keys
    - Integrate IAM into CI/CD & IaC templates
    - Conduct regular compliance reviews + audits

### 123. 122. Secrets Management for AI Systems
- AI stacks use many credentials across multiple systems
  - Cloud API keys
  - Database passwords and feature store tokens
  - Model registry & vector DB authentication
- Types of secrets in AI workflows
  - Data access: S3/GCS/Blob storage credentials, SQL DB passwords, NoSQL connection strings, data warehouse access tokens
  - Model access: Hugging face tokens, private model registry credentials, openAI API keys, vector embedding service authentication
  - Pipeline: CI/CD tokens, container registry credentials
  - Infrastructure: GPU cluster kubeconfigs, cloud IAM keys, provisioning credentials, monitoring system tokens
- Common pitfalls
  - Repository leaks: API keys and tokens committed to Github repo
  - Static credentials
  - Shared accounts
  - Environment exposure
- Secrets vaults and managers
  - HashiCorp Vault
  - AWS secrets manager
  - GCP secret manager
  - Azure Key Vault
- Kubernetes Secrets
  - Store small secretes as K8 Secrets objects
  - Enable encryption at rest with KMS backing
  - Implement strict RBAC policies for access
- Secret rotations and expiry
  - Automated rotation
  - Short-lived tokens
  - Dynamic secrets
  - Immediate expiry
- Integration with AI pipelines
  - Orchestration integration
  - Container security
  - Model access
  - Deployment security
- Monitoring & auditing secrets
  - Anomaly detection
  - Leak detection
  - Audit trails
  - Policy enforcement
- Best practices
  - No hardcoding
  - Centralize management
  - Automate lifecycle
  - Least privilege
  - Team training

### 124. 123. Data Encryption at Rest and In Transit
- Why encryption matters
  - Sensitive datasets
  - Intellectual property
  - Regulatory compliance: GDPR, HIPAA, SOC2
- Encryption at Rest
  - Disk-level
    - LUKS (Linux)
    - BitLocker (Windows)
    - Cloud provider managed encryption
  - Database level
    - Transparent Data Encryption (TDE)
    - Column-level encryption
    - Native DB security features
  - File/Object-level
    - S3 server-side encryption
    - GCS/Azure Blob encryption
    - KMS-managed keys
- Encryption in Transit
  - TLS (HTTPS): secure all REST/gRPC endpoints with TLS 1.2+ and strong cipher suites for public-facing APIs
  - Mutual TLS (mTLS): certificate-based mutual authentication b/w services in K8 clusters
  - Secure Tunnels: VPN/IPsec/TLS tunnels
- Key Management Systems (KMS)  
  - Centralized encryption key lifecycle
    - Creation
    - Distrbituion
    - Rotation
    - Revocation
  - Popular KMS solutions: AWS KMS, Google cloud KMS, Azure Key Vault, HashiCorp Vault
  - Always implement least-privilege IAM polices for KMS access
- Encryption in AI workflows
  - 256-bit AES encryption
  - TLS 1.3 
  - 100% pipeline coverage  
- End-to-end security architecture
  - Disk encryption
  - Monitoring
  - Transport encryption
  - Secrets & KMS
- Common encryption pitfalls
  - Insecuire transport
  - Storage misconfigurations
  - Weak cryptography
  - Static keys  
- Advanced encryption technologies
  - Homomorphic encryption
  - Secure enclaves
  - Client-side encryption
- Encryption best practices
  - Encryption by default
  - Managed KMS
  - Strong authentication
  - Continuous monitoring

### 125. 124. Model Theft and Adversarial Attacks
- Why this matters
  - Intellectual property
  - Security threats
  - Business impact
- Model theft (extraction attacks)
  - Adversary queries repeatedly to rebuild a replica by analyzing inputs and outputs
  - Common techniques
    - Output mimicking (block-box extraction)
    - API probing to map decision boundaries
    - Gradient-based extraction on exposed model parameters
- Model inversion attacks
  -Infers training data samples from model outputs by exploiting model memory
  - Higher risk areas:
    - Models trained on personally identifiable information (PII)
    - Medical records and patient data
    - Financial and transaction information
- Membership inference attacks
  - Adversaries determine whether specific records were part of the model's training dataset
  - Attack method: analyzes confidence scores and prediction patterns that reveal whether data was seen during training
  - Exploits overfitting: 
  - Regulatory implication: direct threat to privacy under GDPR, HIPAA, and other regulatory frameworks
- Adversarial examples: specially crafted inputs with subtle perturbations that cause AI models to make incorrect predictions
- Attack vector sin AI infrastructure
  - Inference APIs
  - Public datasets
  - Unsecured model artifacts
  - Third-Party Libraries
- Mitigation strategies: model theft
  - Rate limiting: query limits per user/IP
  - Watermarking and fingerprinting
  - Differential privacy
  - Anomaly detection
- Mitigation strategies: adversarial attacks
  - Adversarial training
  - Input validation
  - Randomization
  - Ensemble defenses: combine multiple models with different architectures to increase robustness through diversity
- Monitoring & detection
  - Proactive detection methods
  - Query distribution analysis
  - Boundary probing detection
  - Confidence validation
- Best practices
  - Asset protection
  - Access controls
  - Defense in depth
  - Security testing  

### 126. 125. Compliance Standards (GDPR, HIPAA, SOC2)
- Meeting regulatory and trust requirements
- Why compliance matters
  - Sensitive data protection
  - Regulatory safeguards
  - Risk mitigation
  - Trust foundation
- GDPR (General Data Protection Regulation)
  - Core principles
    - Lawfulness, fairness, transparency in data processing
    - Data minimization and purpose limitations to restrict collection
    - Rights of access, rectification, and erasure (right to be forgotten)
  - AI specific implications
    - Model explainability requirements
    - Explicit consent for automated decision-making
    - Algorithmic transparency obligations
- GDPR in AI infrastructure
  - End-to-end encryption
  - Comprehensive audit trails
  - Model explainability
  - Data subject rights
- HIPAA (Health Insurance Portability & Accountability Act)
  - Covered entities
    - Healthcare providers handling patient data
    - Health plans and insurers
    - Business associates processing PHI
  - Key requirements
    - Comprehensive safeguards
    - De-identification standards
    - Breach notification
- HIPAA in AI infrastructure
  - Encrypted PHI storage
  - Least privilege access
  - Vendor compliance
  - Comprehensive auditing
- SOC2 (Service Organization Control 2)
  - Voluntary compliance framework
  - Trust Service Principles (TSPs)
    - Security
    - Availability
    - Processing Integrity
    - Confidentiality
    - Privacy
- SOC2 in AI infrastructure
  - Access governance
  - Workload monitoring
  - Resilience planning
  - Enterprise readiness
- Challenges in AI compliance
  - Explainability Gap
  - Data residency
  - Unstructured content
  - Evolving standards
- Best practices across standards
  - Universal encryption
  - Comprehensive audit trails
  - Rigorous access management
  - Privacy by design
  - Regular validation

### 127. 126. Lab – Secure a Model Endpoint with Authentication
- Goal: Take a plain FastAPI inference API and secure it with:
  - JWT-based user authentication
  - API Key auth (for service-to-service calls)
  - Rate limiting
  - Deployment to Kubernetes with TLS support
```
Step 0 — Prerequisites

    Python 3.10+

    (Optional) Kubernetes cluster with an Ingress controller (e.g., NGINX)

    Docker if you want to containerize

Step 1 — Create the FastAPI app

app/main.py

    from fastapi import FastAPI, Depends, HTTPException, status, Security
    from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyHeader
    from pydantic import BaseModel
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from fastapi.responses import JSONResponse
    import time
     
    # FastAPI app + rate limiter
    limiter = Limiter(key_func=get_remote_address)
    app = FastAPI()
    app.state.limiter = limiter
     
    @app.exception_handler(RateLimitExceeded)
    def rate_limit_handler(request, exc):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
     
    # Demo secrets
    SECRET_KEY = "change-me"
    ALGORITHM = "HS256"
    API_KEY = "super-secret"
     
    # Password hashing
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
     
    # Demo users
    fake_users = {
        "alice": {"username": "alice", "hashed_pw": pwd_context.hash("alicepass")},
        "admin": {"username": "admin", "hashed_pw": pwd_context.hash("adminpass")}
    }
     
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
     
    class Token(BaseModel):
        access_token: str
        token_type: str
     
    class Features(BaseModel):
        features: list[float]
     
    def authenticate_user(username, password):
        user = fake_users.get(username)
        if not user or not pwd_context.verify(password, user["hashed_pw"]):
            return None
        return user
     
    def create_access_token(data: dict, expires_in: int = 3600):
        payload = data.copy()
        payload.update({"exp": time.time() + expires_in})
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
     
    async def get_current_user(token: str = Depends(oauth2_scheme)):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload.get("sub")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
     
    async def get_api_key(api_key: str = Security(api_key_header)):
        if api_key == API_KEY:
            return api_key
        raise HTTPException(status_code=403, detail="Invalid API key")
     
    # Routes
    @app.post("/token", response_model=Token)
    async def login(form: OAuth2PasswordRequestForm = Depends()):
        user = authenticate_user(form.username, form.password)
        if not user:
            raise HTTPException(status_code=401, detail="Bad credentials")
        token = create_access_token({"sub": form.username})
        return {"access_token": token, "token_type": "bearer"}
     
    @app.get("/healthz")
    def health():
        return {"status": "ok"}
     
    @app.get("/readyz")
    def ready():
        return {"status": "ready"}
     
    @app.post("/predict")
    @limiter.limit("30/minute")
    async def predict(
        data: Features,
        user: str = Depends(get_current_user),
        api_key: str = Depends(get_api_key)
    ):
        # Demo "model": sum of features
        score = sum(data.features)
        return {"prediction": "class_A" if score > 5 else "class_B"}

Step 2 — Install dependencies

    pip install fastapi uvicorn python-jose passlib[bcrypt] slowapi

Step 3 — Run locally

    export JWT_SECRET='change-me' API_KEY='super-secret'
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Test:

    curl -s http://localhost:8000/healthz

Step 4 — Get a JWT

    curl -s -X POST http://localhost:8000/token \
      -H "Content-Type: application/x-www-form-urlencoded" \
      -d "username=alice&password=alicepass"

Step 5 — Call the protected endpoint

Bearer token path

    TOKEN="<paste JWT>"
    curl -s -X POST http://localhost:8000/predict \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"features":[5.1,3.5,1.4,0.2]}'

API key path

    curl -s -X POST http://localhost:8000/predict \
      -H "X-API-Key: super-secret" \
      -H "Content-Type: application/json" \
      -d '{"features":[6.0,2.2,4.0,1.0]}'

Step 6 — Rate limiting

    Global default: 60/minute

    /predict endpoint: 30/minute

Exceed → 429 Too Many Requests
Step 7 — Containerize

Dockerfile

    FROM python:3.11-slim
    WORKDIR /app
    COPY app/ app/
    RUN pip install fastapi uvicorn python-jose passlib[bcrypt] slowapi
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

Build & run:

    docker build -t secure-endpoint .
    docker run -p 8000:8000 -e JWT_SECRET=change-me -e API_KEY=super-secret secure-endpoint

Step 8 — Deploy to Kubernetes

k8s/deployment.yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: secure-ml
      namespace: ai-sec
    spec:
      replicas: 1
      selector:
        matchLabels: { app: secure-ml }
      template:
        metadata:
          labels: { app: secure-ml }
        spec:
          containers:
          - name: api
            image: YOUR_REGISTRY/secure-endpoint:latest
            ports:
            - containerPort: 8000
            envFrom:
            - secretRef:
                name: jwt-secret
            - secretRef:
                name: api-key
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: secure-ml-svc
      namespace: ai-sec
    spec:
      selector: { app: secure-ml }
      ports:
      - port: 80
        targetPort: 8000

Create namespace + secrets:

    kubectl create namespace ai-sec
    kubectl -n ai-sec create secret generic jwt-secret --from-literal=JWT_SECRET='change-me'
    kubectl -n ai-sec create secret generic api-key --from-literal=API_KEY='super-secret'
    kubectl -n ai-sec apply -f k8s/deployment.yaml

(Optionally add an Ingress with TLS.)
Step 9 — Hardening checklist

    Use OIDC (Auth0, Cognito, Entra) instead of demo /token in production.

    Enforce HTTPS at ingress (consider mTLS internally).

    Store secrets in K8s Secrets or Vault, never in code.

    Apply RBAC and namespace isolation.

    Add DoS protection: WAF rules, tuned rate limits.

    Log everything: request IDs, user IDs, model version → ship to SIEM.

✅ End result: You have a model API protected with JWT, API keys, and rate limiting, containerized and deployable to Kubernetes.
```

## Section 20: Week 19: Reliability & High Availability

### 128. 127. Why AI Systems Fail in Production
- Uncovering weak points in real-world ML deployments
- The harsh reality
  - Deployment gap
  - Rapid degradation
  - Complex failure modes
  - Root cause
- Data related failures
  - Distribution drift
  - Quality degradation
  - Label errors
  - Pipeline breakages
- Model related failures
  - Overfitting
  - Bias amplification
  - Inadequate retraining
  - Model staleness: the world changes faster than model updates
- Infrastructure failures
  - Processing bottlenecks
  - Container misconfigurations
  - Latency issues
  - Cost explosions
- Monitoring & observability gaps
  - Untracked metrics
  - Absent alerting
  - Traceability deficits
  - Inadequate dashboards
- Security & compliance risks
  - Exposed endpoints
  - Secret leakage
  - Regulatory violations
  - Adversarial attacks
- Organizational failures
  - Siloed teams
  - Ownership gaps
  - Missing playbooks
  - Expectation management
- Case studies: when AI systems fail
  - Fraud detection: unexpected data drift
  - Customer support chatbot: evolving slang and terminology
  - Healthcare diagnostic tool: missing audit logs
  - Recommendation engine: Unannounced upstream schema change caused production crash

### 129. 128. Fault Tolerance in AI Inference
- Designing systems that survive failures gracefully
- Why fault tolerance matters
  - Revenue impact
  - User experience
  - Safety risks
- Sources of failure in AI inference systems
  - HW failures
  - Network issues
  - Software bugs
  - External dependencies
- Core design principles for fault-tolerant AI
  - Redundancy
  - Graceful degradation
  - Isolation
  - Resilience testing
- Load balancing & replication strategies
  - Deploy multiple replicas of inference servers behind a load balander
  - Implement auo-healing via K8, deployments with health probes
  - Scale horizontally with HPA based on CPU, memory, or custom metrics
  - For GPU intensive workloads, consider overprovisioning by 20-30% to handle unexpected traffic spikes and instance failures
- Fallback models: the safety net
  - Deploy shadow or lieghtweight models that can take over when primary models fail
- Circuit breakers: preventing cascading failures
  - Circuit breaker monitor failure rates and stop traffic to failing components
  - Track error rates and latency for each inference endpoint
  - When failures exceed threshold, "trip" the circuit
  - Reject new requests to failing service for a recovery period
- Retry & timeout strategies
  - Implement retries with exponential backoff
  - Set strict tiemouts
  - Monitor retry patterns
- Chaos engineering for AI systems
  - Systematically inject failures to validate resilience mechanism
  - Simulate HW failure
  - Create network issues
  - Induce resource constraints
  - Test dependency failures
- Best practices for fault-tolerant AI inference
  - Design for failure
  - Multi-region deployment
  - Automate everything
  - Regular testing
  - Business-aligned SLAs

### 130. 129. Redundancy and Failover for AI APIs
- Why redundancy matters
  - Customer experience and satisfaction
  - Revenue and business operations
  - Brand reputation and trust
- Redundancy fundamentals
  - Compute redundancy
  - Network redundancy
  - Storage redundancy
- Failover mechanism
  - Active-Active: all replicas simultaneously handle traffic with load balancing
    - Maximum resource utilization
    - Instant failover
    - Higher oeprational complexity
  - Active-Passive: Standby replicas activated when primary fails
    - Cost-effective redundancy
    - Slight longer recoverty time
    - Simpler implementation
  - Regional failover: traffic shifts to healthy region when primary region fails
    - Protects against regional outages
    - Requires multi-regional deployment
  - DNS failover: health checks trigger DNS record updates to reroute traffice
    - Slower but highly effective
    - Works across cloud providers
- Multi-zone/multi-region deployment
  - Deploy inference servers across multiple zones and regions
  - Store replicated models in regional caches to minimize latency
  - Configure automated traffic routing b/w healthy instances
  - Protect against natural disasters, power outages, ...
- Load balances in failover architecture
  - Container orchestration level
  - Cloud load balancer level
  - DNS routing level
- Database & feature store redundancy
  - Feature stores
    - Read replicas for scaling feature retrieval
    - Asynchronous replication for offline features
    - Synchronous replication for critical online features
  - Vector databases
    - Multi-zone deployments of vector DBs 
    - Sharded replicas for high-availability semantic search
    - Consistent embedding indexing across regions
  - Hot vs cold replicas: Hot replicas provide instant failover but at higher cost, while cold replicas offer more economical disaster recovery with longer recovery times
- Testing failover scenarios
  - Pod/node failure
  - Zone outage
  - Database failover
- Best practices for AI API redundancy
  - Implement robust health checks
  - Automate all failover procedures
  - Replicate models & features
  - Run regular chaos drills  

### 131. 130. Designing High-Availability AI Clusters
- What is High Availability (HA)?
  - Measured as nines of uptime
    - 99.9%: 8.76 hours downtime/year
      - Single-region, active-passive
      - Multi-zone deployment
      - Automated failover
    - 99.99%: 52.6 min downtime/year
      - Multi-region, active-active
      - Global load balancing
      - Real-time data replication
    - 99.999%: 5.26min downtime/year
      - Highly engineered redundancy
      - Complete infrastructure duplication
      - Instant failover mechanisms
- HA in AI context
  - Model accessibility
  - Training resilience
  - Inference stability    
- Core design principles
  - Redundancy
  - Isolation
  - Failover
  - Resilience testing
- Compute layer HA
  - K8 foundations
    - Deployments with replica sets (3+ minimum)
    - Horizontal/vertical pod autoscalers
    - Pod disruption budgets to maintain minimum availability
    - Dedicated node pools for critical ML workloads
  - Advanced configurations
    - GPU scheduling with node affinity and tolerations
    - Pod anti-affinity to spread across zones
    - Readiness/liveness probes for auto-healing
- Storage layer HA
  - Persistent storage
    - Replicated storage backends: Ceph, EBS, GCP PD
    - Multi-attach volumes where supported
    - Storage classes with appropriate redundancy
  - ML data storage
    - Data lakes and vector dB in multi-region mode
    - Feature stores with hot replicas
    - Distributed fiel systems (HDFS, GCS, S3)
  - Recovery mechanisms
    - Continuous snapshots and backups
    - Point-in-time recovery options
    - Cross-region replication for critical data
  - Network layer HA
    - Intelligent load balancing
    - Resilient ingress
    - Global routing
    - Dedicated connectivity
- HA for training
  - Checkpoint management
  - Elastic training frameworks
  - Cost-effective resilience
- HA for inference
  - Multi-replica serving
  - Safe deployment strategies: canary + blue/green rollouts
  - Advanced resilience: latency based autoscaling, circuit breakers for dependent services
- Observability in HA clusters
  - Key metrics to monitor
    - Uptime per service/endpoint
    - Failover events and recovery time
    - Latency spikes during degraded states
    - Replica count mismatches or unhealthy nodes
    - Resource utilization during failure scenarios
  - Chaos engineering integration
    - Regularly inject failures to validate your HA design
      - Node terminations
      - Network partitions
      - Disk failures
      - Resource exhaustion
- Best practices
  - Design for failure as default
  - Distribute across failure domains
  - Automate recovery processes
  - Validate with real-world testing
  - Right-size your HA investment

### 132. 131. Auto-Healing Infrastructure with Kubernetes
- With proper auto-healing implementation, your infrastructure can:
  - Maintain higher uptime to meet strict SLAs
  - Build resilience against pod and node crashes
  - Recover faster with minimal human intervention
- How K8 enables auto-healing 
  - Controllers: reconcile desired state vs. actual state of the system
  - Pod recovery
  - Node replacement
  * The combination of health probes and controllers creates a truly self-healing system
- Pod-level healing
  - RestartPolicy options
    - Always
    - OnFailure
    - Never
- Node level healing
  - Node controller monitors node heartbeats to detect failures
  - When a node goes down, K8:
    - Marks the node a unhealthy
    - Evicts pods from the failed node
    - Reschedules workloads t healthy nodes
  - Cloud provider integration
    - AWS auto scaling groups
    - GCP node auto-repair
    - Azure VMSS auto-scaling
- Probes for health checks
  - Liveness probe: detects if a pod is stuck or unresponsive
  - Readiness probe: determines if pod can receive traffic
  - Startup probe: special check for slow-starting AI workloads
- StatefulSets & Persistent Apps
  - Identity Preservation
  - Storage Persistence
  - Ordered Operations
- Chaos testing auto-healing
  - Pod & node termination
  - Resource contention
  - Network partitions
  * Popular tools: Chaos Mesh, LitmusChaos, Gremlin
- Auto-healing in AI workloads
  - Training: check points + job restart, failed nodes automatically replaced by cloud provider
  - Feature stores: StatefulSets maintain data integrity during crashes. Failed tasks retried by workflow engines (Airflow, Kubeflow)
  - Inference: Quick pod replacement ensures low API downtime, multiple relicas across zones for regional resilience
- Best practices
  - Define effective probes
  - Conduct chaos testing
  - Implement worload disribution
  - Monitor healing events
  - Enable cloud integration
  - Use resource limits

### 133. 132. Chaos Engineering for AI Systems
- Why chaos engineering?
  - AI infrastructures are complex and distributed, creating numerous potential failure points
  - Inevitable failures
  - Proactive confidence: simulate failures before they happen in production
- Core principles of chaos engineering
  - Define steady state
    - Establish normal operational metrics such as inference latency, throughput, model accuracy/performance, resource utilization baselines
  - Inject controlled failure
  - Observe system response
  - Improve and repeat
- Chaos targets in AI systems
  - Infrastructure
  - Network
  - Data pipeline
  - Model serving
- Tools for chaos engineering
  - Chaos Mesh: pod/node failures, network chaos, IO chaos
  - LitmusChaos: application-level faults, node-level faults
  - Gremlin: fine-grained blast radius, advanced scheduling
  - Istio/Envoy: HTTP error codes, request delays
- Best practices
  - Start small, scale gradually
  - Automate your chaos
  - Define clear abort conditions
  - Measure MTTR (Mean Time To Recovery)
- AI-specific challenges
  - Beyond uptime metrics
  - Silent drift dtection
  - Complex blast radius
  - Integrated observability

### 134. 133. Lab – Build a HA AI Inference Cluster
- Goal: Deploy a multi-replica, fault-tolerant inference service on Kubernetes with:
  - Load-balancing
  - Auto-healing pods
  - Health probes
  - Multi-zone deployment (optional)
```
Step 0 – Prereqs

    Kubernetes cluster (≥ 2 nodes, ideally across zones)

    kubectl and helm installed

    Container image from Lab 126 (or any FastAPI/Triton inference app)

Step 1 – Namespace

    kubectl create namespace ai-ha

Step 2 – Deployment (Multiple Replicas)

    # k8s/deployment.yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: ha-inference
      namespace: ai-ha
    spec:
      replicas: 3                # ensure redundancy
      selector:
        matchLabels: { app: ha-inference }
      template:
        metadata:
          labels: { app: ha-inference }
        spec:
          containers:
          - name: model-api
            image: YOUR_REGISTRY/secure-ml:latest   # replace with your image
            ports:
            - containerPort: 8000
            readinessProbe:
              httpGet: { path: /readyz, port: 8000 }
              initialDelaySeconds: 5
              periodSeconds: 5
            livenessProbe:
              httpGet: { path: /healthz, port: 8000 }
              initialDelaySeconds: 15
              periodSeconds: 10
            resources:
              requests: { cpu: "250m", memory: "512Mi" }
              limits:   { cpu: "500m", memory: "1Gi" }

Apply:

    kubectl -n ai-ha apply -f k8s/deployment.yaml

Step 3 – Service (Load Balancer)

    # k8s/service.yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: ha-inference-svc
      namespace: ai-ha
    spec:
      selector: { app: ha-inference }
      ports:
      - name: http
        port: 80
        targetPort: 8000
      type: LoadBalancer

Apply:

    kubectl -n ai-ha apply -f k8s/service.yaml

Step 4 – Horizontal Pod Autoscaler (HPA)

    kubectl -n ai-ha autoscale deployment ha-inference \
      --cpu-percent=70 --min=3 --max=10

    Ensures scaling under high load.

    Keeps ≥3 pods always alive.

Step 5 – Pod Disruption Budget (PDB)

    # k8s/pdb.yaml
    apiVersion: policy/v1
    kind: PodDisruptionBudget
    metadata:
      name: ha-inference-pdb
      namespace: ai-ha
    spec:
      minAvailable: 2
      selector:
        matchLabels:
          app: ha-inference

Apply:

    kubectl -n ai-ha apply -f k8s/pdb.yaml

This ensures at least 2 pods remain during maintenance or upgrades.
Step 6 – Multi-Zone Scheduling (Optional)

Add anti-affinity rules so pods spread across zones/nodes:

          affinity:
            podAntiAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
              - labelSelector:
                  matchExpressions:
                  - key: app
                    operator: In
                    values: [ "ha-inference" ]
                topologyKey: "kubernetes.io/hostname"

Step 7 – Chaos Test

Kill one pod:

    kubectl -n ai-ha delete pod <pod-name>
    kubectl -n ai-ha get pods -w

    Deployment immediately recreates pod.

    Service routes only to healthy pods (readiness probe).

Step 8 – Validation

    Get service IP:

    kubectl -n ai-ha get svc ha-inference-svc

    Send multiple requests, watch load balance across pods:

    while true; do \
    curl -s http://<svc-ip>/healthz; \
    sleep 1; \
    done

    Test HPA by running a load generator (e.g., hey or ab).

Step 9 – Cleanup

    kubectl delete namespace ai-ha

📂 Folder Structure

    lab133_ha_cluster/
     ├── k8s/
     │   ├── deployment.yaml
     │   ├── service.yaml
     │   └── pdb.yaml
     └── README.md

✅ With this lab, you now have a self-healing, load-balanced, auto-scaling inference cluster ready for production chaos.
Learning
```

## Section 21: Week 20: Multi-Cloud AI Infrastructure

### 136. 134. Why Enterprises Go Multi-Cloud
- Why multiple clouds matter
  - Diversified excellence
  - Risk distribution
  - Service optimization
  - Global complicance
- Business drivers for multi-cloud strategy
  - Resilience and continuity
  - Vendor leverage
  - M & A reality  
  - Regulatory compliance
  - Proximity performance
- Technical drivers: AI-specific considerations
  - GPU Capacity and choice
  - Specialized services
  - Performance optimization
  - Hybrid alignment
- When multi-cloud is not worth it
  - Limited operational maturity
  - Deep PaaS integration
  - Data gravity constraints
  - Simplified compliance
  * Cost/benefit analysis: multi-cloud only makes sense when benefits clearly outweight the combined additional complexity + cost
- Data strategy: the hard part of multi-cloud
  - Data gravity challenge
  - Replication tradeoffs
  - Open table formats
  - Change data capture
  - Egress optimization
- Networking & identity: connecting your clouds
  - Private connectivity
  - Zero-trust security
  - Unified authentication
  - Intelligent DNS
  - IP Management
- Platform and orchestration across clouds  
  - K8 as Portability layer
  - GitOps for declarative management
  - Multi-cloud control planes
  - Service Mesh federation
  - Artifact Mirroring
- Observability & security: cross-cloud visibility
  - Unified metrics
  - Distributed tracing
  - Security standardization
- AI workload placement strategies
  - Training workloads
  - Inference services
  - Feature store distribution
  - Model registry strategy
- Multi-cloud migration roadmap
  - Standardize infrastructure
  - Pilot program
  - Shard services foundation
  - Data migration
  - Validation exercises
- Pitfalls & anti-patterns to avoid
  - Copy-paste architecture
  - Ignoring economics
  - Configuration drift
  - Operational underinvestment

### 137. 135. Kubernetes Federation Across Clouds
- Why federation?
  - Running K8 across multiple clouds
  - Consistent Deployments
  - Multi-region DR
  - Global traffic routing
- KubeFed (Kubernetes Federation)
  - Cloud Native computing foundation project designed to coordinate management of multiple Kubernetes
  - Resource synchronization
  - Policy-driven placement
  - Multi-cloud & hybrid support: Works across AWS, GCP, Azure
- Key features
  - Multi-cluster deployments
  - Selective placement
  - Failover & migration
  - Global service discovery
- Multi-cloud AI use case
  - Training in Cloud A
  - Shared ML resources
  - Serving in Cloud B
- Traffic management
  - Federated services
  - Multi-cloud DNS
  - Geo-routing
  - Automatic failover
- Federation alternatives
  - Google Anthos
  - Azure Arc
  - Rander/OpenShift
  - Istio Federation
- Challenges
  - Operational Complexity
  - Network latency
  - Identity management
  - Debugging complexity
- Best practices
  - Start small
  - GitOps integration
  - Selective Federation
  - Cross-Cluster Observability
  - Data locality
  - Disaster Recoverty Testing

### 138. 136. Data Replication Across Cloud Providers
- Why replicate data?
  - Disaster recovery
  - Low-latency access
  - Compliance
  - Cross-cloud pipelines
- Key challenges  
  - Data gravity
  - Latency & consistency
  - Egress fees: cross-cloud data transfer costs 
  - Schema drift: schema inconsistencies b/w cloud systems may corrupt data integrity
  - Security
- Replication models
  - Synchronous replication
    - Real time consistency b/w sources
    - Transactions commit only after all replicas confirm
    - Higher latency and operational costs
    - Best for critical transaction data
  - Asynchronous replication
    - Eventual consistency model
    - Primary system commits, secondaries update later
    - Better performance and lower costs
  - Hybrid replication
    - Sync for metadata and critical configs
    - Async for bulk data and large asset
    - Balances performance and consistency
    - Ideal for most ML production systems
- Storage replication options
  - Object storage: S3 <-> GCS <-> Azure Blob
    - CloudSync, rconle, storage gateway  
    - Native bucket replication
    - CDN edge caching for static assets
  - Databases
    - Cloud spanner multi-region
    - Aurora Global Database
    - Cosmos DB multi-region write
    - MongoDB Atlas multi-cloud clusters
  - Data Lakes: portable open formats
    - Apache Iceberg
    - Delta Lake
    - Apache Hudi
    - Log-based incremental replication
  - Streaming: real-time data movement
    - Kakfa MirrorMaker2
    - Pub/Sub <-> Kafka bridges
    - Confluent Cloud multi-cloud
- Feature store and vector DB replication: ML specific data systems
  - Feature stores
    - Feast, Tecton with multi-cloud storage backends
    - Consistent feature values across clouds
    - Offline/online store replication
  - Vector databases
    - Pinecone, Milvus, Weaviate with active-active replication
    - Embedding consistency for distributed inference
    - Read replicas for vector search
- Network & security considerations
  - Private interconnects: reduced latency and egress costs using AWS Direct Connect, Google Cloud Interconnect, Azure ExpressRoute
  - Encryption requirements
  - Monitoring and alerting
- Trade-offs
  - Performance vs cost
  - Comliance vs agility
  - Ops complexity
- Best practices
  - Use portable data formats
  - Replicate critical subsets
  - Monitor replication lag
  - Align with AI workloads
  - Test DR regularly

### 139. 137. Cost Arbitrage in Multi-Cloud AI
- What is cost arbitrage?
  - Strategic multi-cloud: take advantage of pricing differences and specialized offerings
  - Balance factors
  - Critical for AI: valuable for GPU-intensive AI workloads
- Why it matters for AI
  - High-cost components
  - Pricing variability
    - Price changes by regions/vendors
  - Discount opportunities: spot instances, preemptible VMs
- Storage and network arbitrage
  - Object storage: GCS offers better pricing for archival while S3 has better ecosystem tool integration
  - Block storage: performance/price ratios vary by 15-25% across providers
- Network costs: network egress is often the hidden cost driver
- Multi-cloud arbitrage strategies
  - Bursting
  - Workload splitting
  - Spot/preemptible VMs
  - SaaS arbitrage
- Tooling for arbitrage
  - Infrastructure as code
  - K8 Federation
  - Cost monitoring
  - Automated bidding
- Risks & challenges
  - Egress costs
  - Operational complexity
  - Migration overhead
  - Monitoring discipline
- Best practices
  - Calculate total cost of ownership
  - Automate placement decisions
  - Develop egress-aware data strategy
  - Leverage for vendor negotiations

### 140. 138. Disaster Recovery Across Regions
- Why DR matters for AI
  - Business-critical applications
  - Extended outage risk
  - Multiple protection layers
- Threats that trigger DR
  - Cloud region outage
  - Data corruption
  - Infrastructur compromise
  - Compliance issues
- DR objectives: setting clear recovery targets
  - RTO (Recovery Time Objective): How fast can your AI system recover after disaster?
    - Financial fraud model: < 15min
    - Recommendation engine: < 2 hours
  - RPO (Recover Point Objective): How much data loss is acceptable during recovery?
    - Financial fraud model: near zero
    - Recommendation engine: 1-2 hours
  * Validate your RPO/RTO objectives through scheduled replication tests and controlled failover exercises
- DR architecture: finding your recovery balance
  - Cold standby: cheapest option with manual spin-up of resources
    - Minimal ongoing costs
    - Longest recovery time
    - Best for non-critical AI workloads
  - Warm standby: Pre-provisioned but scaled-down resources
    - Moderage ongoing costs
    - Medium recovery time
    - Balance of cost vs readiness
  - Hot active-active: full replicas operating in multiple regions
    - Highest ongoing costs
    - Near-immediate recovery
    - For mission-critical AI systems
  - Hybrid: training on one region, inference replicated
    - Optimized cost profile
    - Prioritizes customer-facing components
    - Balances performance and protection
- Data replication fo rDR
  - Object storage
  - Databases
  - Feature stores & vector DBs
- Networking and routing: seamless traffic redirection
  - Global DNS failover
  - Anycast load balancing
  - Network redundancy
  - Certificate management
- DR for AI pipelines:
  - Training jobs
  - Model registry
  - Inference APIs
  - Shadow deployments
- Testing & chaos drills
  - Regional outage simulation
  - Failover validation
  - Data integrity checks
  - Team readiness
- Compliance & governance considerations
  - Data residency constraints
  - Audit trail continuity
  - Certification requirements
  - Data protection
- Best practices: building resilient AI infrastructure
  - Define clear recovery objectives: RTP/RPO targets for each AI workload
  - Automate failover processes
  - Replicate the full ML stack
  - Regular testing
  - Cost vs risk balance

### 141. 139. Hybrid On-Prem + Cloud AI Setups
- Why hybrid AI?
  - Data residency and compliance
  - Latency senstive inference
  - Cost control
  - Burst workloads
- Typical hybrid patterns
  - Training on-perm, serving in cloud
  - Inference on-prem/edge, retraining in cloud
  - Cloud bursting
  * Split by data sensitivity
- On-prem components
  - Compute infrastructure
  - Storage
  - Orchestration
  - Security
- Cloud components
  - Elastic GPU training: AWS Sagemaker, Google Cloud Vertex AI, Azure Machine Learning
  - Scalable model serving
  - Global distribution
  - Managed data services
- Networking and connectivity
  - Private interconnects to vendors
  - VPN tunnels
  - Service-to-Service Auth
  - Unified Service Discovery
- Data flow example
  - Data collection: raw data stored on-prem for compliance requirements
  - Feature engineering: features pushed to cloud feature store
  - Model training: In cloud with elastic resources
  - Deployment: On-prem for local inference
- Hybrid AI challenges
  - Data gravity
  - Observability gaps
  - Security inconsistency
  - Performance impact
  - Vendor lock-in risk
- Tools & frameworks: enable seamless hybrid AI operations
  - Kubeflow
  - MLflow
  - Service mesh
  - Data fabrics
  - Hybrid K8 Management
- Best practices
  - Data locality
  - Consistent infrastructure
  - Unified security
  - End-to-end observability
  - Resilience planning

### 142. 140. Lab – Serve a Model Across AWS + GCP
- Goal: Deploy the same AI inference service on AWS EKS and GCP GKE, fronted by a global DNS layer that routes requests to the nearest healthy endpoint.
```
Step 0 – Prereqs

    AWS + GCP accounts with permissions for Kubernetes clusters.

    kubectl, helm, eksctl, and gcloud CLI installed.

    Docker image (e.g., from Lab 126’s FastAPI inference app).

Step 1 – Build and Push Docker Image

Build once, push to both registries:

    # Build
    docker build -t fastapi-inference:multi .
     
    # Tag & push to AWS ECR
    aws ecr create-repository --repository-name fastapi-inference
    AWS_URI=<account>.dkr.ecr.<region>.amazonaws.com/fastapi-inference:latest
    docker tag fastapi-inference:multi $AWS_URI
    docker push $AWS_URI
     
    # Tag & push to GCP Artifact Registry
    gcloud artifacts repositories create inference --repository-format=docker --location=us-central1
    GCP_URI=us-central1-docker.pkg.dev/<project-id>/inference/fastapi-inference:latest
    docker tag fastapi-inference:multi $GCP_URI
    docker push $GCP_URI

Step 2 – Create Clusters

AWS EKS

    eksctl create cluster --name ai-eks --region us-east-1 --nodes 3

GCP GKE

    gcloud container clusters create ai-gke --region us-central1 --num-nodes 3

Step 3 – Deploy App to Each Cluster

Use the same Kubernetes manifests with different image URIs.

deployment.yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: inference-api
    spec:
      replicas: 3
      selector:
        matchLabels: { app: inference-api }
      template:
        metadata:
          labels: { app: inference-api }
        spec:
          containers:
          - name: api
            image: <CLOUD_IMAGE_URI>   # AWS or GCP image URI
            ports:
            - containerPort: 8000
            readinessProbe:
              httpGet: { path: /readyz, port: 8000 }
            livenessProbe:
              httpGet: { path: /healthz, port: 8000 }

service.yaml

    apiVersion: v1
    kind: Service
    metadata:
      name: inference-svc
    spec:
      type: LoadBalancer
      selector: { app: inference-api }
      ports:
      - port: 80
        targetPort: 8000

Apply to each cluster:

    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml

Step 4 – Get Load Balancer IPs

AWS:

    kubectl get svc inference-svc
    # EXTERNAL-IP: a1b2c3.amazonaws.com

GCP:

    kubectl get svc inference-svc
    # EXTERNAL-IP: 34.12.45.67

Step 5 – Configure Global DNS

Use AWS Route53 or Cloudflare to set a geo-aware DNS record:

    inference.mycompany.com → routes to AWS LB in us-east-1 for U.S. users

    → routes to GCP LB in us-central1 for others

Example (Route53 latency-based policy):

    Record 1: Name=inference.mycompany.com, Value=a1b2c3.amazonaws.com, Region=us-east-1

    Record 2: Name=inference.mycompany.com, Value=34.12.45.67, Region=us-central1

Step 6 – Test Multi-Cloud Inference

    curl -s http://inference.mycompany.com/healthz
    curl -s -X POST http://inference.mycompany.com/predict \
      -H "Content-Type: application/json" \
      -d '{"features":[5.1,3.5,1.4,0.2]}'

Test from different regions (or VPNs) → traffic should hit the nearest cluster.
Step 7 – Simulate Failover

    Kill pods in AWS:

    kubectl delete pods -n default -l app=inference-api

    DNS automatically shifts to GCP endpoint.

    Validate traffic continues to flow.

Step 8 – Cleanup

    eksctl delete cluster --name ai-eks
    gcloud container clusters delete ai-gke --region us-central1

✅ Learning Outcomes

    Deploy an AI inference service on two cloud providers.

    Configure multi-region/multi-cloud DNS failover.

    Validate resilience & failover of HA inference APIs.
```

## Section 22: Week 21: Edge AI Infrastructure Basics

### 143. 141. What Is Edge AI and Why It Matters
- Edge AI defined
  - Running ML on/near the data source
  - Real world examples: cameras, robots, kiosks, vehicles, factory lines, retail stores, hospital room
  - Key advantage: Low latency, high BW
- Four drivers of Edge AI
  - < 100 ms latency
  - TB/day bandwidth
  - GDPR(General Data Protection Regulation) privacy
  - 99.9% resilience
- Typical edge AI technology stack
  - HW: Nvidia Jetson, ARM SoCs
  - OS/runtime: Linux, JetPack, OpenVINO, CUDA
  - Modelformat: ONNX, TensorRT engine, TFLite, core ML
  - App & Management: gRPC/REST services, Local message bus, OTA updates
- Use cases
  - Vision
  - Audio/NLP
  - Industrial/IoT
  - Mobility
- Edge vs Cloud: complementary roles
  - Edges: handles real-time inference, filtering, and pre-aggregation of data closer to the source
  - Cloud: manages heavy training workloads, global coordination, model registry, and complex analytics
  - Synchronization: Edge sends telemetry up; cloud sends optimized models and policies down
- Navigating Edge Constraints
  - Compute/memory/power limitations
  - Intermittent connectivity
  - Heterogeneous HW
  - Physical security concerns
- Model optimization toolkit
  - Quantization
  - Pruning & distillation
  - Accelerator-aware compilers
  - Efficient pipelines
- Deploying and managing Edge AI at scale
  - Packaging: containerize models and application code using Docker
  - Orchestration: using lightweight K8 variants
  - Updates
  - Observability
- Edge AI security essentials
  - HW security
  - Communication security
  - Identity management
  - Supply chain security
- Cost & ROI considerations
  - Cost reduction strategies
  - HW efficiency
  - Hybrid analytics: upload processed features and events rather than raw media
  - Key metrics: Measure ROI through latency SLOs, accuracy, uptime and cost per inference

### 144. 142. Use Cases: Retail, Healthcare, Smart Cities
- Why industry-specific Edge AI?
  - Unique drivers: specific requirements for latency, privacy, cost, and safety
  - Local inference
  - Shared benefits: real-time decision making, reduced cloud dependencies
- Retail use cases
  - Checkout-free stores
  - Customer analytics
  - Inventory management
  - Personalization
- Retail drivers: why Edge computing matters
  - Latency
  - Privacy
  - Cost
  - Customer experience
- Healthcare use cases
  - Bedside monitoring
  - Medical imaging
  - Smart devices
  - Telemedicine
- Healthcare drivers: where Edge AI becomes mission-critical
  - Compliance
  - Reliability
  - Safety
  - Integration
- Smart cities use case
  - Traffic management
  - Public safety
  - Smart lighting
  - Environmental monitoring
- Smart city drivers
  - Thousands of interconnected endpoints, generating petabytes of data daily
  - < 100 ms latency
  - 90% bandwidth reduction
  - 100% data sovereignty
- Common challenges across sectors
  - HW heteerogeneity
  - Security & trust
  - Model optimization
  - Fleet management

### 145. 143. Introduction to NVIDIA Jetson Devices
- Nvidia Jetson
  - ARM CPUs + CUDA-capable GPUs ona compact module
  - For vision applications, robotics, and autonomous systems that require real-time AI processing at the edge
- Why Jetson for Edge AI?
  - GPU acceleration
  - Compact and power-efficient
  - Scalable family
  - JetPack SDK
- Jetson family overview
  - Nano: entry-level platform
  - TX2: mid-tier solution
  - Xavier NX: professional edge deployments
  - AGX Xavier: advanced autonomous capabilities
  - Orin series: latest generation
- Jetson SW ecosystem
  - JetPack SDK
  - DeepStream SDK
  - TAO toolkit
  - Isaac SDK
- Edge AI use cases with Jetson
  - Computer vision
  - Robotics
  - Healthcare
  - Retail/smart cities
- Why Jetson over alternatives
  - GPU acceleration
  - CUDA/TensorRT ecosystem
  - Nvidia community and support
  - Scalability
- Deployment at scale
  - HW integration
  - Fleet management
  - SW updates
  - Security

### 146. 144. Installing JetPack SDK on Jetson
- JetPack SDK
  - Ubuntu-based OS image
  - CUDA, cuDNN, TensorRT
  - DeepStream SDK
  - Multimedia APIs and drivers
- Why JetPack
  - Optimized for Jetson HW
  - Maximum GPU performance
  - Simplified development
  - Regular updates
- Installation options
  - Option 1: Nvidia SDK manager
  - Option 2: Pre-flashed developer kits    
- Installation steps
    1. Prepare jetson device
        - Nano, Xavier, Orin
    2. Install nvidia sdk manager
        - On your ubuntu host PC
    3. Flash Jetson with JetPack
        - Connect HW
        - Recovery mode
        - SDK manager setup
        - Begin flashing
    4. First boot setup
    5. Verify JetPack Installation
        - nvcc --version
        - Check TensorRT runs at Python
- Post-install extras
  - AI frameworks
  - Jetson.GPIO
  - Performance tuning
  - Jetson Zoo
- Common issues & solutions
  - Power problems: Use nvida-recommended power adapter
  - Storage limitations: Use 32GB+ class 10 microSD or lager eMMC module
  - SW compatibility: update SDK manager

### 147. 145. Edge vs Cloud Tradeoffs for AI Deployment
- Why right balance matters
  - Latency
  - Cost & scalability
  - Privacy and compliance
  - User experience
- Edge AI advantages
  - Ultra-low latency
  - Offline operation
  - Privacy by design
  - Bandwidth savings
- Edge AI limitations
  - Resource constraints
  - Deployment challenges
  - Device heterogeneity
  - Security vulnerabilities
- Cloud AI advantages
  - Elastic scalability
  - Centralized model registry
  - Fleet-wide management
  - Advanced HW access
- Cloud AI limitations
  - Latency constraints
  - Cost inefficiencies
  - Privacy vulnerabilities
  - Network dependencies
- Hybrid approach: best of both worlds
  - Edge capabilities
    - Real-time inference with minimal latency
    - Data filtering and pre-processing
    - Privacy-preserving anonymization
    - Continued operation during connectivity gaps
  - Cloud capabilities
    - Model training on aggregated datasets
    - Centralized model registry and versioning
    - Cross-device analytics and insights
    - Orchestration and fleet management

### 148. 146. ONNX Runtime for Edge Models
- ONNX (Open Neural Network Exchange)
  - Open standard 
  - Framework support: Pytorch, TensorFlow, scikit-learn, and Hugging-Face
  - Interoperability
- ONNX runtime overview
  - x86, ARM, GPUs, NPUs, Jetson boards, mobile and embeded devices
  - HW acceleration (CUDA, TensorRT, DirectML)
  - Quantization for efficiency
  - Cross-language APIs: Python, C++, Java, C#
- Why ONNX for Edge AI?
  - Portability
  - Performance
  - Flexibility
  - Compatibility: integrates with TensorRT, OpenVINO, Core ML and other edge platforms
- Edge HW acceleration
  - Nvidia Jetson: TensorRT 
  - Intel CPU/VPU: OpenVINO execution provider
  - iOS devices: CoreML execution provider
  - Android devices: NNAPI execution provider
- ONNX optimizations
  - Quantization
  - Graph optimization
  - Execution providers
- For python, pip install onnx onnxruntime

### 149. 147. Lab – Deploy YOLOv5 on Jetson Nano
- Goal: Run a real-time object detection model (YOLOv5) on a Jetson Nano, optimized with TensorRT for edge performance.
Step 0 – Prerequisites
  - Jetson Nano Developer Kit (4GB preferred)
  - JetPack SDK installed (Day 144)
  - 16–32 GB microSD card or eMMC
  - USB webcam or CSI camera (Raspberry Pi Camera v2)
  - Internet connectivity
```
Step 1 – Environment Setup

    Update system:

    sudo apt update && sudo apt upgrade -y

    Create a Python environment (recommended):

    python3 -m venv yolov5-env
    source yolov5-env/bin/activate

    Install dependencies:

    sudo apt install -y python3-pip git
    pip install --upgrade pip setuptools wheel
    pip install numpy torch torchvision matplotlib onnx onnxruntime

Step 2 – Clone YOLOv5 Repository

    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip install -r requirements.txt

Step 3 – Test Inference (CPU/GPU PyTorch)

Run a quick detection test:

    python detect.py --source 0 --weights yolov5s.pt --conf 0.4

    --source 0 = webcam

    --weights yolov5s.pt = small YOLOv5 model (fastest)

✅ You should see bounding boxes on live video.
Step 4 – Export to ONNX

Convert YOLOv5 PyTorch model → ONNX format:

    python export.py --weights yolov5s.pt --include onnx

Output: yolov5s.onnx
Step 5 – Optimize with TensorRT

Use NVIDIA TensorRT for faster inference on Jetson GPU:

    python export.py --weights yolov5s.pt --include engine

    Creates a .engine file (TensorRT optimized)

Step 6 – Run Inference with TensorRT

    python detect.py --source 0 --weights yolov5s.engine --conf 0.4

    Expect 2–4× speedup vs PyTorch on Jetson Nano.

Step 7 – Measure Performance

Benchmark inference speed:

    python detect.py --source data/images --weights yolov5s.engine --conf 0.4 --save-txt

    Compare FPS with PyTorch vs TensorRT

    Typical: ~5 FPS (PyTorch) → ~15 FPS (TensorRT optimized)

Step 8 – Optional: Deploy as API

    Install FastAPI:

    pip install fastapi uvicorn

    Wrap YOLOv5 inference in an API:

    from fastapi import FastAPI, UploadFile
    import torch
     
    app = FastAPI()
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.engine')
     
    @app.post("/predict")
    async def predict(file: UploadFile):
        img = await file.read()
        results = model(img)
        return results.pandas().xyxy[0].to_dict(orient="records")

    Run API:

    uvicorn app:app --host 0.0.0.0 --port 8000

Step 9 – Cleanup

    Deactivate environment:

    deactivate

    Free GPU memory by stopping scripts: Ctrl+C

✅ Learning Outcomes

    Setup Jetson Nano with YOLOv5 for object detection

    Convert models to ONNX & TensorRT for optimized edge inference

    Measure performance gains

    (Optional) Serve detections via a lightweight FastAPI endpoint
```

## Section 23: Week 22: Optimizing AI for Edge Devices

### 150. 148. Quantization for Edge Efficiency
- Models developed in datacenter environment might be too large for Edge device
- Why quantization
  - Latency: INT8 operations is much faster ahn FP32 on many edge processors
  - Memory: weights and activations shrink dramatically from FP32->INT8 (4x)
  - Power: fewer bits == less energy consumed per operation
  - Deployability: fit models on microcontroller and SoC memory budgets
- Key quantization concepts
  - Bit-width reduction: FP32-> FP16/BF16 -> INT8/INT4
  - Quantization parameters: maps real values to integers using scal and zero-point
  - Granularity choices: per-tensor vs per-channel quantization
- Quantization approaches
  - PTQ (Post-Training Quantization)
    - No retraining
    - Requires calibration data
    - Best for well-behaved networks
  - QAT (Quantization-Aware Training)
    - Simulates quantization effects during training
    - Higher accuracy but more engineering effort and training
- HW landscape for quantized models
  - ARM CPUs (NEON)
  - NPUs/Edge TPUs
  - Nvidia Jetson
  - Apple/Android: CoreML/NNAPI backends prefer low-precision
- PTQ quick start (PyTorch -> ONNX runtime)
```py
# Collect a small calibration set (10031000 samples)
calib = [prepare_input(x) for x in calib_loader]
# Export to ONNX
torch.onnx.export(model, dummy, "model_fp32.onnx", opset_version=17)
# ONNX Runtime static PTQ (simplified sketch)
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
class MyCalib(CalibrationDataReader):
    def get_next(self):
        # yield dicts: {input_name: np_array}
        for x in calib:
            yield { "input": x }
quantize_static(
    "model_fp32.onnx",
    "model_int8.onnx",
    MyCalib(),
    weight_type=QuantType.QInt8,
    optimize_model=True)
```
- QAT quick start (PyTorch)
```py
import torch, torch.ao.quantization as tq
model.train()
model.fuse_model() # if supported (Conv+BN+ReLU)
qconfig = tq.get_default_qat_qconfig("fbgemm")
model.qconfig = qconfig
tq.prepare_qat(model, inplace=True)
for step, (x, y) in enumerate(train_loader):
    loss = train_step(model, x, y) # standard training
    if step % 1000 == 0:
        validate(model)
model.eval()
tq.convert(model, inplace=True) # produces INT8 modules
```
- TensorRT INT8 (Jetson)
  - Export ONNX -> build INT8 engine
  - Sensitive layers can remain FP16
  - Expect 1.5-3x speedups vs FP16 on supported operations
  - Monitor accuracy drop; adjust per-channel and calibration strategy as needed
- Accuracy guardrails
  - Representative calibration
  - Per-channel quantization
  - Skip sensitive operations
  - Special transformer techniques
- Validation checklist
  - Accuracy metrics: top-1/F1/WE delta vs FP32
  - Performance metrics: p50/p95 latency, throughput
  - Resource usage
  - Production testing: A/B test in production & automated rollback
- Common pitfalls & fixes
  - Big accuracy drop: use QAT instead of PTQ
  - Edge device OOM: quantize activations too, reduce batch size, use INT8 IO types
  - Layer mismatch after export: align ONNX opset versions & fusion patterns, update runtime libraries
  - Jittery latency: pin CPU threads, use static input shapes, implement proper warmup

### 151. 149. Pruning and Model Compression Basics
- Why compression?
  - Compression reduces
    - Model size
    - Compute cost
    - Power consumption
  - Goal: faster, leaner models without major accuracy loss
- What is pruning?
  - Removing unnecessary weights, neurons or filters
  - Based on
    - Magnitude: drop small weights near zero
    - Structured: drop entire channels/layers
    - Unstructured: random connections removed
  - Recovery: pruned models retrain -> recover performance
- Types of prunning
  - Unstructured: fine-grained (individual weights)
  - Structured: remove neurons, filters, heads, layers
  - Dynamic: drop comkputations on the fly; context-dependent calculations
- Pruning workflow
  - Train baseline model
  - Apply pruning
  - Fine-tune retrain
  - Export compressed model
  - Deploy with sparsity-aware runtime
- Ex: unstructured pruning (PyTorch)
```py
import torch
import torch.nn.utils.prune as prune
# Load pre-trained model
model = torch.hub.load('pytorch/vision','resnet18',pretrained=True)
# Prune 30% of connections in first conv layer
prune.l1_unstructured(model.conv1,name="weight",amount=0.3)
# Fine-tune model on training data after pruning
```
- Ex: structured pruning (Filters)
```py
# Remove entire filters from Conv2d layer
prune.ln_structured(
  model.layer1[0].conv1,
  name="weight",
  amount=0.2, # Remove 20% of filters
  n=2, # L2 norm
  dim=0 # Filter dimension
)
```
- Compression beyond pruning
  - Quantization: reduce precision(FP32-INT8)
  - Knowledge distillation: train small student model from large teacher model by mimicking output distribution rather than hard labels
  - Weight sharing: Reuse weights through Huffman coding or clustering similar parameters
  - Low-rank factorization: approximate weight matrices as products of smaller matrices to reduce parameter count and computation
- HW acceleration of sparse models
  - HW support challenges
    - GPUs/CPUs often ignore random sparsity -> no speedup
    - Memory access patterns matter more than FLOP count
    - Structured pruning with HW -> actual latency reduction
  - Acceleration libraries
    - Nvidia TensorRT
    - cuSparse
    - ONNX Runtime
    - TensorFlow lite
    - ARM Compute library
- **Sparsity-aware hardware (HW)** in AI refers to specialized computing architectures (like AI accelerators, GPUs, or TPUs and Spiking Neural Networks (SNNs)) designed to detect and skip computations involving zero-valued weights or activations    
- Trade-offs
  - Accuracy vs compression: aggressive pruning == higher accuracy loss
  - Implementation complexity: sparse models harder to optimize without runtime support
  - Structure vs flexibility: structured pruning = real speedups but less flexible
- Real world examples
  - YOLOv5: 40% pruned model runs 2x faster with only 2.5% mAP drop on Jetson Nano
  - Transformer: heads/layers pruned by 30% fits in 2GB RAM on-device NLP
  - ResNet Vision: filter pruning + quantization -> 4x smaller with < 1% accuracy drop
- Best practices
  - Start conservative
  - Always fine-tune
  - Combine methods: pruning + quantization
  - Favor structure
  - Measure what matters: latency, not parameter count

### 152. 150. TensorRT for Edge Inference
- TensorRT
  - Nvidia's inference optimizer & runtime that converts trained models into optimized execution engines for deployment on edge devices
  - Supported Targets
    - Jetson family
    - Nvidia GPUs
    - Data center AI accelerators
  - Framework support: PyTorch, TensorFlow, ONNX
- Why TensorRT for Edge AI?
  - Reduced latency
  - Smaller memory footprin
  - Increased throughput
  - Extended battery life
- Optimization techniques in TensorRT
  - Layer fusion
  - Precision calibration: FP32 -> FP16 -> INT8
  - Kernel auto-tuning
  - Dynamic memory allocation
- TensorRT workflow overview
  - Train model
  - Export to ONNX
  - Build TensorRT engine
  - Run inference
  - Benchmark & validate
- INT8 quantizawtion may result in 1-2% accuracy loss compared to FP32
- Use cases at the edge
  - Computer vision
  - Robotics
  - Healthcare
  - Retail/smart cities
- TensorRT best practices
  - Representative calibration data
  - Profile before production
  - Maintain multiple engines: keep both FP16 and INT8 engines for fallback
  - Scale with K8 + Triton: for multi-device deployments
  - Validate accuracy vs speed

### 153. 151. Running Vision Models on Raspberry Pi
- Why Rasberry Pi for AI?
  - Low-cost, accessible: $35-75
  - Ideal for prototyping & education
  - Camera interface built-in
  - Broad community support
- HW requirements
  - Rasberry Pi 4: 4/8GB RAM, ARM Cortex-A72 CPU
  - MicroSD card: > 32GB, class 10
  - Camera: Pi Camera module v2 (8MP) or USB webcam
  - Optional: Coral TPU (USB accelerator for 10-50x faster inference)
- SW stack
  - OS: Rasberry Pi OS (64bit recommended), Ubuntu 20.04 LTS
  - Frameworks: TensorFlow Lite, PyTorch Lite, OpenCV
  - Optimization: ONNX runtime ARM builds, Coral Edge TPU runtime, ARM NEON acceleration
  - Languages: Python3, C++, shell script
- Ex: TensorFlow Lite   
```py
import tflite_runtime.interpreter as tflite
import numpy as np
interpreter = tflite.Interpreter(
  model_path="mobilenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img = np.random.rand(1,224,224,3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(
output_details[0]['index'])
```
- ONNX runtime on Rasberry Pi
  - Benefits
    - Run models from any framework
    - ARM-optimized builds for improved performance
    - Memory-efficient inference on Pi's limited RAM
    - Supports popular vision models (ResNet, MobileNet)
- Acceleration with Coral EdgeTPU
  - 10-50x speed up with USB accelerator
- Challenges
  - Limited resources
  - Thermal constraints
  - Power limitations
  - Processing bottlenecks
- Best practices
  - Use lightweight architectures
  - Apply quantization
  - Optimize IO pipeline
  - Consider HW accelerations

### 154. 152. TinyML and Microcontrollers for AI
- AI on devices with KB memory
- Sensors, wearables, IoT
- Tasks: keyword spotting, anomaly detection, gesture recognition
- Why TinyML
  - Ultra-low power
  - Privacy-first
  - Real-time inference
  - Massive scale
- HW examples
  - Arduino Nano 33 BLE Sense
  - STM32 MCUs
  - ESP32
  - Edge TPU/NPU
- TinyML SW stack
  - TensorFlow Lite for microcontrollers
  - Edge Impulse
  - CMSIS-NN
  - ONNX Runtime Mobile
- Ex: keyword spotting
  - Collect audio samples
  - Train CNN on spectograms
  - Convert to TFLM model
  - Deploy on Arduino
  - Test device response
- Ex: anomaly detection in IoT
  - Vibration sensor
  - MCU runs autoencoder model locally
  - Algorithm learns "normal" operation patterns
  - Alerts sent only when anomalies detected
- Model optimizatino techniques
  - Quantization
  - Pruning & distillation
  - Feature engineering
  - Micro-architectures
- TinyML challenges
  - Memory constraints: 32-512KB RAM
  - Power management
  - Bare metal programming: no OS
  - Limited debugging
- Best practices
  - Design for extreme constraints
  - Use specialized tools
  - Benchmark on real HW
  - Optimize power consumption
  - Focus on specific tasks

### 155. 153. Benchmarking Models on Edge Hardware
- Why benchmark?
  - Resource constraints
  - Performance variability
  - Deployment confidence
- Key metrics to measure
  - Performance: latency (p50/p95), throughput (inferences/sec, FPS for vision)
  - Quality: accuracy (top-1/F1/precision-recall), memory footprint (RAM usage, model size)
  - Efficiency: Power draw (Watt consumption, battery drain)
- Benchmarking workflow
  - Select candidate models
  - Convert to edge-optimized format
  - Deploy to target HW
  - Run standardized benchmark dataset
  - Collect comprehensive metrics
  - Compare against baseline
- Tools & frameworks
  - TensorFlow Lite
  - ONNX runtime
  - MLPerf Tiny/Inference
  - PyTorch Mobile
  - Custom profiling: time.perf_counter, torch.cuda.synchronize()  
- Ex: TFLite benchmark on Rasberry Pi
```bash
./benchmark_model \
--graph=mobilenet_v2.tflite \
--input_layer="input" \
--input_layer_shape="1,224,224,3" \
--num_threads=4
```
  - Output metrics:
    - Inference latency (ms)
    - Memory usage (MB)
    - CPU utilization (%)
    - Initialization time
    - Per-layer performance breakdowns
- Edge device categories
  - Microncontrollers: tinyML, MicroNets, ProxylessNAS
  - Rasberyy Pi-class: MobileNetV2, EfficientDetLite
  - Jetson/Edge GPUs: YOLOv5, ResNet18, DistilBERT
- Power & Thermal Testing
  - Measurement tools
    - USB inline power meters
    - Jetson tegrastats utility
    - Thermal imaging cameras
    - System power monitoring APIs
  - Key metrics
    - Watts under various workloads
    - Thermal throttling thresholds
    - Accuracy-per-Watt efficiency
    - Battery life estimation
- Best practices
  - Use real HW
  - Representative datasets
  - Measure complete latency
  - Batch size variations
  - Automate testing
  
### 156. 154. Lab – Optimize a Model with TensorRT
- Goal: Take a pretrained vision model (ResNet18) → convert it to TensorRT engines (FP16 & INT8) → benchmark performance improvements on Jetson / NVIDIA GPU.
```
Step 0 – Prerequisites

    NVIDIA Jetson (Nano/Xavier/Orin) or any NVIDIA GPU system with CUDA + TensorRT installed.

    torch, onnx, onnxruntime, torchvision.

    TensorRT CLI tool trtexec (bundled with TensorRT).

Step 1 – Export PyTorch Model to ONNX

    import torch
    import torchvision.models as models
     
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True).eval()
     
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, "resnet18.onnx",
                      input_names=["input"],
                      output_names=["output"],
                      opset_version=17)
    print("ONNX model saved: resnet18.onnx")

✅ Output: resnet18.onnx
Step 2 – Baseline ONNX Runtime Inference

    import onnxruntime as ort
    import numpy as np, time
     
    session = ort.InferenceSession("resnet18.onnx")
    input_name = session.get_inputs()[0].name
     
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
     
    start = time.time()
    out = session.run(None, {input_name: x})
    print("Latency (ms):", (time.time()-start)*1000)

✅ Record baseline latency (~50–100ms on Jetson Nano, lower on Xavier/desktop GPU).
Step 3 – Build TensorRT FP16 Engine

    trtexec --onnx=resnet18.onnx \
            --saveEngine=resnet18_fp16.engine \
            --fp16

    --fp16 enables half-precision ops.

    Produces resnet18_fp16.engine.

Step 4 – Run FP16 Engine

    trtexec --loadEngine=resnet18_fp16.engine --shapes=input:1x3x224x224

✅ Compare average latency vs ONNX Runtime FP32.
Expect ~2–3× speedup.
Step 5 – INT8 Quantization (Calibration)

    Prepare small calibration dataset (50–100 sample images).

    Build INT8 engine with calibration:

    trtexec --onnx=resnet18.onnx \
            --saveEngine=resnet18_int8.engine \
            --int8 \
            --calib=calib.cache

    Calibration ensures accuracy retention in INT8 mode.

Step 6 – Run INT8 Engine

    trtexec --loadEngine=resnet18_int8.engine --shapes=input:1x3x224x224

✅ Expect ~3–5× speedup vs FP32, with slight accuracy drop (<1–2%).
Step 7 – Compare Results

Mode Latency (ms) Speedup Accuracy Δ FP32 ~50 ms 1.0× baseline FP16 ~20 ms 2.5× ~0% drop INT8 ~12 ms 4× <2% drop
Step 8 – Integrate into Python (Torch-TensorRT)

    import torch_tensorrt
     
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((1,3,224,224))],
        enabled_precisions={torch.float16}  # or int8
    )
    torch.jit.save(trt_model, "resnet18_trt.ts")

✅ Easier integration into PyTorch workflows.
Step 9 – Cleanup

    rm resnet18.onnx resnet18_fp16.engine resnet18_int8.engine

📂 Folder Structure

    lab154_tensorrt_opt/
     ├── export_resnet18.py
     ├── benchmark_onnx.py
     ├── README.md
     └── (generated) resnet18.onnx / .engine files

✅ Learning Outcomes

    Export model → ONNX → TensorRT engine.

    Optimize inference with FP16 & INT8.

    Benchmark latency and accuracy trade-offs.

    Deploy optimized models for real-time edge inference.
```

## Section 24: Week 23: Mobile AI Infrastructure

### 157. 155. Why Mobile AI Is Booming
- AI landscape is evolving from cloud-only to hybrid + on-device inference models
- Core drivers of mobile AI adoption
  - Latency: sub-100ms response
  - Privacy
  - Cost efficiency
  - Reliability
  - Personalization
- HW tailwinds
  - NPUs/neural engines
  - Efficient GPU/DSP pipelines via metal/vulkan APIs
  - Mixed precision support (FP16/INT8)
- SW ecosystem
  - Apple Core ML
  - TensorFlow Lite
  - Android NNAPI
  - ONNX Runtime Mobile
- Business impact
  - Improved metrics
  - Cost reduction
  - Compliance: data residency and user consent requirements
  - Competitive Edge  
- Constraints & trade-offs
  - Memory & compute
  - Thermals & battery
  - Model lifecycle
  - HW fragmentation
- Making models mobile-ready
  - Quantization
  - Pruning & distillation
  - Operator coverage
  - Streaming & chunking
  - On-device caching: store tokens/embeddings for faster inferences
- Architectures that work well on mobile
  - Vision
  - Speech
  - NLP
  - LLMs
- Hybrid AI patterns
  - On-device first
  - Progressive enhancement
  - Privacy gates
- Success checklist for mobile AI
  - Define clear SLAs
  - Ensure framework compatibility
  - Optimize aggressively
  - Test across device matrix
  - Implement telemetry: collect performance parameters

### 158. 156. Core ML for iOS – Basics
- Apple's framework for on-device ML
- Core ML
  - Native ML framework for Apple's entire ecosystem
  - Comprehensive support for vision, NLP, sound analysis, recommendation systems, and tabular ML
  - Performance optimized for Apple Neural Engine (ANE), GPU, and CPU
  - Models deployed as `.mlmodel` files
- Why Core ML?
  - Performance
  - Privacy
  - Battery efficiency
  - Ecosystem integration: Vision, CreateML, ARKit, and Siri
- Core ML workflow
  - Train model: PyTorch, TensorFlow, or Scikit-learn
  - Convert model: .mlmodel format using coremltools or ONNX conversion pipeline
  - Integrate
  - Implement API
  - Run inference
- Ex: Converting PyTorch model
```py
import torch
import torchvision.models as models
import coremltools as ct
# Load pretrained model
model = models.mobilenet_v2(pretrained=True).eval()
# Dummy input
example = torch.rand(1, 3, 224, 224)
# Convert to Core ML
traced = torch.jit.trace(model, example)
mlmodel = ct.convert(
  traced,
  inputs=[ct.ImageType(name="input",
  shape=example.shape)]
)
mlmodel.save("MobileNet.mlmodel")
```
- Core ML model types
  - Image models
  - Text models
  - Audio models
  - Tabular models: recommendation engines, regression models, and decision trees for structured data
- Deployment benefits
  - Offline-ready operation
  - Support for over-the-air model updates
  - Seamless ecosystem integration with apple framework
- Limitations and challenges
  - Model size constraints; keep models under 100MB for App store distribution
  - Operator support gaps: not all PyTorch/TF operations are compatible in CoreML
  - Optimization requirements: quantization (INT8) and pruning for acceptable performance
  - Debugging complexity

### 159. 157. TensorFlow Lite for Android – Basics
- TensorFlow Lite
  - Mobile & embedded ML framework
  - HW optimized: tuned for ARM CPUs, GPUs, NPUs, and DSPs
  - On-device processing: runs offline, private, and fast on android devices
- Why TFLite for android?
  - Performance
  - Battery life
  - Device compatibility
  - Developer integration
- TFLite development workflow
  - Train/export model
  - Convert to TFLite
  - Add to android project
  - Implement interpreter API
  - Optimize with delegates
- Converting a model to TFLite
```py
import tensorflow as tf
# Load model (Keras example)
model = tf.keras.applications.MobileNetV2( weights="imagenet")
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # optional quantization
tflite_model = converter.convert()
# Save file
open("mobilenet_v2.tflite", "wb").write(tflite_model)
```
- Key conversion options  
  - Quantization (INT8, FP16)
  - Pruning for size reduction
  - Op compatibility selection
  - Input/output shape specification
- Accelerators (Delegates)
  - NNAPI delegate
  - GPU delegate
  - Hexagon DSP delegate
  - Edge TPU delegate: optimized for Google Coral and other custom HW
- Common challenges
  - Model size constraints: App store/play store limits (~100MB)
  - Operator compatibility
  - Device fragmentation
  - Performance tradeoffs

### 160. 158. Deploying On-Device NLP Models
- Why on-device NLP?
  - Privacy
  - Latency
  - Cost
  - Personalization
  - Reliability
- Typical on-device NLP tasks
  - Text classification: spam detection, sentiment analysis
  - Sequence labeling
  - Generation: auto-complete, predictive text input, suggested replies
  - Speech-to-text/translation
  - Conversational AI
- Model architectures that work
  - Compressed transformers
    - DistillBERT
    - TinyBERT
    - MobileBERT
  - Parameter sharing
    - ALBERT
  - Lightweight models
    - FastText
    - Bag-of-Embeddings
  - Ultra-low-resource
    - RNN-Lite/GRU variants: for MCUs
- Workflow overview
  - Train/fine-tune
  - Optimize
  - Export
  - Integrate
  - Run inference
- Ex: converting DistilBERT -> TFLite  
```py
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
# Load model
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # quantization
tflite_model = converter.convert()
open("distilbert.tflite", "wb").write(tflite_model)
```
  - Key points
    - Default quantization
    - Size reduction: 240MB -> 60MB
    - Direct Keras model conversion
- Performance optimizations
  - Quantization
  - Pruning
  - Operator fusion
  - Distillation    
- Best practices
  - Start with mobile-optimized models
  - Apply QAT
  - Test across device tiers
  - Build hybrid pipelines

### 161. 159. Power Management in Mobile AI
- Why power matters in mobile AI
  - User retentioanl challenge
  - Computational intensity
  - Thermal constraints
- Sources of power drain
  - Compute operations
  - Sensor usage
  - Inefficient code
- HW efficiency levers
  - NPUs deliver up to 100x enery efficiency vs CPU for AI workloads
  - GPU delegates for parallel operations can achieve 4-8x better energy efficiency than CPU
  - DSPs enable 20x more efficient audio/sensor inference
- SW optimization techniques
  - Model optimization
    - Quantization
    - Pruning/distillation
  - Runtime optimization
    - Batching & caching
    - Lazy loading
- Duty cycling for sensors
  - Motion triggers
  - Keyword spotting
  - Preprocessing offload
  - Camera activation
- Developer tools for power profiling
  - Anroid tools
    - Android studio profiler
    - Battery historian
    - ADB shell dumpsys batterystats
  - iOS tools
    - Xcode instruments: Energy Log
    - Energy Gauge
    - MetricKit
  - Embedded Tools
    - Jetson tegrastats
    - External power monitors
- Real-world best practices
  - Use smaller models
  - Quantize aggressively
  - Profile per-feature
  - User controls: implement low-power mode toggle in app settings
  - A/B testing
  - Thermal testing

### 162. 160. Privacy Considerations for On-Device AI
- Why privacy matters
  - Sensitive data
  - Trust & adoption
  - Regulatory landscape: GDPR, CCPA, and HIPAA
  - Data minimization
- Privacy advantages of on-device AI
  - Local inference
  - Reduced attack surface
  - Offline-first operation
  - Private personalization
- Core privacy risks
  - Model leakage
  - Side-channel leaks
  - Unintended storage
  - Over-collection
- Privacy-by-design principles
  - Data minimization
  - On-device preprocessing
  - Transparency
  - User control
  - Secure defaults
- Techniques for enhanced privacy
  - Layer: federated learning - train across devices without sharing raw data
  - Outer: secuire enclaves and storage - HW isolation and encrypted model files
  - Core: differential privacy - adds noise to data to protect individuals
- Regulatory considerations
  - GDPR(EU)  
    - Right to explanation of algorithmic decisions
    - Explicit consent for data procesing
    - Right to erasure
  - CCPA (California)
    - Opt-out of data sales/sharing
    - Transparent disclosure of data usage
    - Access to collected personal information
  - HIPAA (US Healthcare)
    - Strict protocols for health data handling
    - Breach notification requirements
    - Limited disclosure permissions
- Ex: on-device NLP 
  - Keyboard AI for text prediction: a privacy-sensitive application of on-device AI
    - Local processing: no raw text transmission
    - Federatd updates: only anonymized gradients are shared to improve the central model, not user text
    - Differential privacy: statistical noise masks individual patterns while preserving group insights
  - Face unlock on smartphones
    - Secure storage
    - Local-only processing
    - Continuous improvement
- Developer best practices
  - Encryption
  - Minimal logging
  - Process isolation
  - Clear consent
  - Regular audits
- Common pitfalls 
  - Unncessary cloud reliance
  - Excessive data retention
  - Third-Party exposure
  - Inference opacity

### 163. 161. Lab – Build a Mobile AI App with TFLite
- Goal: Create a simple Android app that uses TensorFlow Lite to run an image classification model (MobileNet) fully on-device.
```
Step 0 – Prerequisites

    Android Studio installed (latest version).

    Android device or emulator (with camera access).

    mobilenet_v2.tflite model (from Day 157 export).

    Basic knowledge of Android app structure.

Step 1 – Create Android Studio Project

    Open Android Studio → New Project → “Empty Activity”.

    Set package name: com.example.tfliteclassifier.

    Language: Kotlin.

    Minimum SDK: API 23+.

Step 2 – Add TensorFlow Lite Dependencies

In app/build.gradle:

    dependencies {
        implementation 'org.tensorflow:tensorflow-lite:2.12.0'
        implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
        implementation 'org.tensorflow:tensorflow-lite-gpu:2.12.0'
    }

Sync project.
Step 3 – Add Model to Project

    Place mobilenet_v2.tflite in:

        app/src/main/assets/

    Also add labels.txt with ImageNet class labels.

Step 4 – Load TFLite Model in Kotlin

Classifier.kt:

    import org.tensorflow.lite.Interpreter
    import android.content.res.AssetFileDescriptor
    import java.nio.MappedByteBuffer
    import java.nio.channels.FileChannel
     
    class Classifier(private val assetManager: android.content.res.AssetManager) {
        private var interpreter: Interpreter
     
        init {
            interpreter = Interpreter(loadModel("mobilenet_v2.tflite"))
        }
     
        private fun loadModel(model: String): MappedByteBuffer {
            val fileDescriptor: AssetFileDescriptor = assetManager.openFd(model)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val channel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
     
        fun runInference(input: Array<Array<Array<FloatArray>>>): FloatArray {
            val output = Array(1) { FloatArray(1001) }
            interpreter.run(input, output)
            return output[0]
        }
    }

Step 5 – Connect Camera Input

    Use Android CameraX API to capture frames.

    Preprocess image → resize to 224×224, float32 normalized [0,1].

    Pass tensor into Classifier.runInference().

Step 6 – Display Prediction

In MainActivity.kt:

    val probs = classifier.runInference(preprocessedImage)
    val topIdx = probs.indices.maxByOrNull { probs[it] } ?: -1
    val label = labels[topIdx]
    textView.text = "Prediction: $label"

Step 7 – Run on Device

    Build & install app.

    Point camera at objects.

    App shows real-time classification results.

Step 8 – Optimize (Optional)

    Enable GPU delegate:

    val options = Interpreter.Options().addDelegate(GpuDelegate())
    val interpreter = Interpreter(model, options)

    Expect 2–3× faster inference vs CPU.

✅ Learning Outcomes

    Integrated TensorFlow Lite into an Android app.

    Performed real-time image classification locally.

    Explored GPU acceleration for faster results.

    Learned how to package AI models for mobile deployment.
```

## Section 25: Week 24: Data Pipelines for AI at Scale

### 164. 162. Introduction to ETL and ELT Pipelines
- Building scalable data workflows for AI infrastructure
- Why data pipelines matter
  - Foundation for AI
  - Automation
  - Scalability
- ETL
  - Extract: pull data from multiple sources
  - Transform: clean, normalize, and structure the data
  - Load: store processed data in target system
- ELT
  - Extract
  - Load: store raw data in a data lake or warehouse
  - Transform: leverage in-warehouse comkpute for prcessing after loading, transform data on-demand for specific use cases
- ETL vs ELT
  - ETL approach
    - Process data before loading
    - Optimized for smaller datasets
    - Well-suited for legacy, on-premises systems
  - ELT approach
    - Process dat after loading
    - Designed for large, raw datasets
    - Ideal for modern cloud and AI workflows
- AI use cases of pipelines
  - Data preprocessing
  - Real-time analytics
  - Feature engineering
  - Knowledge bases
- Common tools in ETL/ELT
  - ETL tools
    - Talend
    - Informatica PowerCenter
    - Apache NiFi
    - MS SSIS
  - ELT Tools
    - dbt (data build tool)
    - Snowflake
    - Google BigQuery
    - Databricks
  - Orchestration
    - Apache Airflow
    - Prefect
    - Dagster
    - Luigi
  - Streaming
    - Apache Kafka
    - Apache Flink
    - Apache Spark Streaming
    - AWS Kinesis
- Best practices
  - Error handling
  - Data quality
  - Modularity
  - Alignment

### 165. 163. Apache Airflow for AI Workflows
- Why workflow orchestration matters
  - Manual process break down
  - Dependency management fails
  - Reliability suffers
- Apache Airflow
  - Authors, schedules, and monitors workflows as code
  - Directed Acyclic Graphs (DAGs) to define workflow relationships, dependencies, and execution order
  - Configuration as Python code
  - Built-in scheduler for both time and event-based triggers
  - Rich monitoring and visualization UI
- Key features of Airflow
  - Advanced scheduling
  - Robust error handling
  - Rich ecosystem: BigQuery, Snowflake, Databricks, K8, ...
  - Flexible scaling
- AI/ML use cases
  - Data pipeline orchestration
  - Training pipeline automation
  - LLM & RAG workflows
- Airflow architecture components
  - Webserver UI
  - Metadata database
  - Scheduler
  - Executors
  - Workers
- Best practices for ML workflows
  - Build Idempotent Tasks
  - Version control everything
  - Implement comprehensive monitoring
  - Use dynamic DAGs

### 166. 164. Streaming Data with Apache Kafka
- How to build responsive AI systems
- Why streaming data matters
  - Real-time requirements
  - Instantaneous decisions
  - Continuous learning
  - Essential for IoT monitoring, fraud detection, and recommendation engines
- Apache Kafka
  - Distributed streaming platform
  - Publish-subscribe: enables applications to publish and subscribe to streams of records
  - Durable storage
  - Real-time processing: immediate response
  - Horizontal scalability
- Kafka core concepts
  - Producer: applicatino that sends data to Kafka topics
  - Topic: named feed or category where records are published
  - Consumer: application that reads and processes data from topics
  - Broker: individual Kafka server instance storing data
  - Cluster: group of brokers  
- AI/ML use cases
  - Feature engineering: extract, transform and deliver real-time features for training and inference, enabling models to response to changing conditions
  - Knowledge streaming: feed fresh information to large language models to prevent staleness and hallucination in responses
- Integrations with AI infrastructure
  - Kafka + Spark/Flink: distributed processing frameworks that consume Kafka streams for complex transformation and analytics
  - Kakfa +  Airflow: orchestrate complex data workflows triggered by streaming events for ETL and model training
  - Kafka + Feast: real-time feature store that standardizes features for training and serving with low latency
  - Kafka + TF Serving: stream real-time data directly to model endpoints for continuous inference
- Best practices for Kafka in AI systems
  - Strategic partitioning
  - Fault tolerance
  - Performance monitoring
  - Security implementation

### 167. 165. Feature Stores for ML (Feast, Tecton)
- Why feature stores matter
  - Feature duplication
  - Training-serving skew
  - Slow iteration cycles
- What is a feature store?
  - Centralized repository
  - Offline/online bridge
  - Covernance & metadata
- Feast: open source feature management
  - Python SDK and CLI for simple developer workflows
  - Offline stores: BigQuery, Redshift, Snowflake, File
  - Online stores: Redis, DynamoDB, Datastore
  - Streaming: Kafka, Kinesis integration
  - Ideal for research teams, startups, and organization with strong engineering capabilities
- Tecton: enterprise-grade feature platform
  - Managed infrastucture: production-ready, fully maanged feature platform with SLAs and expert support
  - Enterprise integration: Spark, Snowflake, Databricks, and AWS/Azure/GCP
  - Governance & compliance: RBAC, audit logs, approval workflows
  - Performance at scale: optimized for high-throughput, low-latency
- Feature store best practices
  - Standardize feature definitions
  - Version and track lineage
  - Monitor quality and freshness
  - Implement access controls

### 168. 166. Real-Time Data Preprocessing at Scale
- Why real-time preprocessing?
  - Fresh
  - Clean
  - Formatted
  - Actionable
- Critical use cases
  - Fraud dtection
  - IoT sensor processing
  - Real-time recommendations
  - Autonomous systems
- Core preprocessing tasks
  - Cleaning: handle missing, duplicate, or corrupted data points in the stream
  - Transformation: convert data into ML-ready formats such as feature scaling/normalization, one-hot encoding, dimensionality reduction
  - Aggregation: rolling statistics over time windows, tumbling windows for batch processing
  - Enrichment: join stream data with external sources - reference data lookups, feature store integration, cross-stream correlations
- Tools & Frameworks
  - Apache Kafka + Kafka Streams: event sourcing, log aggregation
  - Apache Flink: complex event processing, time-series analysis
  - Spark Structured Streaming: ML feature engineering at scale
- AI/ML use cases
  - Robotics & reinforcement learning
  - Recommendation engines
  - Anomaly detection
  - Conversational AI
- Scaling strategies
  - Windowing techniques: implement appropriate time-based aggregations to manage computational load
  - Parallelization: scale horizontally
  - Optimization techniques: reduce resource utilization while maintaining throughput
- Best practices
  - Continuous data quality validation
  - Lightweight processing design
  - Schema evolution planning
  - Comprehensive monitoring

### 169. 167. Data Quality Monitoring for AI Systems
- Why data quality matters
  - Garbage in/garbage out
- Dimensions of data quality
  - Completeness
  - Accuracy
  - Consistency
  - Timeliness: fresh enough?
  - Validity
- Tools for monitoring
  - EvidentlyAI
  - Great Expectations
  - WhyLabs
  - Monte Carlo & Soda
- AI/ML use cases
  - Feature validation
  - Schema monitoring
  - Fairness checks
  - Anomaly detection
- Scaling strategies
  - Automate validation
  - Visual monitoring
  - Set Thresholds
  - CI/CD integration
- Best practices
  - Treat data as first-class infrastructure
  - Monitor both training & inference data
  - Version datasets & track schema changes
  - Collaborate with domain experts  

### 170. 168. Lab – Build a Streaming Pipeline with Kafka
- Learning goals
  -  Stand up a single-broker Kafka on your laptop with a visual UI
  - Publish synthetic transactions in real time
  - Build a stream processor that validates, aggregates, and emits features
  - Observe and troubleshoot with a UI + logs
  -  (Optional) Add a dead-letter queue and basic data quality checks
```
0) Prereqs (install once)

    Docker Desktop (running)

    Python 3.9+ (python --version)

    Terminal + code editor

1) Project scaffold

    mkdir kafka-streaming-lab && cd kafka-streaming-lab
    mkdir app

Repo layout

    kafka-streaming-lab/
      docker-compose.yml
      app/
        requirements.txt
        producer.py
        processor.py
        consumer.py

2) Docker Compose (Kafka + Kafka UI)

Create docker-compose.yml:

    version: "3.8"
    services:
      kafka:
        image: bitnami/kafka:3.7
        container_name: kafka
        ports:
          - "9092:9092"    # external for host apps
          - "29092:29092"  # internal for containers
        environment:
          - KAFKA_ENABLE_KRAFT=yes
          - KAFKA_CFG_NODE_ID=1
          - KAFKA_CFG_PROCESS_ROLES=broker,controller
          - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=1@kafka:9093
          - KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT,CONTROLLER:PLAINTEXT
          - KAFKA_CFG_LISTENERS=INTERNAL://:29092,EXTERNAL://:9092,CONTROLLER://:9093
          - KAFKA_CFG_ADVERTISED_LISTENERS=INTERNAL://kafka:29092,EXTERNAL://localhost:9092
          - KAFKA_CFG_INTER_BROKER_LISTENER_NAME=INTERNAL
          - KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE=true
        volumes:
          - kafka_data:/bitnami/kafka
     
      kafka-ui:
        image: provectuslabs/kafka-ui:latest
        container_name: kafka-ui
        ports:
          - "8080:8080"
        environment:
          - KAFKA_CLUSTERS_0_NAME=local
          - KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS=kafka:29092
        depends_on:
          - kafka
     
    volumes:
      kafka_data:

Up the stack

    docker compose up -d

    UI: http://localhost:8080 (you’ll see cluster “local”)

3) Python deps

Create app/requirements.txt:

    kafka-python==2.0.2
    faker==25.9.2
    pydantic==2.8.2
    python-dateutil==2.9.0.post0

Install:

    cd app
    python -m venv .venv
    source .venv/bin/activate # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

4) Create topics (optional—auto-create is on)

You can let Kafka auto-create, or do it explicitly:

Via UI → Topics → “Create”:

    transactions (partitions: 3)

    features (partitions: 3)

    dlq-transactions (optional, partitions: 1)

5) Producer – real-time synthetic events

Create app/producer.py:

    import json, os, random, time
    from datetime import datetime, timezone
    from faker import Faker
    from kafka import KafkaProducer
     
    BOOTSTRAP = os.getenv("BOOTSTRAP", "localhost:9092")
    TOPIC = os.getenv("TOPIC", "transactions")
    EPS = float(os.getenv("EVENTS_PER_SECOND", "5"))
     
    fake = Faker()
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
        linger_ms=10
    )
     
    EVENT_TYPES = ["view", "add_to_cart", "purchase"]
    DEVICES = ["web", "ios", "android"]
    COUNTRIES = ["US", "IN", "BR", "DE", "GB", "CA"]
     
    def make_event():
        user_id = random.randint(1, 500)
        etype = random.choices(EVENT_TYPES, weights=[0.7, 0.2, 0.1])[0]
        amount = round(random.uniform(5, 200), 2) if etype == "purchase" else 0.0
        return {
            "event_time": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "event_type": etype,
            "amount": amount,
            "device": random.choice(DEVICES),
            "country": random.choice(COUNTRIES)
        }, user_id
     
    def main():
        print(f"Producing to {TOPIC} @ {EPS} eps")
        interval = 1.0 / EPS
        while True:
            payload, key = make_event()
            producer.send(TOPIC, key=key, value=payload)
            time.sleep(interval)
     
    if __name__ == "__main__":
        main()

Run:

    python producer.py

Tip: Watch messages in Kafka UI → Topics → transactions → Messages.
6) Stream processor – validate → aggregate → emit features

Create app/processor.py:

    import json, os, time
    from collections import defaultdict
    from datetime import datetime, timezone
    from dateutil import parser as dtparser
    from pydantic import BaseModel, Field, ValidationError
    from kafka import KafkaConsumer, KafkaProducer
     
    BOOTSTRAP = os.getenv("BOOTSTRAP", "localhost:9092")
    SRC_TOPIC = os.getenv("SRC_TOPIC", "transactions")
    SINK_TOPIC = os.getenv("SINK_TOPIC", "features")
    DLQ_TOPIC = os.getenv("DLQ_TOPIC", "dlq-transactions")  # optional
    WINDOW_SEC = int(os.getenv("WINDOW_SEC", "60"))
     
    class Txn(BaseModel):
        event_time: str
        user_id: int = Field(ge=1)
        event_type: str
        amount: float
        device: str
        country: str
     
    def epoch_minute(ts_iso: str) -> int:
        ts = dtparser.isoparse(ts_iso)
        return int(ts.timestamp() // WINDOW_SEC) * WINDOW_SEC
     
    consumer = KafkaConsumer(
        SRC_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id="processor-1",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: int(k.decode("utf-8")) if k else None,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        max_poll_records=200
    )
     
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
     
    # state: (user_id, window_start) -> counters
    state = defaultdict(lambda: {"event_count":0, "purchase_count":0, "revenue":0.0})
     
    last_flush = time.time()
     
    def flush_ready(now_epoch: int):
        """Flush windows that ended before current window."""
        to_delete = []
        for (user_id, win_start), agg in state.items():
            if win_start + WINDOW_SEC <= now_epoch - 1:
                out = {
                    "user_id": user_id,
                    "window_start": win_start,
                    "window_end": win_start + WINDOW_SEC,
                    "event_count": agg["event_count"],
                    "purchase_count": agg["purchase_count"],
                    "revenue": round(agg["revenue"], 2),
                    "emitted_at": datetime.now(timezone.utc).isoformat()
                }
                producer.send(SINK_TOPIC, value=out)
                to_delete.append((user_id, win_start))
        for k in to_delete:
            del state[k]
     
    try:
        while True:
            records = consumer.poll(timeout_ms=1000)
            if not records: 
                flush_ready(int(time.time()))
                continue
     
            for tp, msgs in records.items():
                for msg in msgs:
                    try:
                        txn = Txn(**msg.value)
                        win = epoch_minute(txn.event_time)
                        key = (txn.user_id, win)
                        st = state[key]
                        st["event_count"] += 1
                        if txn.event_type == "purchase":
                            st["purchase_count"] += 1
                            st["revenue"] += float(txn.amount)
                    except ValidationError as e:
                        # optional DLQ
                        producer.send(DLQ_TOPIC, value={
                            "error": "validation_error",
                            "reason": e.errors(),
                            "payload": msg.value
                        })
     
            consumer.commit()
            now = time.time()
            if now - last_flush > 5:  # flush every ~5s
                flush_ready(int(now))
                last_flush = now
     
    except KeyboardInterrupt:
        flush_ready(int(time.time()))
        consumer.commit()

Run:

    python processor.py

7) Feature consumer – view results (and/or write CSV)

Create app/consumer.py:

    import csv, os, json
    from kafka import KafkaConsumer
     
    BOOTSTRAP = os.getenv("BOOTSTRAP", "localhost:9092")
    TOPIC = os.getenv("TOPIC", "features")
    WRITE_CSV = os.getenv("WRITE_CSV", "false").lower() == "true"
     
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="features-reader"
    )
     
    if WRITE_CSV:
        f = open("features.csv", "w", newline="")
        writer = None
     
    print("Consuming features...")
    for msg in consumer:
        rec = msg.value
        print(rec)
        if WRITE_CSV:
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=rec.keys())
                writer.writeheader()
            writer.writerow(rec)

Run (choose one):

    python consumer.py
    # or write to CSV as a sink
    WRITE_CSV=true python consumer.py

Watch Kafka UI → topics → features → Messages for aggregated outputs.
8) Sanity checks

    Throughput: In UI, check consumer groups → processor-1 lag ~ 0

    Windows roll: You should see features where window_start increments every minute per user

    DLQ (if enabled): Force a bad event by editing producer.py to sometimes emit negative amount. Confirm records land in dlq-transactions.

9) Experiments (short, impactful)

    Scale partitions: In UI, increase transactions partitions, run a 2nd processor (GROUP_ID=processor-1 keeps both in same group → load-share).

    Backpressure: Bump rate EVENTS_PER_SECOND=50 for the producer; watch consumer lag.

    Latency: Shrink WINDOW_SEC=30 and flush interval to emit faster windows.

    Schema evolution: Add a new field (e.g., marketing_channel) and keep processor tolerant (ignore unknown keys).

    Fault injection: Kill processor.py and restart—confirm it resumes from last committed offsets.

10) Cleanup

    # stop Python processes
    # then
    docker compose down -v
    deactivate  # exit venv

What you should see (expected)

    transactions: frequent JSON events (views, add_to_cart, purchase)

    features: minute-bucketed aggregates per user:

        {
          "user_id": 123,
          "window_start": 1724782380,
          "window_end": 1724782440,
          "event_count": 9,
          "purchase_count": 2,
          "revenue": 153.17,
          "emitted_at": "2025-08-27T01:23:45.678901+00:00"
        }

Stretch goals (pick any)

    Replace Python aggregation with Spark Structured Streaming or Flink

    Add a Postgres container + write features to SQL (via your consumer)

    Containerize producer, processor, consumer with Dockerfiles

    Add Prometheus + Grafana and export simple metrics (messages/sec, lag)

    Replace in-memory windows with Redis or RocksDB for durability
```

## Section 26: Week 25: Generative AI Infrastructure - Foundations 

### 171. 169. Infrastructure Challenges of Large Language Models
- From experiment to planet-scale production
- Why LLM infrastructure is hard
  - Massive model scale: billions-trillions of parameters
  - Data pipeline complexity: terabytes-petabytes datasets
  - Extended training windows
  - Conflicting inference demands
- Compute & parallelism constraints
  - GPU scarcity issues
  - Parallelism trade-offs
  - Interconnect bottlenecks
  - Distributed recovery
- Networking & IO bottlenecks
  - Critical network constraints
    - All-reduce operations for gradient synchronization saturate network links
    - Uneven sharding or heterogeneous nodes create performance stragglers
    - Vector databases experience hot-spotting during high-volume RAG operations
  - Mitigation strategies
    - Locality-aware data placements to minimize cross-rack traffic
    - Intelligent prefetching based on access patterns
    - Hierarchical synchronization to reduce global communication
- Cost & energy management
  - Cost optimization strategies
    - Spot/preemptible instances with checkpoint recovery
    - Mixed precision training (FP16/BF16) and inference (INT8/INT4)
    - Kernel fusions and operator optimization for throughput gains
    - Attention mechanisms sparsity to reduce computation
  - Sustainability considerations
    - Power consumption monitoring and capping
    - Cooling efficiency optimization
    - HW utilization targets to justify environmental impact
    - Carbon-aware scheduling for training workloads
- Reliability & observability gaps
  - Training reliability
  - Drift detection
  - End-to-end tracing
  - Safety monitoring
- Data, privacy & governance
  - Data quality at scale
  - Privacy workflows
  - Compliance & lineage
  - Safety evaluation
- Serving at scale
  - Performance balancing
  - Multi-tenant operations
  - Caching architecture
  - Deployment control
- Best-in-class LLM infrastructure
  - Unified platform
  - Composable optimization
  - Robust CI/CD pipeline
  - Cost-aware experimentation

### 172. 170. Memory and Storage Needs of LLMs
- Why memory & storage are critical
  - Billions-trillions of parameters
  - Training stores parameters + optimizer states + activations simultaneously
  - Inference requires fast weight loading and context caching
  - Checkpoint storage costs grow exponentially
- Memory demands in training
  - Parameter storage: 1 billion parameters consume 4GB in FP32, optimizer (Adam) adds 4x overhead (~16GB)
  - Activation memory: forward pass activations often exceed parameter size during backpropagation. Gradient checkpoint trades compute for memory by recomputing activations
  - Precision tradeoffs: mixed precision traininig (FP16/BF16) cuts memory footprint ~50%
  - Overall, model weights ~25%, optimizer states ~35%, activations ~40% of memory
- Storage demands in training
  - Dataset storage: Petabyte-scale datasets
  - Checkpoint volume: 1-10TB per training run
  - Tiered architecture: NVMEe for hot data + object store for checkpoints + cold storage for archives
- Inference time memory needs
  - Weight loading
  - KV-Cache Growth: In transformers, Key-Value Cache for attention grows linearly with sequence length
  - Optimization techniques: batch inference amortizes memory usage
- Storage for model lifecycle
  - Training
  - Fine-tuning
  - Archiving
  - Deployment
- Optimization strategies
  - Distributed training optimizations
    - Parameter sharding: distributes model across GPUs
    - Activation offloading: moves tensors to CPU/NVMe when not needed
    - Gradient accumulation: reduces optimizer state memory pressure
  - Storage optimizations  
    - Compression & deduplication
    - Differential checkpoints: store only parameter changes
    - Checkpoint pruning: removes intermediate saves systematically

### 173. 171. Vector Databases – FAISS, Pinecone, Weaviate
- Why vector databases?
  - Semantic search
  - Vector operations: high dimensional numeric vectors
  - Similarity metrics
- Key features of vector databases
  - ANN indexing: Approximate Nearest Neighbor algorithms make billion scale vector search practical
  - Massive scale: Billions of embeddings while maintaining sub-second query response times
  - Metadata filtering: combine vector similarity with traditional filters (date ranges, categories, keywords)
  - Framework integration: Langchain, Llamaindex, and other LLM framekwork
- FAISS
  - Facebook AI Similarity Search
  - Optimized for both CPU & GPU vector operations
  - Supports multiple index types
  - Excellent for research and prototyping
  - May not scale well
- Pinecone
  - Fully managed cloud service
  - Enterprise-grade features
  - Advanced search capabilities
- Weaviate 
  - Built-in ML modules
  - GraphQL + REST API
  - Hybrid deployment options: self-hosted or Weaviate cloud service for managed infrastructure
- AI/ML use cases
  - RAG chatbots: retrieve relevant documents
  - Multimodal search: CLIP embeddings enable searching images with text queries or finding similar images
  - Recommendation systems
  - Anomaly detection: embedding sequences in time series and finding outliers
- Best practices
  - Choose the right index
    - Flat index: 100% accuracy, slower searches
    - HNSW: high accuracy, fast, memory-intensive
    - IVF: scalable, moderate accuracy, best for huge datasets
  - Version your embeddings
  - Monitor quality metrics
  - Keep vectors normalized

### 174. 172. RAG (Retrieval-Augmented Generation) Pipelines
- Why RAG?
  - Limited context
  - Knowledge cutoff
  - Fresh information: RAG injects fresh, domain-specific info when needed
  - Reduced hallucinations
- Core RAG workflow
  - User query: convert user's query into an embedding vector representation
  - Vector DB search: find semantically relevant document
  - Context assembly: select and organize top-k results to fit within context window
  - LLM Generation
- Key components
  - Embedding model: OpenAI, BERT, CLIP
  - Vector database: FAISS, Pinecone, Weaviate
  - Retriever
  - LLM
- AI/ML use cases
  - Enterprise knowledge assistants
  - Proprietary document chatbots
  - Scientific research assistants
  - Multimodal RAG: combinations of text, images, audio, and other data types
- Infrastructure challenges
  - Scale
  - Relevance
  - Latency
  - Freshness
- Best practices
  - Chunking strategy: meaningful chunks of 500-1000 tokens
  - Hybrid retrieval: combine semantic (embedding-based) and keyword (BM25) search for better coverage
  - Metrics monitoring; track relevance scores, recall rates, and hallucination frequencies
  - Performance optimization: Cache frequent queries/results

### 175. 173. Caching Strategies for LLMs
- Why caching matters
  - LLM inference is expensive and slow
  - Caching provides:
    - Eliminating redundant compute operations
    - Improving latency and throughput
    - Reducing operational costs by 30-70% in high-volume applications
- Types of caching strategies for LLMs
  - Prompt/response cache: stores complete query/answer pairs for exact matching
  - Embedding cache: reuses vector representations for repeated documents/queries
  - KV-Cache: preserves attention key-value states b/w inference steps
  - Retrieval cache: stores RAG search results for frequent queries
- Prompt/response caching
  - For frequently repeated queries
  - Hash the entire query +  model parameters as cache key
  - Store full model responses as cache values
  - Implement fuzzy matching for near-duplicate detection
  - Set appropriate TTL based on content freshness needs
  - Ex: customer support bots handling FAQs
- KV-cache in transformers
  - Technical benefits
    - Reduces inference complexity from N^2 to N
    - Cuts per-token generation time by 30-50%
    - Essential for long-context inference (8k+ tokens)
    - Enables real-time chat applications at scale
  - Implementation challenges
    - Memory footprint grows linearly with context length
    - Requires GPU memory management strategy
    - Must balance b/w sequence batching and cache size
    - Pruning techniques needed for ultra-long contexts
- Infrastructure considerations
  - Cache store options
    - Reids
    - Memcached
    - In-GPU memory
    - Hybrid approaches: tiered caching
  - Eviction policies
    - LRU: Removes least recently used entries
    - LFU: prioritizes frequently accessed items
    - TTL-based: expires entries after set druation
    - Size-based: caps memory usage with watermarks
  - Consistency management
    - Model versioning in cache keys
    - Flush strategy on model updates
    - Canary deployments with cache warning
    - Monitoring for stale response detection
- AI/ML use cases
  - Customer service chatbots
  - Search & RAG applications
  - Multi-turn dialogue systems
  - Content moderation  
- Best practices
  - Implementation guidelines
    - Cache selectively
    - Monitor hit/miss ratios
    - Implement cache warming
    - Version cache keys
  - Optimization strategies
    - Semantic deduplication
    - Adaptive TTL
    - Partial result caching
    - Probabilistic caching

### 176. 174. Serving LLMs in Production
- Why serving is challenging
  - Size constraints
  - Performance demands
  - Traffic unpredicatability
  - Resource efficiency: multi-tenant, cost-aware infrastructure
- Core serving architecture
  - Frontend API
  - Inference Engine
  - Caching layer
  - Orchestration
  - Observability
- Model deployment options
  - Single-node serving: simple deployment, limited scale
    - FastAPI
    - Limited by a single GPU memory
    - Suitable for smaller models
  - Distributed serving: shareded models across GPUs/nodes
    - Tensor parallelism across multiple GPUs
    - Supports larger models (70B+)
    - Higher operational complexity
  - Serverless APIs: on-demand, cost-efficient, higher latency
    - Pay-per-token pricing model
    - Cold start penalties
    - No infrastructure management
- Performance optimization
  - KV-Cache reuse
  - Request batching
  - Quantization & pruning
  - Memory pinning
- Observability & monitoring
  - Performance metrics: p50, p95, p99, throughput, error rates
  - Resource utilization
  - Distributed tracing
  - Proactive alerting
- Security & governance
  - API protection
  - Data security
  - Compliance
  - Access control

### 177. 175. Lab – Deploy a Simple RAG Pipeline
- Learning Goals
  - Build a retrieval-augmented generation pipeline end-to-end
  - Index documents as embeddings with FAISS
  - Retrieve relevant chunks at query time
  - Pass them into an LLM via API
  - Serve the pipeline through FastAPI for real-time usage
```
0) Prerequisites

    Python 3.9+

    An LLM API key (OpenAI, Anthropic, etc.)

    Install dependencies:

    mkdir rag-lab && cd rag-lab
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install faiss-cpu openai fastapi uvicorn tiktoken pydantic

1) Prepare Sample Documents

Create docs/ folder and add a few .txt files. Example:

    ai_infra.txt

    etl_vs_elt.txt

    kafka_streaming.txt

Each file should have ~3–5 paragraphs.
2) Create Embedding + Indexing Script

File: build_index.py

    import os, faiss, pickle, glob
    from openai import OpenAI
    import tiktoken
     
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    EMBED_MODEL = "text-embedding-3-small"
     
    def embed(texts):
        res = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in res.data]
     
    def chunk_text(text, size=500, overlap=50):
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks = []
        for i in range(0, len(tokens), size - overlap):
            sub = enc.decode(tokens[i:i+size])
            chunks.append(sub)
        return chunks
     
    docs, metadatas = [], []
    for file in glob.glob("docs/*.txt"):
        with open(file) as f:
            text = f.read()
        chunks = chunk_text(text)
        docs.extend(chunks)
        metadatas.extend([{"source": file}] * len(chunks))
     
    embeds = embed(docs)
     
    dim = len(embeds[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeds).astype("float32"))
     
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump((index, docs, metadatas), f)
    print("Index built with", len(docs), "chunks")

Run:

    python build_index.py

3) Create RAG Query Function

File: rag.py

    import os, pickle, faiss, numpy as np
    from openai import OpenAI
     
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    EMBED_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"  # adjust if needed
     
    with open("faiss_index.pkl", "rb") as f:
        index, docs, metadatas = pickle.load(f)
     
    def embed(query):
        res = client.embeddings.create(model=EMBED_MODEL, input=[query])
        return np.array(res.data[0].embedding).astype("float32").reshape(1, -1)
     
    def retrieve(query, k=3):
        qvec = embed(query)
        D, I = index.search(qvec, k)
        results = [docs[i] for i in I[0]]
        return results
     
    def rag_answer(query):
        contexts = retrieve(query, k=3)
        prompt = f"Answer based on context:\n{contexts}\n\nQuestion: {query}\nAnswer:"
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

Test in REPL:

    from rag import rag_answer
    print(rag_answer("What’s the difference between ETL and ELT?"))

4) Deploy with FastAPI

File: server.py

    from fastapi import FastAPI
    from pydantic import BaseModel
    from rag import rag_answer
     
    app = FastAPI()
     
    class Query(BaseModel):
        question: str
     
    @app.post("/ask")
    def ask(q: Query):
        answer = rag_answer(q.question)
        return {"question": q.question, "answer": answer}

Run server:

    uvicorn server:app --reload --port 8000

5) Test the API

    curl -X POST http://127.0.0.1:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question":"Explain Kafka streaming for AI systems"}'

Response:

    {
      "question": "Explain Kafka streaming for AI systems",
      "answer": "Kafka enables real-time data ingestion..."
    }

6) Observability + Improvements

    Log queries & cache responses in SQLite/Redis

    Add a /health endpoint for monitoring

    Add a prompt/response cache to cut cost

    Deploy on cloud (Render, AWS ECS, GCP Cloud Run)

7) Stretch Goals

    Replace FAISS with Pinecone or Weaviate

    Add hybrid retrieval (keyword + vector search)

    Add auth (API keys) for multi-tenant serving

    Containerize with Dockerfile → deploy in Kubernetes

✅ Outcome: You now have a working RAG API serving answers from your own documents using embeddings + vector DB + LLM.
```

## Section 27: Week 26: Generative AI Infrastructure - Advanced

### 178. 176. DeepSpeed and ZeRO Optimization for LLMs
- Train bigger models faster & cheaper
- Why DeepSpeed?
  - Memory constraints
  - Communication bottlenecks
  - Storage explosion
- ZeRO: core idea
  - Zero Redundancy Optimizer
  - Instead of data parallelism, partitions:
    - Optimizer states
    - Gradients
    - Parameters
- ZeRO stages:
  - Stage 0: Baseline DDP - standard distributed data parallel with no sharding. Full model replica on each GPU
  - Stage 1: shared optimizer states - distributes Adam optimizer states across GPUs, saving ~4x memory for these tensors
  - Stage 2: Shard gradients - additionally partitions gradients, eliminating their redundancy during backpropagation
  - Stage 3: Shard parameters - full sharding. Distributes model parameters across GPUs, enabling training of massive models
- Offloading variants
  - ZeRO-offload
    - Moves optimizer states and gradients to CPU RAM
    - Reduces GPU memory pressure
    - PCIe transfer overhead
  - ZeRO-Infinity
    - Offloads model parameters and activations to CPU/NVMe
    - Virtually unlimited model size
    - Higher latency trade-off
- Performance techniques
  - Communication overlap
  - Kernel fusion
  - Gradient accumulation
  - Activation checkpointing
- DeepSpeed vs FSDP
  - DeepSpeed ZeRO-3
    - Mature offloading capabilities to CPU/NVMe
    - Advanced memory compression techniques
    - Flexible training schedulers and optimizers
    - Microsoft-supported, dedicated project
    - Custom communicatino backend
  - PyTorch FSDP
    - Native PyTorch API
    - Stornger ecosystem compatibility
    - Similar setup within PyTorch workflows
    - PyTorch's native communication
- Practical tuning tips
  - Progressive implementation
  - Measure Key Metrics
  - Tune communication parameters
  - Optimize offloading
- Failure modes & debugging
  - OOM erros despite ZeRO
    - Reduce micro-batch size
    - Increase ZeRO stage (1 -> 2 -> 3)
    - Enable offloading or activation checkpointing
    - Check memory leaks in custom code
  - Slow CPU/NVMe offloading
    - Verify PCIe bandwidth and utilization
    - Check disk IO performance and queue depth
    - Enable pinned memory for CPU transfers
    - Increase compute/communication overlap
  - Communication issues
    - Chjeck NCCL debug logs
    - Verify infiniband configuration
    - Monitor network bandwidth utilization
    - Test with smaller bucket sizes
  - Training instability
    - Review loss scaling approach
    - Try BF16 instead of FP16 if available
    - Check for NaN/Inf values in gradients
    - Implement gradient clipping
- When to use what
  - < 10B parameters
    - ZeRO-2/3 + BF16
    - Minimal offloading
    - Focus on throughput optimization
  - > 20B parameters
    - ZeRO-3 with aggressive offloading
    - Consider NVMe extension
    - Balance throughput vs capacity trade-offs
  - Long context models
    - Aggressive activation checkpointing
    - Implement KV-cache optimization techniques
    - Selective attention patterns
  - Multi-tenant clusters
    - Prioritize elasticity and job recovery
    - Optimize communication overlap
    - Implement checkpoint/resume strategies
  
### 179. 177. Megatron-LM for Large Model Training
- Why Megatron-LM?
  - Standard DDP + ZeRO can't scale alone for truly enormous models
  - Megatron-LM pioneered tensor + pipeline parallelism for extreme scale
- Core parallelism approaches
  - Data parallelism
  - Tensor parallelism: split individual weight matrics across GPUs
  - Pipeline parallelism: split model layers sequentially across different devices or nodes
- Tensor parallelism
  - Split large matrix multiplications across multiple GPUs
  - Requires high-speed interconnect
- Pipeline parallelism
  - Divide layers
  - Micro batching: schedule small batches to keep all GPUs busy
  - Bubble overhead: manage idle stages at pipeline start/end
- 3D parallelism in Megatron-LM
  - Maximum scale: DP + TP + PP for training models with 100B-1T parameters
  - Topology aware
  - Industry proven: GPT-3, MT-NLG, BLOOM
- Key features of Megatron-LM
  - Optimized CUDA kernels
  - Fused operations
  - Activation checkpointing
  - DeepSpeed integration: works with ZeRO optimizer and CPU/NVMe offloading
- Infrastructure requirements
  - HW
    - Multinode GPU clusters with infiniband
    - High GPU memory capacity
    - Fast storage for massive checkpoints
  - SW stack
    - Mixed precision (FP16/BF16)
    - NCCL 
    - CUDA 11.0+ with cuDNN
  - Orchestration
    - Slurm for HPC environment
    - K8 for cloud deployment
    - Ray for flexible research workflows
- Practical challenges
  - Communication bottleneck
  - Debugging complexity
  - Pipeline efficiency
  - Parameter tuning
- Best practices
  - Match parallelism to HW
  - Balance pipeline stages
  - Optimize memory usage
  - Focus on right metrics

### 180. 178. Flash Attention and Memory Optimizations
- Why optimize attention?
  - Standard attention mechanisms face significant scaling challenges
  - N^2 compute + memory complexity
  - Long sequences create a quadratic explosion in resource requirements
  - Attention activations dominate GPU memory usage during training
- Flash attention
  - Optimized CUDA kernel (kernel fusion)
  - Fused operations: avoids large attention matrices by combining operations
  - Chunk-based computation: smaller memory blocks (tiling)
    - 
  - Mathematically exact
- Benefits of Flash Attention
  - 2-4x speed up
  - 50-70% memory reduction
  - 8k-32k token context
  - Available in:
    - PyTorch core
    - Hugging Face Transformers
    - Nvidia Megatron-LM
    - Google JAX/Flax
    - Meta's Fairseq
- Memory optimization techniques
  - Activation checkpointing
  - Gradient accumulation
  - Mixed precision
  - Parameter sharding
- Inference time optimizations
  - KV-cache
  - Quantization
  - Paged attention (vLLM)  
  - GPU memory pinning
- Real world impact
  - Advanced model training
  - Responsive chatbots
  - Smaller infrastructure
  - Cost reduction
- Best practices
  - Enable Flash Attention by default
  - Layer optimizations
  - Profile before scaling
  - Monitor throughput: focus on tokens/sec, not FLOPS, as memory bandwidth often limits performance

### 181. 179. Multi-Node Distributed Training for LLMs
- Why multi-node trainig?
  - Training requires hundreds to thousands of GPUs 
- Core parallelism dimensions
  - Data parallelism
  - Model parallelism: tensor/pipeline parallelism
  - 3D parallelism + Hybrid: DP+TP+PP and ZeRO/FSDP
- Communication backbone
  - NCCL
  - HW interconnects
  - Hierarchical communication
  - Fault tolerance
- Orchestration and scheduling
  - HPC environment: Slurm
  - Cloud deployment: K8 + Ray, dynamic cluster management
- Challenges in multi-node training
  - Stragglers: slow nodes bottleneck synchronous training
  - Network bottlenecks
  - Storage demands
  - Debugging complexity
- Optimization strategies
  - Computation optimization
    - Mixed precision
    - Gradient compression using quantization and sparsification
  - Communication optimization
    - Overlap gradient communication with backward pass computation
    - Map pipeline parallelism to physical network topology
    - Dynamic GPU allocation for elasticity during training
- Real world examples
  - GPT-3: 175B parameters on 1,024 V100 GPUs
  - BLOOM: 176B parameters on 384 A100 GPUs
- Best practices
  - Performance metrics
  - Scaling strategy
  - Monitoring & debugging
  - Automation

### 182. 180. Fine-Tuning vs Parameter-Efficient Tuning
- Why tuning matters
  - General-purpose foundation
  - Domains sepcialization
  - Resource constraints
- Full fine-tuning
  - Update all parameters
  - Advantages
    - Maximum flexibility
    - Highest potential accuracy/performance
    - Full control over model behavior  
  - Limitations
    - Requires enormous compute resource
    - High storage costs
    - Needs substantial training data
    - Deployment challenges for multiple versions
- Parameter-EFficient Tuning (PEFT)
  - Selective updates: only tune a small subset of paramters (0.1-3%)
  - Lightweight deltas: stores small weight changes on top of frozen base model
  - Multi-task efficiency
  - Key PEFT methods
    - LoRA: injects trainable low-rank matrices into attention layers
    - Adapters: adds small bottleneck layers b/w transformer blocks
    - Prefix/Prompt tuning: learns continuous prompt embeddings that are prepended to inputs
    - P-tuning & BitFit: tunes only bias terms in mokdel weights
- PEFT advantages
  - 10-100x cheaper training costs
  - 10-1000x smaller storage
  - Faster training cycles
  - Simpler deployment
- Use cases
  - Domain adaptation
  - On-device personalization
  - Multilingual fine-tuning
  - Rapid prototyping
- Best practices  
  - Start with PEFT
  - Track trade-offs
  - Hybrid approaches: combine PEFT with quantization for edge deployment

### 183. 181. Cost Challenges of Generative AI Training
- Balancing scale, performance, and economics
- Why costs explode
  - Scale issues: billions-trillions of parameters
  - HW demands: thousands of GPUs
  - Hidden expenses: storage + bandwidth, energy & cooling
- Key cost drivers
  - Compute: GPU/TPU
  - Data: collecting, cleaning, and storing petabyte 
  - Networking: high-bandwidth interconnects
  - Storage
  - Energy: power + cooling
- Hidden & indirect costs
  - Experimentation overhead: failed runs, hyperparameter tuning, instabilities, bugs
  - Idle capacity
  - Engineering time
  - Cloud egress fees  
- Optimization levers
  - Mixed precision (FP16/BF16)
  - ZeRO/FSDP sharding
  - Flash attention & fused kernels
  - Gradient accumulation & scheduling
- Cost-aware strategies
  - Spot/preemptible GPU usage
  - Elastic training
  - Checkpoint optimization
  - Model distillation
- Business trade-offs
  - Size vs efficiency
  - Build vs adapt
  - ROI calculation
  - Total cost of ownership

### 184. 182. Lab – Fine-Tune a Small LLM with PEFT
- Learning Goals
  - Understand parameter-efficient fine-tuning (PEFT)
  - Apply LoRA adapters to a small LLM
  - Train on a sample dataset with minimal compute
  - Evaluate and generate text from the fine-tuned model
```
0) Prerequisites

    Python 3.9+

    GPU recommended (Colab, Kaggle, or local CUDA)

    Install dependencies:

    mkdir peft-lab && cd peft-lab
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install torch transformers datasets peft accelerate bitsandbytes

1) Load a Base Model

We’ll use DistilGPT-2 (tiny, fast to train).

    from transformers import AutoModelForCausalLM, AutoTokenizer
     
    MODEL_NAME = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

2) Load a Toy Dataset

We’ll use a subset of the SST2 (sentiment) dataset for demonstration.

    from datasets import load_dataset
     
    dataset = load_dataset("sst2")
    train_texts = [f"Review: {x['sentence']} Sentiment: {x['label']}" for x in dataset['train'][:2000]]

Tokenize:

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
     
    dataset = dataset["train"].select(range(2000)).map(lambda x: {"text": f"Review: {x['sentence']} Sentiment: {x['label']}"})
    tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=64), batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

3) Apply LoRA with PEFT

    from peft import LoraConfig, get_peft_model, TaskType
     
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=8,               # rank
        lora_alpha=16,     # scaling
        lora_dropout=0.1
    )
     
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

👉 Only a few million params will be trainable.
4) Training Setup

    from transformers import Trainer, TrainingArguments
     
    training_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=8,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=2e-4,
        fp16=True
    )
     
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )

5) Train the Model

    trainer.train()

Watch logs → you should see loss decreasing. Training should finish in minutes on GPU.
6) Generate with the Fine-Tuned Model

    prompt = "Review: The movie was exciting and"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

You should see the model generate text with sentiment-flavored completions.
7) Save & Reload Adapters

    model.save_pretrained("lora-finetuned")
    tokenizer.save_pretrained("lora-finetuned")

Reload with:

    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(base_model, "lora-finetuned")

8) Stretch Goals

    Try different datasets (e.g., IMDB reviews, dialogue data)

    Swap to a larger model (e.g., gpt2, facebook/opt-1.3b)

    Experiment with other PEFT methods (prefix tuning, adapters)

    Deploy with FastAPI (like in Lab 175 RAG)

✅ Outcome: You fine-tuned a small GPT model with LoRA adapters → faster, cheaper, and lightweight vs full fine-tuning.
```

## Section 28: Week 27: Infrastructure for Computer Vision at Scale

### 185. 183. Image and Video Data Challenges in AI Infra
- Why vision data is hard
  - Scale issues: massive volumes from frame/sec $\times$ devices
  - Format complexity
  - Label challenges
  - Edge constraints: strict latency + bandwidth limitations b/w edge and cloud
- Data ingestion and stroage architecture
  - High-throughput ingest
  - Tiered storage
  - Metadata indexing: chunk and index by time, camera ID, and scene for efficient retrieval
  - Granula access
- Compression, codecs, and formats
  - Critical trade-offs
    - Bitrate vs visual quality vs CPU/GPU decode cost
    - Higher compression == more compute
    - 10x storage difference b/w raw and compressed
  - HW acceleration
    - Leverage NVDEC(Nvidia) and VAAPI (Intel) for decode
    - Dedicated decode engines free compute resource
    - 4-10x performance gain vs SW decode
  - Format standardization
    - H.264/H.265/AV1 when available
    - Store codec metadata for optimized processing
    - Track keyframe indices for random access
- Labeling & curation challenges
  - Active learning loops: prioritize uncertain/hard samples for human reviews
  - Bootstrapped labeling: Use weak/auto labels initially with human-in-the-loop QA to scale annotation effect
  - De-duplication
  - Bias mitigation
- Feature & dataset management
  - Version control
  - Augmentation tracking
  - Embedding precomputation
  - Compliance metadata
- Training Pipeline Bottlenecks
  - IO optimization
  - Dataloader engineering: async dataloader with prefetching; cache preprocessed shards
  - Precision & accumulation: mixed precision training
  - Distributed strategies
- Serving & latency constraints
  - Real-time processing: end-to-end latency under 100ms
  - Batching strategies: Maximize throughput
  - Tracking optimization
  - Edge processing
- Edge-to-cloud architectures
  - Edge processing
    - Capture raw video streams
    - Lightweight filtering
    - GEnerate event triggers for relevant content
    - Buffer important segments locally
  - Transport layer
    - Use message buses (Kafka/MQTT) for event transport
    - Implement reliable delivery with at-least-once semantics
      - Data durabilty over uniqueness
    - Synchronize clocks via PTP/NTP for multicamera analytics
  - Cloud processing
    - Run computationally intensive models
    - Perform multi-camera correlation and analysis
    - Manage long-term storage and archiving
    - Enable historical analysis and model training
- Security and compliance
  - Data protection: TLS for streams, KMS for storage encryption
  - Privacy preservation: on-device blurring, consent tracking
  - Access controls: role-based access, tamper-proof logs
  - Regional compliance: GDPR, CCPA

### 186. 184. Training Pipelines for CV Models
- Why pipelines matter
  - Repeatable automated flows
  - End-do-end management: preprocessing, augmentation, training, and comprehensive logging
  - Production scalability
- Data preparation stage
  - Ingest: from cloud, edge, or archives
  - Normalize: resize, color space, channels
  - Validate: missing/corrupted files
  - Privacy: Face/plate blurring when needed
- Augmentation & enrichment
  - Geometric transforms: flip, rotate, crop, scale
  - Photometric transforms: brightness, constrast, saturation
  - Noise & degradation: noise, blur, cutout
  - Policy-driven approaches
- Feature & label management
  - Consistency is critical
    - Maintain uniform class labels across dataset versions
    - Track dataset splits with metadata
    - Document annotation
  - Optimization opportunities
    - Precompute embeddings for similarity checks
    - Imlpement feature stores for embedding reuse
    - Cache processed data to improve training startup time
- Model training stage
  - Transfer learning: pretrained CNNs/Vision transformers
  - Optimization: mixed precision
  - Distribution: GPUs with DDP/FSDP for larger models
  - Logging: MLflow, W&B, or ClearML
- Validation & monitoring
  - Performance metrics: accuracy, precision, recall, mAP
  - Data drfit detection: monitor for new classes
  - Failure analysis
  - Alert systems
- Automation tools
  - Kubeflow pipelines: containerized ML workflows
  - Airflow/Prefect: scheduling and dependency management for preprocessing + training
  - MLflow: experiment tracking, artifact storage, and model registry
  - Weight & Biases
- Best practices
  - Version everything
  - Measure what matters
  - Optimize for deployment conditions, not just training accuracy

### 187. 185. EfficientNet and Model Scaling Tradeoffs
- Fundamental tension in CV
  - Bigger models: higher accuracy but significantly slower inference and more computational demands
  - Smaller models: faster processing but risk underfitting on complex visual recognition tasks
- EfficientNet's compound scaling appraoch
  - Depth: more layers to capture complex features
  - Width: more channels
  - Resolution
- The EfficientNet Family
  - B0 (baseline): compact model (5.3M params)
  - B1-B3 (Mid-range)
  - B4-b7 (Large): high capacity model (66M params)
- Scaling trade-offs in practice
  - B0 to B3 may increase inference time by 2-5x
  - B0 -> B7 may consume 10-15x more GPU RAM
  - B5 -> B7 yields 2% accuracy gain while 4x more parameters
  - B0 -> B7 may consume 8x more energy
- Deployment scenarios
  - Edge devices: EfficientNet-Lite, MobileNet, and int8-quantized models
  - Real-time video analytics: B0-B2 (15-30fps)
  - Cloud inference: B3-B5
  - Research applications: B6-B7
- Beyond EfficientNet: emerging scaling paradigms
  - Vision transformers
  - Neural Architecture Search
  - Mixture of Experts (MoE)  
  - ConvNeXt/RegNet
- Implementation best practices
  - Establish a baseline
  - Profile key metrics
  - Apply optimizations
  - Match model to workload

### 188. 186. Serving Real-Time Video Inference
- From camera streams to AI predictions in milliseconds
- Why real-time matters
  - Stream density: 30-60 fps
  - Latency budget: ultra-low latency response of less than 100ms per frame
- Core pipeline for video inference
  - Capture
  - Decode
  - Preprocess
  - Inference
  - Postprocess
- Performance challenges
  - GPU saturation
  - Bandwidth constraints
  - Multi-camera synchronization
  - Accuracy/speed tradeoffs
- Infrastructure optimizations
  - Frame batching
  - Pipeline parallelism
  - Inference optimization
  - Edge preprocessing  
- Scaling architectures
  - Deployment models
    - Single GPU edge
    - Multi-stream servers
    - Hybrid edge-cloud
  - Use message brokers (Kafka, MQTT) for reliable and scalable distribution
- Monitoring & reliability
  - Performance metrics: FPS, latency
  - Health monitoring
  - Resiliency features
- Use cases
  - Smart cities
  - Retail analytics
  - Industrial safety
  - Sports analytics
  
### 189. 187. DeepStream SDK for Video Analytics
- DeepStream
  - Nvidia's SDK for video understanding applications at scale
  - Real-time multi-stream
  - Cross-platform
  - GStreamer-based
- DeepStream pipeline solution workflow
  - Decode
  - Preprocess
  - Inference
  - Postprocess
  - Output
- Core components
  - nvinfer: supports TensorRT and ONNX models
  - nvtracker: implements SORT and DeepSORT algorithms
  - nvvideoconvert: formation conversion
  - message broker: publishes metadata to Kafka, MQTT, and REST endpoints
- Deployment scenarios
  - Jetson Edge: self-contained AI at factories, retail stores, and hospitals
  - Multi-stream server: hundreds of cameras in smart city deployments
  - Cloud-hosted analytics
  - Hybrid edge-cloud: edge inference + analytics at cloud
- Integration ecosystem
  - Model serving: Triton inference server
  - Event streaming: Kafka, RabbitMq, and MQTT
  - Data storage: Store metadata in RDBMS
  - Monitoring: Grafana + Prometheus integration
- Best practices
  - Profile pipeline performance
  - Optimize models with TensorRT
  - Balance GPU workloads
  - Implement batching strategies

### 190. 188. Edge-to-Cloud Video Processing
- Why Edge-to-Cloud?
  - Video streams generate massive data volumes
  - Sending all raw frames to cloud is expensive and slow
  - Edge processing cuts latency & saves bandwidth
- Edge processing capabilities
  - Real-time inference
  - Pre-filtering
  - On-device caching
  - Privacy filters
- Cloud processing capabilities
  - Heavy compute processing
  - Cross-camera correlation
  - Long-term storage and replay
  - System integration
- Data flow architecture
  - Capture
  - Preprocess + infer
  - Transport: Kafka, MQTT, REST
  - Cloud  
- Performance challenges
  - Bandwidth bottlenecks
  - Synchronization issues
  - Edge model accuracy
  - Cloud costs
- Optimization strategies
  - Intelligent compression
  - Event-driven transmission: send only anomalies/events, not continuous streams
  - Model cascading
  - Distribution optimization: use CDN & regional cloud zones
- Use cases
  - Smart factories
  - Retail analytics
  - Transportation: traffic signal control
  - Security
- Best practices
  - Event-driven architecture
    - Threshold-based triggers
    - Metadata-first approach
  - Edge privacy protection
  - Performance monitoring
  - Hybrid deployement: autoscaling in cloud

### 191. 189. Lab – Deploy Real-Time Object Detection
- Learning Goals
  - Run a pretrained object detection model (YOLOv8)
  - Perform inference on live video or webcam stream
  - Deploy a FastAPI inference server for real-time detection
  - Visualize detections with bounding boxes
```
0) Prerequisites

    Python 3.9+

    GPU recommended (CUDA device or Colab)

    Install dependencies:

    mkdir object-detection-lab && cd object-detection-lab
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install ultralytics fastapi uvicorn opencv-python-headless python-multipart

1) Run YOLOv8 Locally (Quick Test)

    from ultralytics import YOLO
     
    # Load pretrained COCO model
    model = YOLO("yolov8n.pt")  # n = nano, fast & lightweight
     
    # Run inference on an image
    results = model.predict("https://ultralytics.com/images/bus.jpg", show=True)
     
    for r in results:
        print(r.boxes.xyxy)  # print bounding boxes

✅ You should see a bus, people, and objects detected.
2) Real-Time Webcam Detection

File: webcam.py

    import cv2
    from ultralytics import YOLO
     
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)  # 0 = default webcam
     
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow("YOLOv8 Real-Time", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
     
    cap.release()
    cv2.destroyAllWindows()

Run:

    python webcam.py

Press q to quit.
3) Deploy FastAPI Inference Server

File: server.py

    from fastapi import FastAPI, UploadFile, File
    from ultralytics import YOLO
    import cv2
    import numpy as np
     
    app = FastAPI()
    model = YOLO("yolov8n.pt")
     
    @app.post("/detect/")
    async def detect(file: UploadFile = File(...)):
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = model(img)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })
        return {"detections": detections}

Run server:

    uvicorn server:app --reload --port 8000

4) Test API with curl

    curl -X POST "http://127.0.0.1:8000/detect/" \
      -F "file=@bus.jpg"

Output (sample):

    {
      "detections": [
        {"class": "bus", "confidence": 0.89, "bbox": [34, 56, 280, 190]},
        {"class": "person", "confidence": 0.77, "bbox": [310, 80, 400, 250]}
      ]
    }

5) Optional – Streamlit UI for Easy Testing

    pip install streamlit

File: app.py

    import streamlit as st
    from ultralytics import YOLO
    import cv2, numpy as np
     
    model = YOLO("yolov8n.pt")
    st.title("Real-Time Object Detection")
     
    uploaded = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        results = model(img)
        st.image(results[0].plot(), channels="BGR")

Run:

    streamlit run app.py

6) Stretch Goals

    Switch to YOLOv8m or YOLOv8l for higher accuracy

    Connect FastAPI with Kafka/MQTT for edge → cloud video streams

    Deploy with Triton Inference Server for GPU optimization

    Quantize with INT8 to run on Jetson/edge devices

✅ Outcome: You deployed a real-time object detection pipeline, accessible via webcam, API, or web UI.
```

## Section 29: Week 28: Infrastructure for NLP at Scale

### 192. 190. NLP Workloads: Tokenization, Embeddings, Transformers
- Challenges to NLP process:
  - Unicode variations and normalization challenges
  - Inconsistent casing and formatting
  - Special content like emojis and code blocks
  - Throughput bottlenecks at tokenizationand batching stages
- Tokenization basics
  - Whitepsace/word tokenization
  - Subword tokenization
  - Out-of-Vocabulary handling
- Practical tokenization tips
  - Preprocessing
  - Performance optimization
  - Sequence management
  - Versioning
- Embeddings 101
  - What are embeddings?
    - Dense vector representations of text that capture semantic meaning
    - Fixed dimension floating point vectors (128-1536)
    - Measure similarity via cosine/dot product/L2
    - Enable semantic matching beyond keyword search
  - Use cases
    - RAG
    - Sematic search
    - Clustering & deduplication
    - Recommendation
- Building embedding pipelines
  - Optimize throughput
  - Store & index: FAISS, Pinecone, Weaviate, store metadata alongside vectors for filtering
  - Balance tradeoffs    
- Transformers at a glance
  - Encoder-only (BERT)
    - Bidirectional context
    - Classification and encoding
    - Ex: sentiment, NER, embedding
  - Decoder-only (GPT)
    - Unidirectional (left-to-right)
    - Text generation
    - Ex: completion, chat, QA
  - Encoder-decoer (T5)
    - Full sequence transformation
    - Sequence-to-sequence tasks
    - Ex: translation, summarization
- The attention mechanism
  - Learning contextual relationship b/w words
  - Capturing long-range dependencies in text
  - Parallel processing of sequence elements
  - Self-attention: N^2 complexity with sequence length
- Throughput & latency levers
  - Sequence length
  - Optimization techniques: Flash attention, mixed precision
  - Efficent batching: dynamic batching +  KV-cache
  - Quantization: INT8/INT4 for edge deployment
- Quality and evaluation
  - Classification metrics
    - Accuracy, precision, recall
    - F1 score
    - ROC-AUC
    - Confusion matrix
  - Generation metrics
    - BLEU, ROUGE (n-gram overlap)
    - METEOR (synonym matching)
    - BERTScore (semantic similarity)
    - Perplexity (prediction confidence)
  - Retrieval metrics
    - Recall@k (found relevant items)
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (nDCG)
    - Latency-at-percentile (p95,p99)
  - Human evaluation
    - Factual accuracy and hallucination detection
    - Safety and bias assessment
    - Writing style and tone appropriateness
    - Overall usefulness for intended application
- Best practices
  - Version management
  - Monitoring
  - Guardrails: profanity filters and PII detection

### 193. 191. Training Large Transformer Models
- Why transformers dominate
  - Long-range dependencies
  - Strong generalization
  - Cross-domain adaptability
- Training challenges
  - Model size: billions of params exceed a single GPU memory
  - Compute requirements
  - Data scale: petabyte scale datasets
  - Training duration
- Core training techniques
  - Data parallelism
  - Model/tensor parallelism
  - Pipeline parallelism
  - 3D parallelism: DP+TP+PP (MegaTron-LM, DeepSpeed, Alpa)
- Memory optimization
  - Mixed precision
  - Gradient checkpointing
  - ZeRO/FSDP
  - CPU/NVMe offloading
- Efficiency boosters
  - Flash attention
  - Fused operations
  - Gradient accumulation
  - Elastic scaling
- Data pipeline requirements
  - High-performance tokenization
  - Streaming architecture
  - Data quality processing
  - Batch construction
- Checkpointing & fault tolerance
  - Comprehensive state saving
  - Sharded checkpoint IO
  - Asynchronous operations
  - Validation and redundancy
- Evaluation during training
  - Perplexity tracking
  - Benchmark evaluation
  - Scaling law analysis
- Infrastructure needs
  - HW requirements
    - Multi GPU servers
    - 200-400 Gbps infiniband/RoCE
    - High-bandwidth NVMe storage
    - Specialized cooling systems
  - Orchestration & monitoring
    - Job scheduler
    - Distributed training framework
    - Real-time metrics: tokens/sec, GPU utilization, memory usage
    - Cost tracking: $/epoch
- Best practices
  - Start small, scale gradually
  - Comprehensive logging
  - Automated recovery
  - Realistic resource planning

### 194. 192. Infrastructure for BERT and GPT
- BERT: pioneered bidirectional pretraining
- GPT: popularized autoregressive generation for text completion
- BERT infrastructure needs
  - Encoder-only training
  - Data scale
  - Downstream tasks
  - Serving requirements: need fast embedding serving, latency sensitive, not compute intensive
- GPT infrastructure needs
  - Decoder-only architecture
  - Memory challenges: long-sequence training introduces N^2 memory costs
  - Inference optimization: 
  - Application focus
  - Cache-heavy infrastructure is required
- Training pipelines
  - Distributed parallelism
  - Precision optimization
  - Memory management
  - Observabilty
- Data engineering for transformers
  - Tokenization
  - Data cleaning
  - Versioning
  - IO optimization
- Serving BERT models
  - BERT models typically power latency-sensitive embedding generatin for search, recommendation, and classification systems
  - Optimization techniques
    - Deploy via ONNX or TensorRT for 3-5x speedup
    - INT8 quantization for 2-4x throughput improvement
    - Batch queries intelligently for higher GPU utilization
  - Deployement architecture
    - Scale with k8 + HPA based on GPU utilization
    - Implement embedding cache for frequent queries
    - Monitor latency p50/p95/p99 as critical SLIs
- Serving GPT models
  - GPT inference requires low-latency for interactive chat
  - KV-cache management
  - Request optimization
  - Deployment options
  - Continuous batching
- Best practices
  - BERT optimization: apply distillation + quantization to reduce size by 75% + while maintaining 95%+ of accuracy
  - GPT optimization: implement PEFT (LoRA, P-Tuning) for 99% parameter reduction during fine-tuning
  - Production monitoring: monitor embedding drift and generation quality with automated evaluation pipelines

### 195. 193. Efficient Serving of Embedding Models
- Embeddings are the foundation of modern AI
  - RAG, semantic search, and recommendation systems
  - FAce demanding workloads with high QPS + low-latency requirements
  - Balance from throughput, latency and memory usage
- Core challenges in embedding serving
  - High dimensionality: thousands of dimensions create significant memory pressure and computational demands
  - Tokenization overhead: expensive tokenizations cause a CPU bottleneck at scale
  - Latency constraints
  - Multi-tenant complexity
- Infrastructure patterns for embedding services
  - Deployment architecture
    - Expose as microservice APIs via FastAPI, gRPC, or Triton inference server
  - Performance optimization
    - Leverage async IO patterns
    - Configure intelligent load-balancing across multiple GPU pods
- Technical optimizations for embedding efficiency
  - Quantization
  - Strategic caching
  - HW selection: smaller models on CPU
  - Pre-computation
- Critical metrics to track
  - Performance: p50/p95/p99 latency, tokens/sec, embeddings/sec, GPU utilization
  - Quality: embedding drift metrics, semantic shift detection, retraining triggers
  - Efficiency: request deduplication rates, cache hit/miss ratios, cost per embedding
- Integrating with vector databases
  - Full pipeline: Embed -> index -> retrieve -> rerank
  - FAISS
    - Excellent for in-moemroy workloads
    - Supports CPU/GPU acceleration
    - Requires custom scaling solutions
  - Pinecone
    - Simplified operations
    - Auto-scaling capabilities
    - Higher operational costs
  - Weaviate  
    - Rich filering capabilities
    - GraphQL-based API
- Production use cases
  - Information retrieval
    - RAG pipelines
    - Semantic search
  - Matching & analysis
    - Real-time recommendations
    - Anomaly detection
- Engineering best practices
  - Performance benchmarking
  - Vector normalization
  - Versioning strategy
  - Index selection

### 196. 194. Latency Reduction in NLP Inference
- Why latency matters
  - User experience: each 100ms delay reduces engagement by 8%
  - Cost efficiency
  - System reliability
- Key sources of latency
  - Tokenization
  - Model loading
  - Attention comlexity: N^2 for long sequences
  - Infrastructure overhead
- Model level optimizations
  - Quantization
  - Pruning
  - Distillation
  - Optimized runtimes: TensorRT, ONNX runtimes, vLLM
- Runtime optimizations
  - KV-cache reuse
  - Flash attention
  - Batching & micro-batching
  - Memory management
- Infrastructure optimizations
  - Deployment options
    - Triton & Ray Serve
    - Autoscaling
- Monitoring latency
  - Track percentiles
  - Decompose metrics
  - End-to-end tracing
  - SLOs & Alerts
- Use cases
  - Chatbots 
  - Search & RAG
  - Streaming Apps
  - Edge applications
- Best practices
  - Profile the full pipeline
  - Implement strategic caching
  - Test under peak load
  - Balance trade-offs: latency vs accuracy vs cost

### 197. 195. Deploying Multilingual Models at Scale
- Core challenges
  - Vocabulary size
  - Inference latency
  - Quality variance: 15-30% performance gaps b/w high-resource (english, chinese) and low-resource language(Nepali)
  - Fairness & bias
- Infrastructure demands
  - Larger embedding tables -> 2.5x memory footprint 
  - Smart batching across mixed-language queries to maximize GPU utilization
  - Cache common embeddings/prompts for frequent languages to reduce redundant compuation
  - Ex: A production multilingual BERT service typically requires 3-4x HW resources of its monolingual counterpart
- Training & fine-tuning strategies
  - Foundation pretraining: start with large, diverse multilingual corpora (CommonCrawl, Wikipedia in 100+ languages)
  - Efficient adaptation: PEFT enables language-specific tuning while sharing base parameters
  - Data-centric approach
- Serving optimizations
  - Infrastucture solutions
    - Deploy via Triton, vLLM, Ray Serve for scalable, distributed inference
    - Region-based autoscaling -> reduce latency for global users
  - Model optimizations
    - Quantization
    - Hybrid inference: fallback to smaller models for rare languages
- Monitoring & evaluation
  - Performance metrics
  - Quality assessment
  - User feedback
- Use cases
  - Multilingual chatbots & copilots
  - Cross-lingual semantic search & RAG
  - Translation services
  - Global content moderation
- Best practices
  - Unified architecture
  - Geographic distribution
  - Continuous evaluation
  - Consistent preprocessing


### 198. 196. Lab – Deploy a BERT Model with FastAPI
- Learning Goals
  - Load a pretrained BERT model for text classification
  - Expose it as a FastAPI endpoint
  - Send queries and receive predictions in JSON format
  - Understand how to containerize/deploy in production
```
0) Prerequisites

    Python 3.9+

    Install dependencies:

    mkdir bert-fastapi-lab && cd bert-fastapi-lab
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install torch transformers fastapi uvicorn

1) Load Pretrained BERT Model

We’ll use distilbert-base-uncased-finetuned-sst-2-english for sentiment classification.

File: model.py

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
     
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
     
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
     
    labels = ["negative", "positive"]
     
    def predict(text: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        return {labels[i]: float(probs[i]) for i in range(len(labels))}

2) Build FastAPI App

File: server.py

    from fastapi import FastAPI
    from pydantic import BaseModel
    from model import predict
     
    app = FastAPI()
     
    class Query(BaseModel):
        text: str
     
    @app.post("/classify/")
    def classify(query: Query):
        result = predict(query.text)
        return {"input": query.text, "prediction": result}

3) Run the API Server

    uvicorn server:app --reload --port 8000

    API runs at: http://127.0.0.1:8000

    Docs at: http://127.0.0.1:8000/docs (auto-generated Swagger UI)

4) Test the Endpoint

Using curl:

    curl -X POST "http://127.0.0.1:8000/classify/" \
      -H "Content-Type: application/json" \
      -d '{"text": "I really enjoyed this movie"}'

Sample output:

    {
      "input": "I really enjoyed this movie",
      "prediction": {
        "negative": 0.02,
        "positive": 0.98
      }
    }

5) Optional – Add Batch Inference

Modify predict() in model.py to accept a list of texts:

    def predict_batch(texts):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return [{labels[i]: float(p[i]) for i in range(len(labels))} for p in probs]

Add a /batch endpoint in server.py.
6) Optional – Dockerize for Deployment

Dockerfile

    FROM python:3.10-slim
    WORKDIR /app
    COPY . .
    RUN pip install --no-cache-dir torch transformers fastapi uvicorn
    CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

Build & run:

    docker build -t bert-fastapi .
    docker run -p 8000:8000 bert-fastapi

7) Stretch Goals

    Add a /health endpoint for monitoring

    Connect with Prometheus/Grafana for latency metrics

    Deploy on Kubernetes/Ray Serve for scaling

    Extend to multilingual models (e.g., bert-base-multilingual-cased)

✅ Outcome: You deployed a BERT text classification API using FastAPI, with options for scaling, monitoring, and containerization.
```

## Section 30: Week 29: Infrastructure for Multimodal AI

### 199. 197. What Is Multimodal AI?
- Multimodal
  - Beyond text: real world signas in diverse formats
  - Complementary fusion
  - Richer experiences
  - Enhanced RAG: text + image + audio + video
- Core concept of multimodal AI
  - Modality: distinct data types including text, image, audio, video, tabular data, and sensor readings
  - Encoders: BERT for text, ResNet for images, etc
  - Fusion: methods to combine embeddings across modalities: early (raw inputs), late (decisions), or cross-attention (transformers)
  - Alignment: creating a shared semantic space where different modalities can understand and relate to each other
- Model archetypes
  - Dual-encoder
  - Encoder-decoder
  - Unified transformers
  - Tool-use models  
- Typical multimodal pipelines
  - Perception: ASR for audio, OCR for image/documents, object/scene detectors for video frames
  - Embedding: create dense vector representations for each modality chunk, frame, or segment
  - Fusion/retrieval: cross-modal search or attention-based fusion of relevant information
  - Generation/decision: produce outputs of captions, answers, search results, recommendations
- Key infrastructure challenges
  - Scale
  - Synchronization: aligning audio, video, and text timestamps for coherent understanding
  - Latency
  - Quality control
- Common use cases
  - Search
  - Assistants
  - Commerce
  - Operations
- Best practices for implementation
  - Preprocessing: normalize and timestamp all inputs. preserve provenance metadata for traceability and debugging
  - Specialized processing: Use purpose-built encoders (ASR/OCR/ViT) for initial perception before fusion
  - Optimization: cache embeddings to avoid redundant computation; chunk long videos into shots/scenes
  - Responsible AI: implement guardrails for sensitive media including PII detection, face/license plate blurring, and audio consent management

### 200. 198. Handling Text + Image Pipelines
- Why text + image together?
  - Many tasks involve joint reasoning: captions, VQA, search
  - Images provide context, text adds semantics
  - Combining improves accuracy, robustness, and user experience
- Core pipeline stages
  - Ingest: collect text+images from database, APIs, user uploads
  - Preprocess: normalize text, resize/augment images, handle multiple languages
  - Encode: modality specific models (BERT, ViT, CLIP)
  - Fuse: combine embeddings into shared representation space
  - Serve: deliver results (retrieval, generation, classification)
- Fusion strategies
  - Early fusion: combine raw features before encoding
  - Late fusion: indepenent encoders -> combine embeddings
  - Cross-attention: transformer layers align modalities
- Infrastructure requirements
  - GPU acceleration
  - Vector database
  - Intelligent batching
  - Streaming pipelines
- Use cases
  - Visual search
  - Image captioning
  - Visual question answering
  - E-commerce
- Challenges
  - Data alignment
  - Bias & fairness
  - Latency
  - Scale
- Best practices
  - Standardize pipelines
  - Leverage pretrained models
  - Monitor relevance
  - Implement caching

### 201. 199. Training and Serving CLIP Models
- Bridging text and images with join embeddings
- CLIP
  - Developed by OpenAI to create a unified understanding b/w text and images
  - Web-scale training: trained on millions of image-caption pairs
  - Joint Embedding space: maps both text and images into the same high-dimensional vector space
  - Zero-shot capabilities: can classify images into arbitrary categories
- How CLIP works
  - CLIP uses dual encoder architecture to map text and images into a shared embedding space
    - Text Encoder: Transformer maps text tokens into a fixed-dimensional embedding vector
    - Image Encoder: Vision transformer (ViT) or ResNet processes image patches, creating a comparable embedding vector
  - During training, CLIP maximizes similartiy b/w matched image-text pairs while minimizing similarity b/w unmatched pairs using **contrastive loss**
- Training CLIP at scale
  - Data requirements
    - Millions to billions of diverse image-caption pairs
    - Extensive data cleaning: deduplication, NSFW filtering, caption normalization
    - Balanced domain coverage to prevent performance skew
  - Training infrastructure
    - Distributed training across GPU clusters
    - Mixed precision
    - 3D parallelism
    - Checkpointing for fault tolerance
  - Steps
    - Data processing: ETL pipelines for image-text pairs
    - Distributed training: scale across hundreds of GPUs
    - Evaluation: Zero-shot benchmarking
- Serving CLIP models
  - Offline processing: precompute and store embeddings for your entire corpus in a vector database
  - Query-time processing: encode the query (text or image), perform approximate nearest neighbor search, return top-k results
- Infrastructure optimizations
  - Request batching
  - Model quantization
  - Embedding caching
  - Deployment frameworks
- CLIP use cases
  - Semantic search
  - Intelligent captioning
  - Content moderation
  - Multimodal RAG
- Challenges in CLIP deployment
  - Data biases
  - Infrastructure scaling
  - Domain adaptation
  - Workload management
- Best practices
  - Model fine-turning
  - Retrieval strategies
  - Embedding management
  - Quality monitoring

### 202. 200. Infrastructure for Speech + Text Models
- Connect spoken language with NLP pipelines
- Why speech + text integration matters
  - Natural interface: speech is 3x faster than typing
  - Accessibility: enables hands-free operation
  - Multimodal AI: foundation for next-gen assistants
- Modern AI must handle both ASR (speech -> text) and TTS (text -> speech) for complete communication loops
- Core pipeline components
  - Audio capture
  - Preprocessing
  - ASR models: acoustic signals to text
  - NLP processing
  - TTS models: natural speech responses using neural vocoders (voice encoders)
- Infrastructure demands
  - Low-latency requirements
  - Streaming capabilities
  - Codec support: PCM, MP3, Opus, ...
  - GPU acceleration
- Serving speech pipelines
  - Streaming APIs: WebSocket and gRPC interfaces
  - Micro-batching: Group audio frames for optimal GPU
  - Containerization: Deploy ASR + NLP + TTS as a separate service
  - Response caching: store frequent TTS outputs
- Latency optimization techniques
  - Edge-cloud hybrid architecture
  - Model quantization
  - Model distillation
  - Parallel TTS processing
- Monitoring and reliability
  - Word Error Rate (WER)  
  - End-to-end latency
  - Throughput
  - Audio drift
- Real-world use cases
  - Voice assistants: Alexa, Siri, ...
  - Call analytics
  - Meeting intelligence
  - Accessibility
- Engineering best practices
  - Audio preprocessing standardization
  - Microservice architecture
  - Regional model deployment
  - Demographic evaluation

### 203. 201. Deploying Video + Text Search Systems
- Bridging natural language queries with visual content
- Why video+text search?
  - Rich but complex modality
  - Natural Language Interface
  - Cross-domain applications
  - Technical requirements: multimodal embeddings +  scalable infrastructure
- Core pipeline architecture
  - Ingest video
  - Preprocess
  - Embedding: CLIP for visual frames, BERT/LLM for text transcripts
  - Indexing: vector DB for efficient multimodal storage and retrieval
  - Query
- Infrastructure demands
  - Storage: petabyte-scale video storage
  - Compute
  - Indexing: FAISS/Pinecone/Weaviate for efficient vector similarity search
  - Serving: FastAPI/gRPC endpoints with robust caching layers
- Optimizations for scale & performance
  - Precompute embeddings
  - Hierarchical indexing
  - Vector compression
  - Result caching
- Real world use cases
  - Media
  - Security
  - Education
- Monitoring & evaluation  
  - Critical metrics
    - Recall@k: percentage of relevant results returned
    - Query latency: end-to-end response time
    - System throughput
    - User engagement
  - Cross-modal evaluation: separate benchmark across text-only, video-only, and cross-modal queries
- Best practices
  - Dual representation
  - Hybrid retrieval
  - Intelligent partitioning
  - Continuous improvement

### 204. 202. Challenges of Multimodal Model Serving
- Why multimodal serving is hard
  - Processing of different input formats simultaneously
  - Handling varied workloads from lightweight text to heavyweight video
  - Strict latency expectations for interactive pipelines
- Input complexity
  - Text: tokenization pipelines, embedding generation, and vocabulary management
  - Images: resizing, augmentation, CNN/ViT encoding
  - Audio: spectogram generation, ASR preprocessing, and feature extraction
  - Video: frame extraction, temporal alignment, and motion analysis
- Model fusion challenges
  - Dual-encoder models (like CLIP) process modalities separately then combine
  - Cross-attention fusion allows modalities to influence each other's processing
  - Both requires significant memory and compute demands
  - System designer must balance throughput with accuracy requirements
- Infrastructure constraints
  - Specialized GPUs
  - Decoding bottlenecks
  - Batching complexity
- Latency & throughput issues
  - Text: ms latency, p95/p99 met
  - Image: 10-100 ms, some p99 spike
  - Audio: 0.1-1 sec, p95 sometimes exceeded
  - Video/fusion: seconds, p95/p99 often exceeded 
- Scaling & deployment challenges
  - Memory constraints
  - Orchestration complexity
  - Varied autoscaling needs
  - Microservice architecture
- Monitoring & observability
  - Metrics
    - Speech: WER
    - Text: BLEU, perplexity
    - Images: Recall@k, precision
  - Additional monitoring
    - Drift across varied inputs (audio noise, image lighting)
    - End-to-end tracing for multi-hop, cross-modal queries
    - Unifying quality metrics
- Security and governance
  - Privacy concerns
  - Compliance requirements
  - Audit challenges
- Best practices
  - Modular architecture
  - Strategic caching
  - Model optimization
  - Regional deployment  

### 205. 203. Lab – Deploy CLIP for Image Search
- Learning Goals
  - Encode images & text into a shared embedding space
  - Store image embeddings in FAISS (vector database)
  - Query with natural language to retrieve similar images
  - Deploy as a FastAPI microservice
```
0) Prerequisites

    Python 3.9+

    Install dependencies:

    mkdir clip-search-lab && cd clip-search-lab
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install torch torchvision faiss-cpu fastapi uvicorn pillow transformers

1) Load CLIP Model

File: clip_model.py

    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
     
    MODEL_NAME = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
     
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
     
    def embed_image(image_path: str):
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        return emb.cpu().numpy()
     
    def embed_text(query: str):
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
        return emb.cpu().numpy()

2) Build Image Index with FAISS

File: build_index.py

    import os, faiss, pickle
    from clip_model import embed_image
     
    image_folder = "images"   # put your sample JPG/PNG images here
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
     
    embs, metadata = [], []
    for path in image_paths:
        embs.append(embed_image(path))
        metadata.append(path)
     
    import numpy as np
    embs = np.vstack(embs).astype("float32")
     
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
     
    with open("clip_index.pkl", "wb") as f:
        pickle.dump((index, metadata), f)
     
    print(f"Indexed {len(metadata)} images.")

Run:

    python build_index.py

3) Create Search Function

File: search.py

    import faiss, pickle, numpy as np
    from clip_model import embed_text
     
    with open("clip_index.pkl", "rb") as f:
        index, metadata = pickle.load(f)
     
    def search_images(query: str, k=3):
        qvec = embed_text(query).astype("float32")
        D, I = index.search(qvec, k)
        results = [metadata[i] for i in I[0]]
        return results

Test in REPL:

    from search import search_images
    print(search_images("a dog playing in the park"))

4) Deploy FastAPI Service

File: server.py

    from fastapi import FastAPI
    from pydantic import BaseModel
    from search import search_images
     
    app = FastAPI()
     
    class Query(BaseModel):
        text: str
     
    @app.post("/search/")
    def search(query: Query):
        results = search_images(query.text, k=5)
        return {"query": query.text, "results": results}

Run:

    uvicorn server:app --reload --port 8000

5) Test API

    curl -X POST "http://127.0.0.1:8000/search/" \
      -H "Content-Type: application/json" \
      -d '{"text": "a man riding a bicycle"}'

Sample response:

    {
      "query": "a man riding a bicycle",
      "results": ["images/bike1.jpg", "images/bike2.jpg", "images/bike3.jpg"]
    }

6) Stretch Goals

    Add an image upload endpoint to dynamically insert new images

    Store embeddings in Weaviate or Pinecone for scale-out search

    Build a Streamlit UI to visualize retrieved images

    Quantize CLIP model for faster inference on CPU/edge

✅ Outcome: You built and deployed a CLIP-powered image search API, capable of finding images based on natural language queries.
```

## Section 31: Week 30: Infrastructure for Reinforcement Learning

### 206. 204. Basics of Reinforcement Learning Workloads
- What makes RL different?
  - Data generation: data is generated, not given through agent environment interaction loop
  - Shifting distribution
  - Reward-based learning: no explicit labels
  - Infrastructure demands: fast simulations, scalable rollout systems
- Core RL loop
  - Observe state
  - Choose action
  - Policy update
  - Environment response
- Common algorithms
  - Policy gradient/PPO
    - On-policy approach with clipping for stability
    - Best for general problems requiring stable convergence
  - Actor-critic/A2C/A3C
    - Combines parallel actors with a value baseline
    - Best for distributed training across many CPUs
  - DQN/Rainbow
    - Value based approach for discrete action spaces
    - Best for environments with clear state representations and limited action choices
  - SAC/TD3
    - Off-policy algorithms for continuous control
    - Best for robotics and physical control tasks requiring precise movements
- RL workload components
  - Environment & rollout
    - Environment: Gym/Isaac/Unity simulators
    - Rollout workers: Generate trajectories from agent-environment interaction
  - Learning & evaluation
    - Replay buffer/batcher: store experiences for off-policy or on-policy learning
    - Learner: GPU-accelerated neural network training
    - Evaluator: periodic checks of policy performance
- Throughput & scaling
  - Horizontal scaling
  - Actor-learner separation
  - Asynchronous queueing
  - Vectorized environments
- IO & serialization
  - Challenges
    - Trajectories are small but numerous _> high messaging overhead
    - Image observation can create bandwidth bottlenecks
    - Environment-agent communication can become a throughput limiter
  - Solutions
    - Optimize IPC (Inter-process communication)
    - Compress observations
    - Shared memory: zero-copy transfers where available
    - Version everything
- Training stability
  - Normalization and clipping
  - Curriculum learning
  - Target networks
  - Early stopping
  * RL is notoriously unstable !!!
- Infra & tooling
  - Training frameworks
    - Ray RLlib
    - CleanRL
  - Experiment tracking
    - weight & biases
    - MLflow
  - Infrastructure
    - Docker container
    - K8/slurm
- Serving RL policies
  - Model export: TorchScript, ONNX, or TensorRT
  - Real-time control
  - Safety measures
  - Monitoring & updates

### 207. 205. Simulation Environments for RL (Gym, Isaac)
- A sandbox for RL prior to running robots
- Why simulation matter
  - Safety and exploration
  - Cost & efficiency
  - Control & repeatability
- OpenAI Gym: the classic choice
  - Lightweight python interface designed for RL research
  - Standardized API: reset(), step(action)
  - Diverse environment collection from simple CartPole to complex MuJoCo
  - Perfect for algorithm prototyping and benchmarking
- Gym's ecosystem advantages
  - Rapid iteration
  - Rich extensions
  - Flexibility
- Nvidia Isaac Gym: GPU-accelerated physics
  - Scale revolution
  - CUDA-powered physics: built on PhysX
  - Robotics focus
- Isaac Gym's technical edge
  - End-to-end GPU acceleration
  - Domain randomization
  - Comprehensive physics
- Alternative simulation environments
  - Unity ML-agents
  - DeepMind lab/Habitat
  - CARLA
  - Gazebo/MuJoCo
- Infrastructure considerations
  - Compute architecture: CPU heavy simulations (OpenAI Gym) or GPU-parallel environment (Isaac Gym)
  - IO communication
  - Reproducibility
  - Deployment
- Scaling your simulations with vectorization strategies
  - Run multiple environment copies per process to amortize overhead
  - Distribute rollout workers across compute nodes using Ray RLlib or MPI
  - Implement asynchronous stepping to prevent slow environments from creating bottlenecks
- Best practices for RL simulation
  - Start simple, then scale
  - Measure performance
  - Ensure robustness
  - Maintain reproducibility

### 208. 206. Distributed RL Training Infrastructures
- Why distributed RL?
  - Sample hungry
  - Speed limitations
  - Scale benefits
- Core distributed architectures
  - Synchronous
    - A2C, IMPALA-style architectures
    - Uses barriers for policy/value updates
    - More stable learning, higher sample efficiency
    - Limited by slowest actor (straggler problem)
  - Asynchronous  
    - A3C, Ape-X approaches
    - Actors push trajectories independently
    - No waiting for stragglers
    - Can introduce policy staleness
- Key components
  - Actors: generate trajectories through environment rollouts. Run on CPU, often parallelized across many machines
  - Learners: update policy/value networks on GPU. Process batches of experience to improve agent performance
  - Buffer/queue
  - Parameter server: synchronizes policy weights back to actors
- Frameworks & tooling
  - Ray RLlib
  - IMPALA/SEED RL
  - Ape-X/Reverb
  - CleanRL + MPI
- Scaling challenges
  - Network bottlenecks
  - Straggler problem
  - Memory pressure
  - Fault tolerance
- Optimization strategies
  - Vectorized environments
  - Prioritized experience replay
  - Asynchronous updates
  - Gradient compression
- Infrastructure requirements
  - CPUs
  - GPUs
  - Networking
  - Orchestration
- Best practices
  - Start small, then scale
  - Use elastic training
  - Frequent checkpointing
  - Separate logging

### 209. 207. Real-Time Serving of RL Agents
- From training pipelines to low-latency action loops
- Why real-time serving matters
  - Speed requirements: agents must act in milliseconds
  - Consequences: latency spikes lead to unsafe actions or missed opportunities
  - Infrastructure need: serving systems must ensure deterministic, low-latency inference
- Latency challenges
  - Computational constraints
  - Data-intensive preprocessing
  - Network bottlenecks
  - Jitter
- Optimization techniques
  - Model optimizations
    - Quantization
    - ONNX/TensorRT export
  - Deployment optimizations
    - Batch observations when safe
    - Use GPU pinning or dedicated edge accelerators
- Infrastructure patterns
  - On-device inference
    - Lowest latency (<5ms)
    - Ideal for critical control loops
  - Edge servers
    - Near real-time (5-20ms)
    - Good balance of power and latency
  - Cloud serving
    - Higher latency (20-100ms)
    - Best for non-critical tasks
- Monitoring and safety
  - Critical metrics
    - Performance metrics: inference latency, action frequency, real-time reward proxies
    - Safety indicators: Action saturation detection, environment drift, OOD observation detection
- Use cases
  - Robotics
  - Finance: algorithmic trading
  - Gaming/simulation
  - Industrial ops
- Best practices
  - Co-locate agents with environment
  - Profile end-to-end latency
  - Implement fallback policies
  - Log continuously for retraining
  
### 210. 208. Scaling RL for Robotics
- Why robotics needs scale
  - Complex state-action spaces
  - Unavoidable noisein sensors and actuation
  - Physical constraints that must be respected
- The simulation-to-real gap
  - Simulation ben
    - Fast iteration
    - Parallel environment
    - Perfect state information
  - The gap
    - Dynamic simplification
    - Contact physics inaccuracies
    - Sensor noise models
  - Real world complexity
    - Unpredictable friction
    - Material properties
    - Environmental variability
- Bridging the gap: domain randomization
- Scalable training pipelines
  - Massively parallel simulation
    - Isaac Gym: Parallel environments over GPU
    - Brax: JAX-accelerated physics
    - MuJoCo: CPU-based, but highly optimized
  - Distributed architecture
    - Actor-critic separation across compute nodes
    - Asynchronous policy updates
    - Curriculum learning adapts task difficulty
- HW infrastructure
  - Computation
    - GPU
    - TPU pods
    - High-memory instances for replay
  - Simulation
    - GPU-accelerated physics
    - Isaac Gym
    - Multi-GPU rendering
  - Edge deployment
    - Nviida Jetson AGX Orin
    - Edge TPU
  - Infrastructure    
    - Infiniband/RoCE for inter-node communication
    - Specialized job schedulers
    - Checkpoint management
- Policy optimization techniques
  - Algorithm selection: SAC, TD3, PPO, IMPALA
- Deployment challenges
  - Real-time constraints: inference times under 1-10ms
  - Safety guarantees
  - Fault recovery
  - Generalization
- Serving in robotics
  - Model optimization pipeline
    - Training model
    - Optimization
    - Deployment
  - Runtime safety systems
    - Multi-tiered control
    - Confidence-based switching b/w controllers
    - Continuous monitoring for policy drift
- Example applications
  - Quadruped locomotion
  - Industrial manipulation
  - Aerial navigation: agile drone
  - Autonomous vehicles
- Best practices
  - Development workflow
    - Simulation development
    - Domain randomization
    - Controlled testing
    - Limited field trials
    - Continuous monitoring
  - Technical recommendations
    - Log everything
    - Implement asymmetric actor-critic for privileged information in simulations
    - use both conservative and exploratory policy variants
    - Design for policy distillation to keep deployment models lightweight

### 211. 209. Infrastructure for Online Learning Agents
- What is online learning?
  - Continuous updates: agents update models at new data arrival
  - Evolving environment
  - Dual optimization
- Key characteristics of online learning systems
  - Streaming data ingestion
  - Incremental model updates
  - Dynamic exploration-exploitation
  - Catastrophic forgetting mitigation
- Core infrastructure components
  - Data stream layer: Kafka, Pulsar, MQTT
  - Experience replay buffer
  - Online learner
  - Serving engine
  - Monitoring & drift detection  
- Training strategies for online learning
  - Mini-batch updates
  - Adaptive optimizers
  - Hybrid training approaches
  - Parameter-efficient techniques
- Infrastructure challenges
  - Latency constraints
  - Horizontal scalability
  - Model stability
  - Resource isolation
- Optimization approaches
  - Asynchronous updates
  - Gradient optimization
  - Resource management
- Monitoring & safety systems
  - Oneline metrics dashboard
  - Concept drift detection
  - Safety mechanism
    - Shadow deployment of updated policies
    - Automatic rollback
    - Bounded exploration
- Realworld use cases
  - Financial trading
  - Intelligent transportation
  - Personalization systems
  - Adaptive robotics
- Best practices for production deployment
  - Hybrid training pipeline
  - Safety infrastructure
  - Robust memory management
  - DevOps Integration

### 212. 210. Lab – Train RL Agent with Ray RLlib
- Learning Goals
  - Install and configure Ray + RLlib
  - Train an RL agent on a standard Gym environment
  - Monitor training metrics and visualize results
  - Save and reload trained policies for inference
```
0) Prerequisites

    Python 3.9+

    Install dependencies:

    mkdir rllib-lab && cd rllib-lab
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install "ray[rllib]" gymnasium[classic_control] matplotlib

1) Quick Test: Hello RLlib

File: train_cartpole.py

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
     
    # Start Ray
    ray.init(ignore_reinit_error=True)
     
    # Configure PPO for CartPole
    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .rollouts(num_rollout_workers=1)
        .training(train_batch_size=4000)
        .framework("torch")
    )
     
    # Build Trainer
    algo = config.build()
     
    # Train for N iterations
    for i in range(5):
        result = algo.train()
        print(f"Iter: {i}, reward_mean: {result['episode_reward_mean']:.2f}")

Run:

    python train_cartpole.py

You should see the reward_mean increase as the agent learns.
2) Save & Load Policy

Extend train_cartpole.py:

    # Save checkpoint
    checkpoint = algo.save()
    print("Checkpoint saved at:", checkpoint)
     
    # Load checkpoint later
    algo.restore(checkpoint)

3) Run Inference with Trained Policy

File: inference.py

    import gymnasium as gym
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
     
    # Init Ray & load trained policy
    ray.init()
    config = PPOConfig().environment("CartPole-v1").framework("torch")
    algo = config.build()
    algo.restore("last_checkpoint_path")   # replace with actual checkpoint path
     
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
     
    done = False
    while not done:
        action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
     
    env.close()

Run:

    python inference.py

You’ll see the trained agent balancing the CartPole. 🎉
4) Visualize Training Rewards

    import matplotlib.pyplot as plt
     
    rewards = []
    for i in range(20):
        result = algo.train()
        rewards.append(result["episode_reward_mean"])
     
    plt.plot(rewards)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Episode Reward")
    plt.title("PPO Training on CartPole")
    plt.show()

5) Stretch Goals

    Swap environment to MountainCar-v0 or LunarLander-v2

    Try a different algorithm: DQN instead of PPO

    Run distributed training with multiple rollout workers:

        .rollouts(num_rollout_workers=4)

    Deploy trained policy as a FastAPI service for online inference

✅ Outcome: You trained and served an RL agent with Ray RLlib, monitored rewards, and deployed a checkpoint for inference.
```

## Section 32: Week 31: Large-scale Training - Basics

### 213. 211. What Is Large-Scale Training?
### 214. 212. Data Parallelism vs Model Parallelism Revisited
### 215. 213. Pipeline Parallelism in Transformers
### 216. 214. Distributed Optimizers and Gradient Sync
### 217. 215. Infrastructure Bottlenecks in Training LLMs
### 218. 216. Fault Tolerance in Multi-Node Training
### 219. 217. Lab – Train Transformer Across Multiple Nodes
2min

### 220. 218. DeepSpeed ZeRO-2/3 Optimizations
### 221. 219. Fully Sharded Data Parallel (FSDP)
### 222. 220. Flash Attention in Large Models
### 223. 221. Checkpointing Strategies for Multi-Node Systems
### 224. 222. Elastic Training with Kubernetes
### 225. 223. Scaling Beyond 1,000 GPUs
### 226. 224. Lab – Train with DeepSpeed ZeRO-3
1min

### 227. 225. What Is Enterprise MLOps?
### 228. 226. Introduction to Kubeflow
### 229. 227. Introduction to MLflow at Scale
### 230. 228. SageMaker Pipelines – Basics
### 231. 229. GCP Vertex AI Pipelines
### 232. 230. Azure ML Pipelines
### 233. 231. Lab – Build a Pipeline in Kubeflow
2min

### 234. 232. Model Registry in Enterprise MLOps
### 235. 233. Continuous Training (CT) Pipelines
### 236. 234. Automating Drift Retraining with Kubeflow
### 237. 235. Feature Store Integration in Pipelines
### 238. 236. Governance in Enterprise MLOps
### 239. 237. Audit Trails and Compliance Logging
### 240. 238. Lab – Automate Drift Retraining with Kubeflow
2min

### 241. 239. What Is Model Optimization for Infra Efficiency?
### 242. 240. Quantization Basics
### 243. 241. Pruning Basics
### 244. 242. Distillation Basics
### 245. 243. Structured vs Unstructured Sparsity
### 246. 244. Benchmarking Optimized Models
### 247. 245. Lab – Quantize a Vision Model
2min

### 248. 246. Mixed Precision Training with AMP
### 249. 247. Quantization-Aware Training (QAT)
### 250. 248. Advanced Distillation Techniques
### 251. 249. Sparse Training and Hardware Impacts
### 252. 250. Compiler Optimizations (XLA, TorchDynamo)
### 253. 251. Infra Tradeoffs: Accuracy vs Efficiency
### 254. 252. Lab – Train with Mixed Precision
2min

### 255. 253. What Is Federated Learning?
### 256. 254. Privacy-Preserving AI at Scale
### 257. 255. Federated Learning with TensorFlow Federated
### 258. 256. Secure Aggregation Protocols
### 259. 257. Federated Data Challenges
### 260. 258. Edge Deployment of Federated Models
### 261. 259. Lab – Train a Federated Model with TFF
2min

### 262. 260. Why Privacy Matters in AI Infra
### 263. 261. Homomorphic Encryption Basics
### 264. 262. Differential Privacy for AI Models
### 265. 263. Secure Multi-Party Computation (MPC)
### 266. 264. Tradeoffs in Privacy-Preserving AI
### 267. 265. Industry Applications of Privacy-Preserving AI
### 268. 266. Lab – Apply Differential Privacy in Training
2min

### 269. 267. Attacks on AI Infrastructure
### 270. 268. Model Poisoning Attacks
### 271. 269. Data Poisoning Attacks
### 272. 270. Membership Inference Attacks
### 273. 271. Adversarial Examples in Deployment
### 274. 272. Mitigation Strategies for Infra Security
### 275. 273. Lab – Defend Against Adversarial Attacks
2min

### 276. 274. What Is Multi-Tenancy?
### 277. 275. Resource Sharing Across Teams
### 278. 276. Cost Allocation for Multi-Tenant Infra
### 279. 277. Role-Based Access Control (RBAC)
### 280. 278. Isolation Strategies in AI Systems
### 281. 279. Monitoring Multi-Tenant Environments
### 282. 280. Lab – Configure Multi-Tenant Cluster
2min
