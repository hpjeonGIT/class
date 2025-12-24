## Strategies for Parallelizing LLMs Masterclass
- Instructor: Paulo Dichone | Software Engineer, AWS Cloud Practitioner & Instructor

## Section 1: Introduction

### 1. Introduction & What Is This Course About
- Strategies for Parallelizing LLMs
  - How to train massive LLMs effectively at scale
  - Data, model and pipeline parallelism techniques
  - Practical skills with PyTorch
  - Optimizaing compute, memory, and fault tolerance
  - Hands - building and parallelizing your own LLM

### 2. Course Structure
- Theory
- Hands-on

### 3. DEMO - What You'll Build in This Course

## Section 2: Course Source Code and Resources

### 4. Get Source Code
- https://github.com/pdichone/llm-parallelism

### 5. Get Course Slides

## Section 3: Strategies for Parallelizaing LLMs - Deep Dive

### 6. What is Parallelism and Why it Matters
- Parallelism
  - A computing strategy for dividing a large task into smaller subtasks
- Motivation behind parallelism
  - LLMMs are enormous!
  - GPT-3 has 175B parameters -> 45TB of text -> trillions of calculations -> sequential processing (single GPU) is impractical

### 7. Understanding the Single GPU Strategy

### 8. Understanding the Parallel Strategy and Advantages

### 9. Parallelism vs Single GPU - Summary

## Section 4: IT Fundamental Concepts

### 10. IT Fundamentals - Introduction

### 11. What is a Computer - CPU and RAM Overview

### 12. Data Storage and File Systems

### 13. OS File System Structure

### 14. LAN Introduction

### 15. What is the Internet

### 16. Internet Communication Deep Dive

### 17. Understanding Servers and Clients

### 18. GPUs - Overview

## Section 5: GPU Architecture for LLM Training Deep Dive

### 19. GPU Architecture for LLM Training

### 20. Why this Architecture Excels
1. Paralleism: thousands okf cores can simultaneously process different parts of the same matrix operation
2. Specialized HW: Tensor cores are specifically designed to accelerate the matrix operaionts
3. Memory bandwidth: fast data feeding
4. Memory Hierarchy

## Section 6: Deep and Machine Learning - Deep Dive

### 21. Machine and Deep Learning Introduction
- AI > ML > DL

### 22. Deep and Machine Learning - Overview and Breakdown
- Neural networks with multiple layers learn from data by:
  - Automatically discovering features
  - Learning hierarchical representations
  - Transforming raw input into desired output

### 23. Deep Learning Key Aspects
1. Automatic feature engineering
    - Raw data -> Deep network -> learned features/no manual engineering
2. Hierarchical structure
    - Simple features -> combined featuers -> complex features -> abstract concepts
3. End-to-end learning
    - Input -> deep neural network -> Output

### 24. Deep Neural Networks - Deep Dive
- Key components of neural networks
  - Neurons and Layers
    - Input layer: data receivers
    - Hidden layers: perform calculationto pass downstream
    - Output layer: produces the final result (prediction/classification)
  - Activation functions: decision-makers (is input important enough to pass forward?)
  - Training with backpropagation: feedback and improvement

### 25. The Single Neuron Computation - Deep Dive
- Key concepts
  - Weights = recipe proportions
  - Bias = base seasoning
  - Activation functions = quality controls

### 26. Weights
- Recipe proportions

### 27. Activation Functions - Deep Dive
- Activation function (quality control)
  - ReLU
    - Keep if good -> pass
    - If bad -> stop
  - Sigmoid: Rate 0-1
  - Softmax: Choose the best

### 28. Deep Learning - Summary

### 29. Machine Learning Introduction - ML vs DL
- DL: automatic feature learning
  - Large data sets
  - Black box model
- ML: manual feature engineering
  - Smaller data sets
  - Human interpretable

### 30. Learning Types and Full ML & DL Analogy Example
- Learning types
  - Supervised: learning the answers
  - Unsupervised: finding patterns
  - Reinforcement: learning from experience

### 31. DL and ML Comparative Capabilities - Summary

## Section 7: Large Language Models - Fundamentals of AI and LLMs

### 32. Introduction
- Transformer architecture - Overview

### 33. The Transformer Architecture Fundamentals
- Transformer architecture is a neural network best suited for text and NLP
- DL layer using self-attention mechanism
- Input text 
  - Tokenization
    - Embedding Layer 
      - Positional Encoding 
        - Transformer Blocks 
          - Output Layer
          - Next word prediction
- Input 
  - Self-attention 
    - Feed Forward 
      - Layer normalization 
        - Output
    - Q/K/V Attention
- Self-attention layer: decodes and looks at other words in the input so to help lead to a better encoding for the words

### 34. The Self-Attention Mechanism - Analogy
- Query + Key -> Attention scores + Value -> Weighted sum
  - Key: index or reference such as book title
  - Value: actual content
- Ex)
  - Word: cat (query) + All words (keys)
    - Attentaion calculations 
    - Word contents (values)
      - Final representation
- Attention weights
  - Higher weight to relevant words
  - Lower weight to less relevant

### 35. The Transformer Architecture Animation

### 36. The Transformer Library - Deep dive
- A comprehensive, modular toolkit for transformer models
  - Tokenization
  - Model configuration
  - High level pipelines
- Transformers library
  - Pipeline
    - Text generation
    - Classification
    - Question Answering
  - Configuraiton
  - Utilities


## Section 8: Parallel Computing Fundamentals & Parallelism in LLM Training

### 37. Parallel Computing Introduction - Key Concepts

### 38. Parallel Computing Fundamentals and Scaling Laws - Deep Dive
- Amdahl's law
  - Parallel parts (P)
  - N : number of processors
  - Speedup = 1 /(S + P/N)
  - Limited by sequential parts (S)  
- Gustafson's Law
  - Speedup = S + PxN

## Section 9: Types of Parallelism in LLM Training - Data - Model and Hybrid Paralleism

### 39. Types of Parallelism in LLM Training
- Essential parallel computing concepts for understanding LLM training
  - Types of parallelism in LLM training
    - Data parallelism
    - Model parallelism
    - Pipeline parallelism
    - Tensor parallelism
  - Common bottlenecks
  - Parallel programming patterns
  - Communication mokdels
  - Memory Architectures

### 40. Data Parallelism - How It Works
- Multiple devices (GPUs/TPUs) have a full copy of the model
- Different batches of data are processed simultaneously on each device
- Gradients are synchronized periodically to update all model copies
- 

### 41. Data Parallelism Advantages for LLM Training
- Throughput scaling: can scale linearly with the number of GPUs up to communication limits
- Dataset size handling: able to process trillions of tokens
- Memory efficiency: each GPU only stores one copy, and not the entire dataset
- Implementation simplicity

### 42. Real-world Example - Data Parallelism in GPT-3 Training

### 43. Model Parallelism and Tensor Parallelism and Layer Parallelism - Deep Dive
- Model parallelism
  - A technique where different parts of a neural network model are distributed across multiple computing devices
- Types of model parallelism in LLMs
  - Tensor parallelism (horizontal splitting)
    - Large attention matrix is splitted over multiple GPUs
    - Nvidia Megatron LLM
  - Layer parallelism (vertical splitting)
    - Activation flow b/w devices
    - Layers are sequential 

### 44. LLM Relevance and Implementaion
- Model parallelism is an absolute necessity for LLMs
  - Scale necessity: trillion parameters
  - Memory footprint: ~Terabytes required
  - Implementation challenge
    - Computational graph partitioning: determine optical cut points
    - Load balancing
    - Memory management: careful tracking of tensor lifetimes

### 45. Model vs Data Parallelism

### 46. Key Differences Highlighted - Data vs Model Parallelism

### 47. Data vs Model Parallelism

### 48. Hybrid Parallelism - Animation
- Hybrid Parallelism combines both model and data parallelism
  - Division of labor
  - Multidimensional scaling: model complexity and data throughput
  - Pipleline parallelism: multiple batches are processed simultaneously
  - Micro-batching: breaking large batches into small ones
- Motivation behind the hybrid parallelism
  - Overcoming size limitations: very large models (modern modls) that are too big for model parallelism alone
  - Improved HW utilization: reduce GPU idle time
  - Balancing communication costs: tradeoffs optimized b/w different types of communication overhead
  - Scalability
  - Memory efficiency

### 49. Hybrid Parallelism - What is It and Motivation

## Section 10: Types of Parallelism - Pipeline and Tensor Parallelism

### 50. Pipeline Parallelism Overview
- A technique where different parts of a neural network model are distributed across multiple computing devices
  - Each device to be responsible for different portions of the model's architecture
  - Model parallelism with many of mini-batches

### 51. Pipeline Parallelism Key Concepts and How it Works - Step by Step
- Key concepts
  - Micro-batching: each mini-batch is divided into smaller microbatches, for efficiency optimization
- How pipeline parallelism works
    - Different mini batches can be processed simultaneously at different stages of the model, creating a pipeline similar to an assembly line
    1. Model paritioning
    2. Mini-batch processing begins
    3. Pipeline starts building
    4. Pipeline Filling
    5. Full pipeline
    6. Steady state: all GPUs active, maximum efficiency is achieved
    7. Pipeline draining (cool-down)

### 52. Pipeline Bubbles Key Concepts
- Pipeline bubbles: idle periods when devices are waiting for inputs
  - At the start. Warm-up phase
  - At the end. Cool-down phase
  - They are unavoidable

### 53. Pipeline Schedules Key Concepts
- Pipeline schedules: different scheduling strategies exist to manage flow of micro-batches
  - GPipe schedule
    - All forward passes first, then all backwards passes
    - Require storing all activations
    - Lower bubble overhead but higher memory usage
  - 1F1B schedule (one-forward-one-backward)
    - Alternate b/w forward and backward passes
    - Reduces memory management
    - Ex: F1-> F2 -> B1 -> F3 -> B2 -> F4 -> B3 -> B4

### 54. Activation Recomputation - Overview and Introduction
- Activation recomputation: to manage memory constraints
  
### 55. Neural Network and Activation and Backward and Forward Passes - Full Dive
- Neural Network (AI)
  - What is an activation?
    - Refers to the output value produced by each layer of a neural network during the forward passing  
  - Forward and backward pass
    - Forward pass
      - Inputs flow through the network layer by layer
      - Each layer produces activations that feed into the next layer
      - The final layer produces the output prediction
    - Backward pass
      - Backpropagation
      - Error calculation happens after comparing the prediciton with the target
      - This error is propagted backwards through the network to update the weights

### 56. Understanding Activation Recomputation vs Standard Training - Deep Dive
- Memory challenge
  - As neural networks grow deeper, the memory required to store activations increase dramatically!
- Gradient Checkpointing
  - A.k.a activation recomputation
  - Storing only a subset of activations (checkpoints) during forward pass
  - Recomputing the missing activations when needed during the backward pass

### 57. Demo - Activation Recomputation Visualization

### 58. Activation Recomputation vs Standard Approach

### 59. Benefits of Activation Recomputation and Implementation Strategies
- At GPT-4 scale model, 67% memory reduction is achieved
- Backward pass becomes slightly slower
- Can train larger models
- Implementation strategies
    1. Optimal checkpoint selection
        - Mathematically determining optimal checkpoint positions minimizes both memory usage and recomputation overhead
    2. Hierarchical checkpointing
        - Uses multiple tiers of checkpoints with different storage/recomputation tradeoffs for maximum efficiency
    3. Activation compression
        - 32bit floating point -> 16bit or 8bit precision
        - Compression techniques: mixed precision(FP16/BF16), quantization (8/4bit), sparsification (prune small values)

### 60. Pipeline Parallelism Implementation Frameworks and Key Takeaways
- Implementations
  - GPipe (Google)
  - PipeDream (MS)
  - Megatron-LM (Nvidia)
  - DeepSpeed (MS)
- Challenges
  - Optimal layer partitioning: finding the ideal division of layers across devices
  - Balancing communication vs computation
  - Pipeline bubbles: minimize idle time
  - Memory management: balancing aviation storage vs recomputation
- Combining with other parallelism techniques - Pipeline parallelism is often combined with:
  - Data parallelism
  - Tensor parallelism: split individual operations across devices
  - Zero redundancy optimizer (ZeRO): partition otimizer stawtes, gradients and parameters

## Section 11. Tensor Parallelism - Deep Dive

### 61. What is Tensor Parallelism and Why - Benefits
- A distributed computing strategy where individual tensors are split across multiple devices, allowing parallel computation on different portions of the same tensor
- Tensor parallelism breaks individual operation into smaller chuncks that can be executed **independently**
- Why tensor parallelism?
  - Scale of modern models
    - Modeln LLM have billions of parameters
  - Memory bottleneck
  - Compute intensity: matrix multiplication is expensive
  - 

### 62. Tensor Parallel Pizza Making Analogy

### 63. Tensors and Partitioning Strategies - Deep Dive
- Tensors
- Partitioning strategy
  - Row-wise
  - Column-wise
  - 1D/2D/3D partitioning
- Communication primitives
- Device mesh

### 64. Tensor Communication Patterns - Deep Dive
- All-reduce
- All-Gather
- P2P
- Bandwidth: how much data can be transferred at once
- Latency: how long it takes to start the transfer

### 65. Device Mesh Communication Pattern - Deep Dive
- 1D/2D device mesh
- Torus mesh (wrapped 2D)
- Tree mesh

### 66. How Components Work Together in Distributed LLM Training

### 67. Understanding Tensor Parallelism with LEGO Bricks Animation Demo

### 68. Putting it All Together - All Strategies in LLM Training

## Section 12: HANDS-ON: Strategies for Parallelism - Data Parallelism Deep Dive

### 69. Strategies for Parallelizing LLMs - Hands- on Introduction

### 70. Pytorch - LLM Training Library Overview

### 71. The Transformers Library - Overview
- HuggingFace transformers library

### 72. Numpy Overview

### 73. TorchVision and TorchDistributed Overview
- torch.distributed
  - Helps multiple devices work together
  - Model parallelism
  - Pipeline parallelism
  - Tensor parallelism
  - Multi-node training

### 74. DeepSpeed and Megatron-LM - Overview
- Deepspeed
  - A library (build on top of PyTorch) that makes training large models (like LLMs) faster and memory-efficient
  - Memory efficiency: ZeRO (Zero Redundancy Optimizer)
  - Pipeline parallelism
  - Tensor parallelism
  - Scalability
  - Easy of use
- Megatron-LM
  - A library by Nvidia for training LLMs efficiently
  - Tensor parallelism
  - Pipeline parallelism
  - Optimized performance: highly optimized for Nvidia GPU - mixed precision, faster computation with less memory
  - Real-world use: GPTs

### 75. Datasets and Why this Toolkit
- PyTorch
- Transformers
- Numpy
- Torchvision
- Torch.distributed
- DeepSeed
- Megatron-LM

### 76. HANDS-On: Data Parallelism - Training a Small Model - MNIST Dataset
- Minimal working example (data parallelism)
```py
# install: pip install torch transformers numpy torchvision datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Step 1: Create a simple transformer model
config = BertConfig(
    vocab_size=30522,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512,
)
model = BertModel(config)
# Modified TransformerClassifier with a projection layer
class TransformerClassifier(nn.Module):
    def __init__(self, bert_model):
        super(TransformerClassifier, self).__init__()
        self.bert = bert_model
        # Project the input to match BERT's hidden_size
        self.projection = nn.Linear(1, 128)  # 1 → 128 (hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(128, 10)  # 128 → 10 (MNIST classes)
    def forward(self, x):
        # x shape: (batch_size, 28*28) for MNIST flattened
        x = x.view(-1, 28, 28)  # Reshape to (batch_size, 28, 28)
        x = x.mean(dim=2)  # (batch_size, 28)
        x = x.unsqueeze(-1)  # (batch_size, 28, 1)
        x = self.projection(x)  # (batch_size, 28, 128)
        outputs = self.bert(inputs_embeds=x)  # (batch_size, 28, 128)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 128)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, 10)
        return logits
model = TransformerClassifier(model).to(device)
# Step 2: Enable Data Parallelism
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
# Step 3: Load MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Step 4: Set up optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# Step 5: Training loop
try:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # (batch_size, 28*28)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
except Exception as e:
    print(f"An error occurred during training: {e}")
else:
    print("Training complete for 1 epoch!")
# Step 6: Evaluate on test set
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# Overall, this code evaluates the model's performance on the test dataset by 
# calculating the total number of correct predictions and the total number of samples, 
# which can later be used to compute the accuracy of the model.
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
# Step 7: Save the model
torch.save(model.state_dict(), "model_data_parallel.pth")
print("Model saved!")
```

### 77. Testing Pseudo Data Parallelism Trained Model
```py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Step 1: Define the model architecture (must match the trained model)
config = BertConfig(
    vocab_size=30522,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512,
)
base_model = BertModel(config)
class TransformerClassifier(nn.Module):
    def __init__(self, bert_model):
        super(TransformerClassifier, self).__init__()
        self.bert = bert_model
        self.projection = nn.Linear(1, 128)  # Matches the trained model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(128, 10)  # 128 → 10 (MNIST classes)

    def forward(self, x):
        x = x.view(-1, 28, 28)  # (batch_size, 28, 28)
        x = x.mean(dim=2)  # (batch_size, 28)
        x = x.unsqueeze(-1)  # (batch_size, 28, 1)
        x = self.projection(x)  # (batch_size, 28, 128)
        outputs = self.bert(inputs_embeds=x)  # (batch_size, 28, 128)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 128)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, 10)
        return logits
model = TransformerClassifier(base_model).to(device)
# Step 2: Load the saved weights
# Check if the model was saved with DataParallel (module prefix)
state_dict = torch.load("model_data_parallel.pth")
if "module." in list(state_dict.keys())[0]:
    # Remove 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
print("Model weights loaded successfully!")
# Step 3: Load the test dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# Step 4: Evaluate the model
model.eval()  # Set to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # Disable gradient computation for efficiency
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # (batch_size, 28*28)
        output = model(data)
        _, predicted = torch.max(output.data, 1)  # Get the predicted class
        total += target.size(0)
        correct += (predicted == target).sum().item()
accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
# Step 5: Manually test the model with a few numbers
def test_single_image(image, model):
    model.eval()
    with torch.no_grad():
        image = (
            transform(image).unsqueeze(0).to(device)
        )  # Add batch dimension and move to device
        image = image.view(image.size(0), -1)  # Flatten the image
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()
# Load a few sample images from the test dataset
sample_indices = [0, 1, 2, 3, 4, 78, 6, 9, 132, 7]  # Indices of the images to test
for idx in sample_indices:
    image, label = test_dataset[idx]
    # Convert the tensor image to a PIL Image
    image_pil = transforms.ToPILImage()(image)
    predicted_label = test_single_image(image_pil, model)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"True Label: {label}, Predicted: {predicted_label}")
    plt.show()
```

### 78. HANDS-ON: Data Parallelism - Colab - Full Demo
- Mimicking multi-gpus using 1 gpu only
- mpirun is not needed. Use `nn.DataParallel()`
```py
    # If trained with DataParallel, remove 'module.' prefix
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
...    
# Step 2: Enable Data Parallelism if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
```  

### 79. Data Parallelism - Simulated Parallelism on GPU Takeaways

## Section 13: HANDS-ON: Data parallelism w/ WikiText Dataset & DeepSeed Mem. Optimization

### 80. Hands-on: Data Parallelism - Wikitext-2 Dataset
- WikiText-2: long-term dependency test
```py
# Function to save the trained model to a file
def save_model(model, path):
    # If using multiple GPUs, save the core model's parameters
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)  # Save the model's parameters
    print(f"Model saved to {path}")
...
# Use multiple GPUs if available (faster training)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # Split work across GPUs
```  

### 81. DeepSpeed - Full Dive
- An open-source deep learning optimization library by MS - enhances training efficiency for large-scale models
- Provides tools for:
    1. Model parallelism including pipeline and tensor parallelism
    2. Memory optimization: ZeRO (Zero Redundancy Optimizer) for efficient memory usage
    3. Mixed precision training: FP16 support for faster computation
    4. Scalability: support distributed training across multiple GPUs/nodes

### 82. Hands-on: Data Parallelism with DeepSpeed Optimization
- pip install mpi4py deepspeed
```py
# DeepSpeed configuration dictionary
# This defines how DeepSpeed will optimize the training process
ds_config = {
    # How many examples to process in one batch during training
    "train_batch_size": 8,
    # Number of gradient accumulation steps (1 means update after every batch)
    "gradient_accumulation_steps": 1,
    # Enable mixed precision training with FP16 (16-bit floating point)
    # This saves memory and can speed up training
    "fp16": {"enabled": True},
    # Optimizer configuration
    "optimizer": {
        "type": "Adam",  # Using Adam optimizer
        "params": {
            "lr": 5e-5,  # Learning rate
            "betas": [0.9, 0.999],  # Adam beta parameters
            "eps": 1e-8  # Small constant for numerical stability
        }
    },
    # ZeRO (Zero Redundancy Optimizer) settings for memory optimization
    "zero_optimization": {
        # ZeRO stage 2 partitions optimizer states across GPUs
        "stage": 2,

        # Offload optimizer states to CPU to save GPU memory
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
# Save DeepSpeed config to a JSON file that DeepSpeed will read
with open("ds_config.json", "w") as f:
    json.dump(ds_config, f)
...    
# Create a class to hold the DeepSpeed arguments
# DeepSpeed requires these arguments to initialize properly
class Args:
    def __init__(self):
        # Set to -1 for automatic local rank detection
        # (local_rank refers to the GPU ID in multi-GPU setups)
        self.local_rank = -1

        # Enable DeepSpeed
        self.deepspeed = True

        # Path to the configuration file
        self.deepspeed_config = "ds_config.json"

        # Use ZeRO stage 2 optimization
        self.zero_stage = 2

        # Enable mixed precision training
        self.fp16 = True
# Create an instance of Args to pass to DeepSpeed
args = Args()
# Initialize DeepSpeed with our model and arguments
# This wraps the model for optimized training
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,  # The model to optimize
    model_parameters=model.parameters(),  # Model parameters to train
    args=args  # The arguments we defined above
)
```

## Section 14: Running TRUE Parallelism on Multiple GPU Systems - Runpod.io

### 83. Setup Runpod.io Environment Overview

### 84. Runpod SSH Setup

### 85. Setting up Runpod Parallelism in JupyterNotebook

### 86. HANDS-ON - Parallelism with IMDB Dataset - Deep Dive - True Parallelism

### 87. Runpod Cleanup

## Section 15: Fault Tolerance and Scalability & Advanced Checkpointing Strategies - Deep Dive

### 88. Fault Tolerance Introduction & Types of Failures in Distributed LLM Training
- Training challenges for Modern LLMs
  - Scale
    - Training on 1,000+ GPUs
    - Weeks to months of runtime
    - Petaflops of computation
  - Inevitable failures
    - HW failure
    - System crash
    - SW error and deadlocks
    - Power or cooling issues
  - Economic impact
    - Training costs of $1-10M+
    - Every failure is extremely costly
    - Restarting from scratch is infeasible
- Types of failures in distributed LLM training
  - HW failure
    - GPU failure
      - Memory error
      - Compute unit failure
      - Overheating
    - Node failure
      - CPU failure
      - Storage issues
  - System failures
    - Power issues
      - Outages
      - Power supply failure
    - Cooling failures
       - Data center cooling
       - Node level cooling
  - Network failure
    - Interconnect issues
      - NVLink failures
      - Infiniband errors
      - Network congestion
    - Communication timeouts
      - Synchronization failures
      - Deadlock
  - Framework bugs
    - Deadlocks
    - Memory leaks
    - Numerical instabilities
  - Resource exhaustion
    - Out-of-memory errors
    - Process crashes

### 89. Strategies for Fault Tolerance
- Checking point strategies for fault tolerance
  - Basic checkpointing: full model snapshots
  - Incremental checkpointing: delta-based checkpoints
  - Asynchronous checkpointing: background checkpointing
  - Multi-level checkpointing: tiered checkpoint strategy
- Fault tolerance benefits
  - Economic benefits
    - Save millions in training costs
    - Minimize wasted computation
    - Reduce training time by avoiding restarts
  - Technical benefits
    - Resilience to inevitable HW failures
    - Distributed checkpointing minimizes overhead
    - Enables training at massive scale

### 90. Checkpointing in LLM Training - Animation
- What gets saved in a checkpoint:
  - Model weights: trained parameters
  - Optimizer states: critical learning rate information
  - Training metadata: current step count, schedule rate, loss, ...
  - Data iterator state: position in the training dataset to resume from the same point
  - Random states: RNG seeds (improve learning)
- Checkpointing frequency considerations
  - Too frequent: excessive IO - can slow down training (5-30% overhead)
  - Too infrequent: risks losing more computatino time if failure occurs
  - Adaptive strategies: modern systems adjust checkpoint frequency based on:
    - Historical failure rates of the HW
    - Value of computation
    - Current training stability

### 91. Basic Checkpointing in LLM Taining

### 92. Incremental Checkpointing in LLM Training

### 93. Asynchronous Checkpointing in LLM Training
- Uses separate threads/processes for IO 
- Minimizes training interruption

### 94. Multi-level Checkpointing in LLM Training - Animation

### 95. Checkpoint Storage Considerations - Deep Dive
- Right storage to choose
  - Reliability
  - Performance
  - Cost
- Checkpoint storage
  - Local: SSD, NVMe
  - Distributed: fault tolerant, parallel IO, redundancy
  - Cloud: object storage solutions

### 96. Implementing a Hybrid Approach - Performance, Failure, Optimizations - Full Dive
- Core components
  - Multi-tiered storage architecture
    - Tier 1: local storage
    - Tier 2: distributed storage
    - Tier 3: cloud storage
  - Intelligent routing and polices
  - Synchronization mechanisms  
    - Write-through caching
    - Asynchronous replication
    - Delta synchronization
- Performance optimization
  - Parallel processing
  - Compression
  - Batching
  - Preemptive caching
- Failure handling
  - Tier failure detection
  - Automatic failover
  - Reconciliation
  - Degraded operation modes

### 97. Checkpoint Storage Strategy - Summary

## Section 16: Advanced Topics and Emerging Trends
- Several trends in LLM parallelization
  - Efficient communication protocols: combining data, model and pipeline parallelism for optimal resource use
  - Specialized HW integraitn
  - Framework advancements
  - Quantum computing exploration
  
### 98. Advanced Topics and Emerging Trends

## Section 17: Wrap up and Next Steps

### 99. Course Summary and Next Steps

