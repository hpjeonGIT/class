## PINNs Using Physics-Nemo [Modulus]
- Instructor: Dr.Mohammad Samara

## Section 1: Introduction

### 1. Introduction
- PhysicsNemo is renamed as Modulus

### 2. Course Structure

### 3. Deep Learning Theory
- Activation functions
  - Sigmoid: $1/(1+exp(-x))$
  - tahn: $tanh(x)$
  - ReLU: max(0,x)
  - Leaky ReLU: max(0.1x,x)
  - Maxout: max(w1^T x + b1, w2^Tx+b2)
  - ELU: x if x>=0 $$\alpha(exp(x)-1)$ otherwise
- Cost function (loss function)
  - If a model is performing well or not
  - Mean Square
  - Cross Entropy

### 4. PINNs(Physics-Informed Neural Networks) Theory
- Use NN to solve PDE
- Two types of losses
  - BC/IC
  - Domain points
- Steps for solving PINNs 
- Define NN
- Define the IC/BC
- Set the optimizers
- Define the loss functions
- Run the training loop
- Results post-processing


## Section 2: PINNs Solution for 1D Burgers Equation with PyTorch

### 5. Define the Neural Network
- ${\partial u \over \partial t} + u {\partial u \over \partial x} = \nu {\partial^2 u \over \partial x^2}$
```py
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
class NN(nn.Module):
  def __init__(self):
    super(NN,self).__init__()
    self.net = torch.nn.Sequential(
      nn.Linear(2,20),
      nn.Tanh(),
      nn.Linear(20,30),
      nn.Tanh(),
      nn.Linear(30,30),
      nn.Tanh(),
      nn.Linear(30,20),
      nn.Tanh(),
      nn.Linear(20,20),
      nn.Tanh(),
      nn.Linear(20,10)      
    )
  def forward(self,x):
    out=self.net()
    return out
```

### 6. Initial Conditions and Boundary Conditions
```py
class Net:
  def __init__(self):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # comp. domain
    self.h = 0.1
    self.k = 0.1
    x = torch.arange(-1,1+self.h,self.h)       
    t = torch.arange(-1,1+self.k,self.k)
    self.X = torch.stack(torch.mechgrid(x,t)).reshape(2,-1).T
    # train data
    bc1 = torch.stack(torch.meshgrid(x[0],t)).reshape(2,-1).T
    bc2 = torch.stack(torch.meshgrid(x[-1],t)).reshape(2,-1).T
    ic  = torch.stack(torch.meshgrid(x,t[0])).reshape(2,-1).T
    self.X_train = torch.cat([bc1,bc2,ic])
    #
    y_bc1 = torch.zeros(len(bc1))
    y_bc2 = torch.zeros(len(bc2))
    y_ic  = -torch.sin(math.pi*ic[:,0])
    self.y_train = torch.cat([y_bc1,y_bc2, y_ic])
    self.y_train = self.y_train.unsqueeze(1)
```
- Regarding unsqueeze():
```py
y_train_test = torch.tensor([1,2,3,4])
y_train_test = y_train_test.unsqueeze(1)
print(y_train_test)
#tensor([[1],
#        [2],
#        [3],
#        [4]])
```

### 7. Optimizer

### 8. Loss Function

### 9. Train the Model

### 10. Results Evaluation

## 

### 11. What is the Wave Equation
### 12. Setting Up Google Colab
### 13. Define the Wave Equation Function
### 14. Define the Config File
### 15. Import Needed Libraries
### 16. Set Up the main RUN File
### 17. Define the B.C, Interior Points
### 18. Add Validator Functionality
### 19. Solve
### 20. Results Extraction
### 21. Results Post Processing
### 22. Nvidia-modulus To Physicsnemo

##

### 23. Setting up env. in your personal computer
### 24. Cavity Flow Problem
### 25. Define the Config File
### 26. Import Needed Libraries
### 27. Set Up the main RUN File
### 28. Define the Navier-Stokes equation and DNN
### 29. Define the B.C, I.C, Interior Points
### 30. Solve
### 31. Results Extraction
### 32. Results Post Processing
### 33. Pretrained Model Inference

##

### 34. 2d heat channel problem
### 35. Define the Config File
### 36. Import Needed Libraries
### 37. Set Up the main RUN File
### 38. Define the geometry
### 39. Define the Navier-Stokes equation and DNN
### 40. Define the B.C, Interior Constraints
### 41. Add Monitor
### 42. Solve
### 43. Results Extraction
### 44. Results Post Processing

##

### 45. 2D Stress Analysis Problem
### 46. Define the Config File
### 47. Import Needed Libraries
### 48. Define the DNN
### 49. Define the geometry - part a
### 50. Define the geometry - part b
### 51. Set Changing Parameters
### 52. Define the B.C, Interior Constraints
### 53. Case Inferencing
### 54. Solve
### 55. Results Post Processing

