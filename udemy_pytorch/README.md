## PyTorch for Deep Learning Bootcamp
- Instructor:
  - Andrei Neagoie
  - Daniel Bourke

## Section 1: Introduction

### 1. PyTorch for Deep Learning

### 2. Course Welcome and What Is Deep Learning

### 3. Join Our Online Classroom!

### 4. Exercise: Meet Your Classmates + Instructor

### 5. Free Course Book + Code Resources + Asking Questions + Getting Help
- https://github.com/mrdbourke/pytorch-deep-learning

### 6. ZTM Resources

### 7. Machine Learning + Python Monthly Newsletters

## Section 2: PyTorch Fundamentals
- https://github.com/mrdbourke/pytorch-deep-learning/blob/main/video_notebooks/01_pytorch_workflow_video.ipynb

### 8. Why Use Machine Learning or Deep Learning

### 9. The Number 1 Rule of Machine Learning and What Is Deep Learning Good For
- What deep learning is good for
  - Problems with long lists of rules
  - Continually changing environments
  - Discovering insights within large collections of data
- What deep learning is not good for
  - When you need explainability
  - When the traditional approach is a better option
  - When errors are unacceptable
  - When you don't have much data

### 10. Machine Learning vs. Deep Learning
- ML
  - For Structured data
- DL
  - For unstructured data

### 11. Anatomy of Neural Networks
- Input layler
- Hidden layer
- Output layer
- Each layer is usually combination of linear and/or nonlinear functions

### 12. Different Types of Learning Paradigms
- Supervised learning
- Unsupervised & self-supervised learning
- Transfer learning

### 13. What Can Deep Learning Be Used For
- Recommendation
- Translation: seq2seq
- Speech recognition: seq2seq
- Computer vision: classification
- Natural Language Processing: classification/regression

### 14. What Is and Why PyTorch
- The most popular research deep learning framework
- Write fast DL code in Python
- Able to access many pre-built DL models
- Whole stack: preprocess data, model data, deploy model in application/cloud
 
### 15. What Are Tensors

### 16. What We Are Going To Cover With PyTorch

### 17. How To and How Not To Approach This Course
1. Code along
2. Explore and experiment
3. Visualize what you don't understand
4. Ask questions
5. Do the exercises
6. Share your work

### 18. Important Resources For This Course
- https://github.com/mrdbourke/pytorch-deep-learning
- https://github.com/mrdbourke/pytorch-deep-learning/discussions
- https://www.learnpytorch.io/

### 19. Getting Setup to Write PyTorch Code
```py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
print(torch.__version__)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```
### 20. Introduction to PyTorch Tensors
```py
#scalar - as lower
scalar = torch.tensor(7)
scalar# Scalar
scalar = torch.tensor(7)
print(scalar)
scalar.ndim # 0, not 1
scalar.item() # 7
scalar.shape # torch.Size([])
# Vector - as lower
vector = torch.tensor([7,7])
vector.ndim # 1, not 2
vector.shape # torch.Size([2])
# Matrix - as upper or Capital
MATRIX = torch.tensor ([[7,8],[9,10]])
MATRIX.ndim # 2
MATRIX[1] # prints tensor([9,10])
MATRIX[0] # prints tensor([7,8])
MATRIX.shape # torch.Size([2,2])
# Tensor - as upper or Capital
TENSOR = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
TENSOR.ndim # 3
TENSOR.shape # torch.Size([1,3,3])
```

### 21. Creating Random Tensors in PyTorch
- Why random tensors?
  - Many neural network iniitates from random numbers
```py
random_tensor = torch.rand(3,4)
random_tensor.shape
rt = torch.rand(size=(2,3,4))
print(rt.ndim, rt.shape) # 3 torch.Size([2, 3, 4])
```

### 22. Creating Tensors With Zeros and Ones in PyTorch
```py
RT = torch.ones(2,3) # all 1.0's
RT = torch.zeros(2,3) # all 0.0's
RT.dtype # torch.float32
```

### 23. Creating a Tensor Range and Tensors Like Other Tensors
```py
torch.range(0,3) # returns a tensor of 0,1,2,3 - 3 included
torch.arange(0,3) # returns a tensor of 0,1,2 - 3 not included
torch.arange(start=123, end=999, step=256) # tensor([123, 379, 635, 891])
one_to_ten = torch.range(0,10)
ten_zeros = torch.zeros_like(input=one_to_ten) # zeros_like find the size of tensors using the input
```
### 24. Dealing With Tensor Data Types
```py
float32_tensor = torch.tensor([3.,6,9], dtype=torch.float16)
print(float32_tensor.dtype) # torch.float16
float32_tensor = torch.tensor([3.,6,9], dtype=None,  # float32 or float16 or int32
                              device="cuda", # "cpu" or "cuda" or "tpu"
                              requires_grad=False # will track gradient or not
                              )
print(float32_tensor.dtype) # torch.float32
```
- TEnsor datatype is one of the 3 big errors that you will run into with PyTorch and DL
  1. Tensors are not right datatype
  2. Tensors not right shape
  3. Tensors not on the right device
- When CUDA is not working well with PyTorch:
```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```
- Type conversion:
```py
tmp_tensor = float32_tensor.type(torch.float16)
print(tmp_tensor) # tensor([3., 6., 9.], device='cuda:0', dtype=torch.float16)
```

### 25. Getting Tensor Attributes
- Datatype: x.dtype
- Shape: x.shape
- Device: x.device

### 26. Manipulating Tensors (Tensor Operations)
```py
tns = torch.tensor([1,2,3])
tns + 100 # 101,102,103
tns * 100 # 100,200,300
# == torch.mul(tns,100)
```

### 27. Matrix Multiplication (Part 1)
```py
%%time # measures wall time in the jupyter
A = torch.tensor([1,2,3])
A*A # tensor([1,4,9])
torch.matmul(A,A) # 14
```
- Use torch's matrix operation instead of loop-wise method
- **Note that when A is a vector (1dim), not matrix, torch.matmul() works as a dot product**

### 28. Matrix Multiplication (Part 2): The Two Main Rules of Matrix Multiplication
- `A@B` is a syntactic surgar of `torch.matmul(A,B)`

### 29. Matrix Multiplication (Part 3): Dealing With Tensor Shape Errors
- `torch.mm(A,B)` is a syntactic surgar of `torch.matmul(A,B)`
```py
A = torch.tensor([[1,2,3],[4,5,6]])
A*A # tensor([[ 1,  4,  9], [16, 25, 36]])
torch.mm(A,A.T) # tensor([[14, 32], [32, 77]])
```

### 30. Finding the Min Max Mean and Sum of Tensors (Tensor Aggregation)
- Tensor aggregation: min, max, sum, etc
```py
A.min() # tensor(1)
A.max() # tensor(6)
A.sum() # tensor(21)
#A.mean() # not working fot int
B = A.type(torch.float32) # conversion into float32
B.mean() # tensor(3.5000)
```

### 31. Finding The Positional Min and Max of Tensors
```py
A.argmin() # tensor(0)
A.argmax() # tensor(5)
```

### 32. Reshaping, Viewing and Stacking Tensors
- Reshaping: reshapes an input tensor to a defined shape
- View: returns a view of an input tensor of certain shape but keep the same as the original tensor
- Stacking: combines multiple tensors on top of each other (vstack) or a side (hstack)
- Squeeze: removes all '1' dimensions from a tensor
- Unsqueeze - adda a '1' dimension to a target tensor
- Permute: returns a view of the input with dimensions permuted (swapped) in a certain way
```py
x = torch.arange(1., 10.)
x, x.shape # (tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]), torch.Size([9]))
x_re = x.reshape(3,3)
x_re, x_re.shape
z = x.view(3,3)
z, z.shape # (tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]), torch.Size([3, 3]))
x_st = torch.stack([x,x], dim=0)
x_st # tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.],        [1., 2., 3., 4., 5., 6., 7., 8., 9.]])
x_st = torch.stack([x,x], dim=1)
x_st # tensor([[1., 1.],[2., 2.],[3., 3.],[4., 4.],[5., 5.],[6., 6.],[7., 7.],[8., 8.],[9., 9.]])
```

### 33. Squeezing, Unsqueezing and Permuting Tensors
- squeeze: [1,9] -> [9]
- unqueeze: [9] -> [1,9] or [9,1]
```py
y = x.squeeze()
z = x.unsqueeze(dim=0)
x, x.shape, y, y.shape, z, z.shape
#(tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]), torch.Size([9]), tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]), torch.Size([9]), tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.]]), torch.Size([1, 9]))
x = torch.rand(size=(3,4,2))
x_p = x.permute(2,0,1)
x.shape, x_p.shape # (torch.Size([3, 4, 2]), torch.Size([2, 3, 4]))
```

### 34. Selecting Data From Tensors (Indexing)
```py
x = torch.rand(size=(3,4,2))
x_p = x.permute(2,0,1)
#x.shape, x_p.shape
print(x[0,0,0], x_p[0,0,0]) # tensor(0.7831) tensor(0.7831)
x[0,0,0]=123.
print(x[0,0,0], x_p[0,0,0]) # tensor(123.) tensor(123.)
```
- x.permute() provides a reference copy
- ":" to select all target dimension

### 35. PyTorch Tensors and NumPy
```py
x = torch.arange(1,10).reshape(1,3,3)
print(x[0,2,2], x[0][2][2]) # tensor(9) tensor(9)
print(x[0,2,:]) # tensor([7, 8, 9])
```
- Numpy data into PyTorch tensor
  - `torch.from_numpy(ndarray)`
  - numpy default datatype is float64 and default conversion will produce torch.float64
- PyTorch tensor to numpy data
  - `torch.Tensor.numpy()`
```py
# np -> torch
array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array)
array, tensor # (array([1., 2., 3., 4., 5., 6., 7.]), tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))
tensor32 = torch.from_numpy(array).type(torch.float32)
print(tensor32.dtype) # torch.float32
# Torch -> np
t = torch.ones(7)
np_t = t.numpy()
print(t.dtype, np_t.dtype) # torch.float32 float32
```
- Conversion uses deep copy

### 36. PyTorch Reproducibility (Taking the Random Out of Random)
- How neural network learns
  - random numbers -> tensor operations -> update random numbers to make them better representation of the data -> again -> again ...
  - random seed
```py
RANDOM_SEED=7
torch.manual_seed(RANDOM_SEED)
a = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED)
b = torch.rand(3,4)
a == b # all Trues
```

### 37. Different Ways of Accessing a GPU in PyTorch
1. Google Colab
2. Own GPU
3. Cloud
- Check for GPU access with PyTorch:
```py
import torch
torch.cuda.is_available()
# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.device_count() # Number of GPUs
```

### 38. Setting up Device-Agnostic Code and Putting Tensors On and Off the GPU
- CPU -> CUDA
  - Works as a deep copy
```py
t_cpu = torch.tensor([1,2,3])
print(t_cpu, t_cpu.device)
t_gpu = t_cpu.to("cuda")
print(t_gpu, t_gpu.device)
```
- CUDA -> CPU
```py
tmp_cpu = t_gpu.cpu().numpy()
print(t_gpu, tmp_cpu)
```

### 39. PyTorch Fundamentals: Exercises and Extra-Curriculum
- https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/00_pytorch_fundamentals_exercises.ipynb

## Section 3: PyTorch Workflow

### 40. Introduction and Where You Can Get Help

### 41. Getting Setup and What We Are Covering
- https://www.learnpytorch.io/01_pytorch_workflow/
- We are covering:
  1. Data (prepare and load)
  2. Build model
  3. Fitting the model to data
  4. Making predictions and evaluating a model (inference)
  5. Saving and loading a model
  6. Putting it all together
```py
import torch
from torch import nn
import matplotlib.pyplot as plt
torch.__version__
```

### 42. Creating a Simple Dataset Using the Linear Regression Formula
- Data can be almost anything
  - Excel spreadsheet
  - Images
  - Videos
  - Audio like songs or podcasts
  - DNA
  - Text
- ML is a game of two parts:
  1. Get data into a numerical representation
  2. Build a model to learn patterns in that numerical representation
```py
weight=0.7
bias = 0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight *X + bias
X[:5], y[:5] # (tensor([[0.0000],[0.0200],[0.0400],[0.0600],[0.0800]]), tensor([[0.3000],[0.3140],[0.3280],[0.3420],[0.3560]]))
```

### 43. Splitting Our Data Into Training and Test Sets
- Three datasets
  1. Training set, 60-80%
  2. Validation set, 10-20%
  3. Test set, 10-20%
```py
train_split = int(0.8*len(X))
X_train,y_train = X[:train_split], y[:train_split]
X_test,y_test = X[train_split:], y[train_split:]
len(X_train), len(y_train), len(X_test), len(y_test) # 40 40 10 10 
```

### 44. Building a function to Visualize Our Data
```py
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test,
                     predictions=None):
  plt.figure(figsize=(10,7))
  plt.scatter(train_data,train_labels, c="b",s=4,
              label="Trainig data")
  plt.scatter(test_data, test_labels, c="g", s=4,
              label="Testing data")
  if predictions is not None:
    plt.scatter(test_data, predictions, c="r", s=4,
                label="Predictions")
  plt.legend(prop={"size": 14});
```

### 45. Creating Our First PyTorch Model for Linear Regression
- Our model does:
  - Start with random values (weight and bias)
  - Look at training data and adjust the random values to better represent (or get closer to) the ideal values (the weight & bias values we used to create the data)
- How does it so?
  1. Gradient descent
  2. Backpropagation
```py
# Linear regression model
# weights and bias are random in the beginning but will be updated as learning progresses
from torch import nn
class LinearRegressionModel(nn.Module): # most of Pytorch classes are inherited from nn.Module
  def __init__(self):
    super().__init__()
    self.weights = \
    nn.Parameter(torch.randn(1,requires_grad=True,
                             dtype=torch.float))
    self.bias = nn.Parameter(torch.randn(1,requires_grad=True,
                                         dtype=torch.float))
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.weights * x + self.bias
```

### 46. Breaking Down What's Happening in Our PyTorch Linear regression Model

### 47. Discussing Some of the Most Important PyTorch Model Building Classes
- PyTorch model building essentials
  - torch.nn: contains all of buildings for computational grpahs (a neural netowrk can be considered as a computational graph)
  - torch.nn.Parameter: what parameters should our model try and learn, often a PyTorch layer from torch.nn will set theses for us
  - torch.nn.Module: the base class for all neural network modules, if you subclass it, you should override forward()
  - torch.optim: this where the optimizers in PyTorch live, they will help with gradient descent
  - def forward(): all nn.Module subclasses require you to override forward(), this method defines what happens in the forward computation

### 48. Checking Out the Internals of Our PyTorch Model
```py
torch.manual_seed(42)
model_0 = LinearRegressionModel()
list(model_0.parameters()) # [Parameter containing: tensor([0.3367], requires_grad=True), Parameter containing: tensor([0.1288], requires_grad=True)]
```

### 49. Making Predictions With Our Random Model Using Inference Mode
- Prediction using `torch.inference_mode()`
  - No-gradient tracking
  - Faster
```py
with torch.inference_mode():
  y_preds = model_0(X_test)
y_preds
```
  - This is equivalent to `y_preds=model_0(X_test)` with gradient tracking

### 50. Training a Model Intuition (The Things We Need)
- Train model
  - The whole idea of training is for a model to move from some unknown parameters (these might be random) to some known parameters
  - Or in other words, from a poor representation of the data toa better representation of the data
  - One way to measure how poor or how wrong your models predictions are is to use a loss function
  * Loss function may also be called **cost function** or criterion in different areas
- Loss function: A function to measure how wrong your model's predictions are to the ideal outputs, lower is better
- Optimizer: Takes into account the loss of a model and adjusts the model's parameters (weight and bias)

### 51. Settinhyg Up an Optimizer and a Loss Function
- nn.L1Loss(): absolute differences
```py
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)
```

### 52. PyTorch Training Loop Steps and Intuition
- Learning rate(lr): a hyper parameter that defines how big/small the optimizer changes the paramters with each step
- Things we need in a training loop
  0. Loop through the data
  1. Forward pass: this involves data moving through our model's `forward()` function to make predictions. Also called forward propagation
  2. Calculate the loss: compares forward pass predictions to ground truth labels  
  3. Optimizer zero grad
  4. Loss backward - move backwards through the network to calculate the gradients of each of the parameters of our model WRT the loss. Called as **backpropagation**
  6. Optimizer step: use the optimizer to adjust our model's parameters to try to improve the loss

### 53. Writing Code for a PyTorch Training Loop
```py
epochs = 10
# 0. Loop through the data
for epoch in range(epochs):
  model_0.train()
  # 1. Forward pass
  y_pred = model_0(X_train)
  # 2. Loss calculation
  loss = loss_fn(y_pred,y_train)
  # 3. Optimizer
  optimizer.zero_grad()
  # 4. Perform backpropagation
  loss.backward()
  # 5. STep the optimizer
  optimizer.step()
  #model_0.eval()
```

### 54. Reviewing the Steps in a Training Loop Step by Step

### 55. Running Our Training Loop Epoch by Epoch and Seeing What Happens

### 56. Writing Testing Loop Code and Discussing What's Happening Step by Step
```py
epochs = 10
# 0. Loop through the data
for epoch in range(epochs):
  model_0.train()
  # 1. Forward pass
  y_pred = model_0(X_train)
  # 2. Loss calculation
  loss = loss_fn(y_pred,y_train)
  print(f"Loss: {loss}")
  # 3. Optimizer
  optimizer.zero_grad()
  # 4. Perform backpropagation
  loss.backward()
  # 5. Step the optimizer
  optimizer.step()
  ## Testing
  model_0.eval() # turns off different settings in th emodel not needed for evaluation/testing (dropout/batchnorm layers). model_0.train() will activate them again
  with torch.inference_mode(): # with torch.no_grad(): # alternative but in old code
    test_pred = model_0(X_test) 
    test_loss = loss_fn(test_pred, y_test)
  if epoch%10 ==0:    
    print(f"Epoch: {epoch} | Test: {loss} | Test loss: {test_loss}")
  print(model_0.state_dict())
```

### 57. Reviewing What Happens in a Testing Loop Step by Step

### 58. Writing Code to Save a PyTorch Model
- Saving a model in PyTorch
  1. `torch.save()` allows you to save PyTorch object in Python's pickle format
  2. `torch.load()` allows you to load a saved PyTorch object
  3. `torch.nn.MOdule.load_state_dict()` allows you to load a model's saved state dictionary
```py
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = '01_pytorch_workflow_model_0.pth' # extension of pth for PyTorch object
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
MODEL_SAVE_PATH
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH) # Saving model to: models/01_pytorch_workflow_model_0.pth
```
### 59. Writing Code to Load a PyTorch Model
- To load ina saved state_dict we have to instantiate a new instance of our model class
```py
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_0.state_dict() # OrderedDict([('weights', tensor([0.6913])), ('bias', tensor([0.2958]))])
```

### 60. Setting Up to Practice Everything We Have Done Using Device Agnostic code
```py
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device= {device}")
```

### 61. Putting Everything Together (Part 1): Data
```py
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight* X  + bias
train_split = int(0.8*len(X))
X_train,y_train = X[:train_split], y[:train_split]
X_test,y_test = X[train_split:], y[train_split:]
print(len(X_train), len(y_train), len(X_test), len(y_test))
#
plot_predictions(X_train, y_train, X_test, y_test)
```

### 62. Putting Everything Together (Part 2): Building a Model
```py
class LinearRegressionModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    # Use nn.Linear() for creating the model parameters
    self.linear_layer = nn.Linear(in_features=1, out_features=1)
  def forward(self,x: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1, model_1.state_dict() # (LinearRegressionModelV2((linear_layer): Linear(in_features=1, out_features=1, bias=True) ), OrderedDict([('linear_layer.weight',tensor([[0.7645]])),('linear_layer.bias', tensor([0.8300]))]))
```

### 63. Putting Everything Together (Part 3): Training a Model
```py
next(model_1.parameters()).device # cpu
model_1.to(device)
next(model_1.parameters()).device # device(type='cuda', index=0)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)
torch.manual_seed(42)
epochs=200
# put data on the device
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
for epoch in range(epochs):
  model_1.train()
  y_pred = model_1(X_train) # 1. Forward pass
  loss = loss_fn(y_pred,y_train) # 2. Calculate the loss
  optimizer.zero_grad() # 3. Optimizer zero grad
  loss.backward() # 4. Backpropagation
  optimizer.step() # 5. Optimizer step
  ## Testing
  model_1.eval()
  with torch.inference_mode():
    test_pred = model_1(X_test)
    test_loss = loss_fn(test_pred, y_test)
  if epoch%10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
```

### 64. Putting Everything Together (Part 4): Making Predictions With a Trained Model
```py
model_1.eval()
with torch.inference_mode():
  y_preds = model_1(X_test)
plot_predictions(predictions=y_preds.cpu())
```

### 65. Putting Everything Together (Part 5): Saving and Loading a Trained Model
```py
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME="01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)
# Load a PyTorch
loaded_model_1 = LinearRegressionModelV2()
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model_1.to(device)
```

### 66. Exercise: Imposter Syndrome

### 67. PyTorch Workflow: Exercises and Extra-Curriculum
- https://www.learnpytorch.io/01_pytorch_workflow/#exercises

## Section 4: PyTorch Neural Network Classification

### 68. Introduction to Machine Learning Classification With PyTorch
- Binary classification: 
  - Spam or not
- Multiclass classification
  - Is this a photo of sushi, steak or pizza?
- Multilabel classification
  - What tags should this article have?
  - Many tags
- What we're going to cover
  - Architecture of a neural entwork classification model
  - Input shapes and output shapes of a classification model (features and labels)
  - Creating custom data to view, fit on and predict on
  - Steps in modeling
    - Creating a model, setting a loss function and optimizer, creating a training loop, evaluating a model
  - Saving and loading models
  - Harnessing the power of non-linearity
  - Different classification evaluation methods
  
### 69. Classification Problem Example: Input and Output Shapes

### 70. Typical Architecture of a Classification Neural Network (Overview)
- https://www.learnpytorch.io/02_pytorch_classification/
- https://docs.pytorch.org/docs/stable/nn.html

### 71. Making a Toy Classification Dataset
- https://github.com/mrdbourke/pytorch-deep-learning/blob/main/video_notebooks/02_pytorch_classification_video.ipynb
```py
import sklearn
from sklearn.datasets import make_circles
# make 1000 samples
n_samples = 1000
# create circles - draws 2 circles as inner/outer cicles
X,y=make_circles(n_samples, noise=0.03, random_state=42)
#print(f"X={X[:5]} y={y[:5]}") 
import pandas as pd
circles = pd.DataFrame({"X1":X[:,0], "X2":X[:,1], "label":y})
circles.head(3)
```
![ch72](./ch72.png)

### 72. Turning Our Data into Tensors and Making a Training and Test Split
```py
import torch
torch.__version__
X = torch.from_numpy(X).type(torch.float) # float64 -> float32
y = torch.from_numpy(y).type(torch.float)
print(X.shape,y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42)
```

### 73. Laying Out Steps for Modelling and Setting Up Device-Agnostic Code
- Building a model
  1. Setup device agonistic code so our code will run on an accelerator (GPU) if there is one
  2. Construct a model (by subclassing nn.Module)
  3. Define a loss function and optimizer
  4. Create a training and test loop
```py
import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 74. Coding a Small Neural Network to Handle Our Classification Data
```py
# 1. Construct a model that subclasses nn.Module
class CircleModelV0(nn.Module):
  def __init__(self):
    super().__init__()
    # 2. Create 2nn.Linear layers
    self.layer_1 = nn.Linear(in_features=2,out_features=5)
    self.layer_2 = nn.Linear(in_features=5,out_features=1)
  # 3. Define a forward()
  def forward(self,x):
    return self.layer_2(self.layer_1(x)) # x-> layer_1 -> layer_2
# 4. Instantiate an instance
model_0 = CircleModelV0().to(device)
model_0
```

### 75. Making Our Neural Network Visual
- In binary classification, any random pickup gives you 50% opportunity

### 76. Recreating and Exploring the Insides of Our Model Using nn.Sequential
- Using nn.Sequential() can make NN in an easy way but having a subclass is recommended
```py
model_0 = nn.Sequential(
  nn.Linear(in_features=2, out_features=5), 
  nn.Linear(in_features=5,out_features=1)).to(device)
```

### 77. Loss Function Optimizer and Evaluation Function for Our Classification Network
- Binary cross entropy: a loss function for binary classification, measuring the difference b/w the model's predicted probability and the actual true label. It penalizes confident wrong predictions heavily
- Logit: the raw, unnormalized scores output by the final layer of a neural network before any activation function is applied
  - Ex) Binary classification yields results as LOGIT, which would be a fractional number. They are converted into probability by activation function. Rounding those converted probability will become 0 or 1
```py
loss_fn = nn.BCELoss() # requires inputs to have gone through the sigmoid activation function prior to input to BCELoss
loss_fn = nn.BCEWithLogitsLoss() # sigmod activation funciton
#nn.Sequential(
#  nn.Sigmoid(),
#  nn.BCELoss()
#)
# why sigmoid in BC? for the numerical stability
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc
```  

### 78. Going from Model Logits to Prediction Probabilities to Prediction Labels
1. Forward pass
2. Calculate the loss
3. Optimizer zero grad
4. Loss backward (backpropagation)
5. Optimizer step (gradient descent)
* Our model outputs are raw **logits**. We can convert these logits into **prediction probabilities** by passing them to some activation function (sigmoid for BC and softmax for multiclass classification)
```py
model_0.eval()
with torch.inference_mode():
  y_logits = model_0(X_test.to(device))
y_logits[:5] # Raw logits
y_pred_probs = torch.sigmoid(y_logits) # Raw logits -> probability
y_pred_labels = torch.round(y_pred_probs) # probability -> 0 or 1
```

### 79. Coding a Training and Testing Optimization Loop for Our Classification Model
- Raw logits -> prediction probabilities -> prediction labels
```py
#torch.manual_seed(42)
#torch.cuda.manual_seed(42)
epochs=100
X_train,y_train = X_train.to(device), y_train.to(device)
X_test.y_test = X_test.to(device),y_test.to(device)
for epoch in range(epochs):
  # Training
  model_0.train()
  # 1. Forward pass
  y_logits = model_0(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))
  # 2. Loss/Accuracy
  loss = loss_fn(y_logits,y_train) # nn.BCEWithLogitsLoss expects RAW logits
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  # 3. optimizer zero grad
  optimizer.zero_grad()
  # 4. Backpropagation
  loss.backward()
  # 5. Optimizer step(gradient descent)
  optimizer.step()
  ## Testing
  model_0.eval()
  with torch.inference_mode():
    test_logits = model_0(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    test_loss = loss_fn(test_logits,y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
  if epoch%10 == 0:
    print(f"Epoch: {epoch} | Loss:  {loss: .5f}, Acc: {acc: .2f} | Test los : {test_loss: .5f}, Test_acc: {test_acc: .2f}")
```

### 80. Writing Code to Download a Helper Function to Visualize Our Models Predictions
- Looks like the aove model is not learning anything ...
- Using helper function to visualize
```py
import requests
from pathlib import Path
# Download helper functions from Learn PyTorch repo (if it's not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
from helper_functions import plot_predictions, plot_decision_boundary
```
- The current model cannot handle the complexity

### 81. Discussing Options to Improve a Model
- Improving a model (from a model perspective)
  - Add more layers: more chances to learn patterns
  - Add more hidden units: from 5 hidden units to 10 hidden units
  - For for longer (more epochs)
  - Changing the activation functions
  - Change the learning rate: exploding gradient problem
  - Change the loss function

### 82. Creating a New Model with More Layers and Hidden Units
- Increase hidden units from 5 to 10
- Increase the number of layers from 2 to 3
- Increase the number of epochs from 100 to 1000
```py
class CircleModelV1(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2,out_features=10)
    self.layer_2 = nn.Linear(in_features=10,out_features=10)
    self.layer_3 = nn.Linear(in_features=10, out_features=1)
  def forward(self, x):
    #z = self.layer_1(x)
    #z = self.layer_2(z)
    #z = self.layer_3(z)
    return self.layer_3(self.layer_2(self.layer_1(x))) # faster than running each layer
model_1 = CircleModelV1().to(device)  
```

### 83. Writing Training and Testing Code to See if Our Upgraded Model Performs Better
```py
loss_fn = nn.BCEWithLogitsLoss() # sigmod activation funciton
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc
#torch.manual_seed(42)
#torch.cuda.manual_seed(42)
epochs=1000
X_train,y_train = X_train.to(device), y_train.to(device)
X_test.y_test = X_test.to(device),y_test.to(device)
for epoch in range(epochs):
  # Training
  model_1.train()
  # 1. Forward pass
  y_logits = model_1(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))
  # 2. Loss/Accuracy
  loss = loss_fn(y_logits,y_train) # nn.BCEWithLogitsLoss expects RAW logits
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  # 3. optimizer zero grad
  optimizer.zero_grad()
  # 4. Backpropagation
  loss.backward()
  # 5. Optimizer step(gradient descent)
  optimizer.step()
  ## Testing
  model_1.eval()
  with torch.inference_mode():
    test_logits = model_1(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    test_loss = loss_fn(test_logits,y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
  if epoch%100 == 0:
    print(f"Epoch: {epoch} | Loss:  {loss: .5f}, Acc: {acc: .2f} | Test loss : {test_loss: .5f}, Test_acc: {test_acc: .2f}")  
```
- Still half and half loss

### 84. Creating a Straight Line Dataset to See if Our Model is Learning Anything

### 85. Building and Training a Model to Fit on Straight Line Data
- Applying BC model into a regression data

### 86. Evaluating Our Models Predictions on Straight Line Data

### 87. Introducing the Missing Piece for Our Classification Model Non-Linearity
- Missing piece: non-linearity
 
### 88. Building Our First Neural Network with Non-Linearity
```py
# build a model with non-linear activation function
from torch import nn
class CircleModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2,out_features=10)
    self.layer_2 = nn.Linear(in_features=10,out_features=10)
    self.layer_3 = nn.Linear(in_features=10,out_features=1)
    self.relu = nn.ReLU()
  def forward(self,x):
    return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
```

### 89. Writing Training and Testing Code for Our First Non-Linear Model
- Artificial neural network are a large combination of linear and non-linear functions which are potentially able to find patterns in data
- https://playground.tensorflow.org
```py
loss_fn = nn.BCEWithLogitsLoss() # sigmod activation funciton
optimizer = torch.optim.SGD(params=model_3.parameters(), lr=0.1)
epochs=1000
X_train,y_train = X_train.to(device), y_train.to(device)
X_test.y_test = X_test.to(device),y_test.to(device)
for epoch in range(epochs):
  # Training
  model_3.train()
  # 1. Forward pass
  y_logits = model_3(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))
  # 2. Loss/Accuracy
  loss = loss_fn(y_logits,y_train) # nn.BCEWithLogitsLoss expects RAW logits
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  # 3. optimizer zero grad
  optimizer.zero_grad()
  # 4. Backpropagation
  loss.backward()
  # 5. Optimizer step(gradient descent)
  optimizer.step()
  ## Testing
  model_3.eval()
  with torch.inference_mode():
    test_logits = model_3(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    test_loss = loss_fn(test_logits,y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
  if epoch%100 == 0:
    print(f"Epoch: {epoch} | Loss:  {loss: .5f}, Acc: {acc: .2f} | Test loss : {test_loss: .5f}, Test_acc: {test_acc: .2f}")
```

### 90. Making Predictions with and Evaluating Our First Non-Linear Model
- How to improve model_3?
  - More layers
  - More hidden units

### 91. Replicating Non-Linear Activation Functions with Pure PyTorch
```py
A = torch.arange(-10.,10,1)
plt.plot(torch.relu(A))
```
![ch91](./ch91.png)
```py
# manual implementation of relu
def relu(x: torch.Tensor) -> torch.Tensor:
  return torch.maximum(torch.tensor(0), x) # inputs must be tensors
relu(A)
```

### 92. Putting It All Together (Part 1): Building a Multiclass Dataset
```py
import torch
from torch import nn
device = "cpu"
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
NUM_CLASSES  = 4
NUM_FEATURES = 2
RANDOM_SEED  = 42
X_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES,
                            centers=NUM_CLASSES, cluster_std=1.5,
                            random_state=RANDOM_SEED)
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor) # multiclass classification
X_blob_train, X_blob_test, y_blob_train, y_blob_test = \
train_test_split(X_blob,y_blob,test_size=0.2, random_state=RANDOM_SEED)
plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob)
```
![ch92](./ch92.png)

### 93. Creating a Multi-Class Classification Model with PyTorch
```py
class BlobModel(nn.Module):
  def __init__(self, input_features, out_features, hidden_units=8):
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
      nn.Linear(in_features=input_features, out_features=hidden_units),
      nn.ReLU(),
      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      nn.ReLU(),
      nn.Linear(in_features=hidden_units, out_features=out_features),
    )
  def forward(self,x):
    return self.linear_layer_stack(x)
model_4 = BlobModel(input_features=2,out_features=4,hidden_units=8).to(device)
```

### 94. Setting Up a Loss Function and Optimizer for Our Multi-Class Model
```py
# loss function for multi-class classification
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(), lr=0.1)
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc
```

### 95. Logits to Prediction Probabilities to Prediction Labels with a Multi-Class Model
```py
model_4.eval()
with torch.inference_mode():
  y_logits = model_4(X_blob_test.to(device))
# logit -> probability
# In multiclass, softmax instead of sigmoid
## each array contains 4 fractional numbers, proabilities for each class
y_pred_probs = torch.softmax(y_logits,dim=1)
# convert probability into prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)
```

### 96. Training a Multi-Class Classification Model and Troubleshooting Code on the Fly
```py
epochs=1000
X_train,y_train = X_blob_train.to(device), y_blob_train.to(device)
X_test, y_test = X_blob_test.to(device),y_blob_test.to(device)
for epoch in range(epochs):
  # Training
  model_4.train()
  # 1. Forward pass
  y_logits = model_4(X_train)
  y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
  # 2. Loss/Accuracy
  loss = loss_fn(y_logits,y_train) # 
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  # 3. optimizer zero grad
  optimizer.zero_grad()
  # 4. Backpropagation
  loss.backward()
  # 5. Optimizer step(gradient descent)
  optimizer.step()
  ## Testing
  model_4.eval()
  with torch.inference_mode():
    test_logits = model_4(X_test)
    test_pred = torch.softmax(test_logits,dim=1).argmax(dim=1)
    test_loss = loss_fn(test_logits,y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
  if epoch%100 == 0:
    print(f"Epoch: {epoch} | Loss:  {loss: .5f}, \
          Acc: {acc: .2f} | Test loss : {test_loss: .5f}, \
          Test_acc: {test_acc: .2f}")
```

### 97. Making Predictions with and Evaluating Our Multi-Class Classification Model
```py
model_4.eval()
with torch.inference_mode():
  y_logits = model_4(X_test)
y_preds = torch.softmax(y_logits,dim=1).argmax(dim=1)
```
- With those data, models without ReLU still works OK as data are distributed quite linearly

### 98. Discussing a Few More Classification Metrics
- Accuracy: out of 100 samples, how many does our model get right?
- Precision:
- Recall:
- F1-score:
- Confusion matrix
- Classification report
- https://towardsdatascience.com/beyond-accuracy-other-classification-metrics-you-should-know-in-machine-learning-ea671be83bb7/
```py
!pip install torchmetrics
from torchmetrics import Accuracy
# Setup metric
torchmetric_accuracy = Accuracy(task="multiclass",num_classes=4).to(device)
# Calculuate accuracy
torchmetric_accuracy(y_preds, y_test)
```

### 99. PyTorch Classification: Exercises and Extra-Curriculum
- https://www.learnpytorch.io/02_pytorch_classification/#exercises

## Section 4: PyTorch Computer Vision

### 100. What Is a Computer Vision Problem and What We Are Going to Cover
- What we're going to cover
  - Getting a vision dataset to work with using torchvision.datasets
  - Architecture of a convolutional neural network (CNN) with PyTorch
  - An end-to-end multi-class image classification problem
  - Steps in modeling with CNNs in PyTorch
    - Creating a CNN model with PyTorch
    - Picking a loss and optimizer
    - Training a PyTorch computer vision model
    - Evaluating model
    
### 101. Computer Vision Input and Output Shapes

### 102. What Is a Convolutional Neural Network (CNN)

### 103. Discussing and Importing the Base Computer Vision Libraries in PyTorch

### 104. Getting a Computer Vision Dataset and Checking Out Its- Input and Output Shapes

### 105. Visualizing Random Samples of Data

### 106. DataLoader Overview Understanding Mini-Batches

### 107. Turning Our Datasets Into DataLoaders

### 108. Model 0: Creating a Baseline Model with Two Linear Layers

### 109. Creating a Loss Function: an Optimizer for Model 0

### 110. Creating a Function to Time Our Modelling Code

### 111. Writing Training and Testing Loops for Our Batched Data

### 112. Writing an Evaluation Function to Get Our Models Results

### 113. Setup Device-Agnostic Code for Running Experiments on the GPU

### 114. Model 1: Creating a Model with Non-Linear Functions

### 115. Mode 1: Creating a Loss Function and Optimizer
### 116. Turing Our Training Loop into a Function
### 117. Turing Our Testing Loop into a Function
### 118. Training and Testing Model 1 with Our Training and Testing Functions
### 119. Getting a Results Dictionary for Model 1
### 120. Model 2: Convolutional Neural Networks High Level Overview
### 121. Model 2: Coding Our First Convolutional Neural Network with PyTorch
### 122. Model 2: Breaking Down Conv2D Step by Step
### 123. Model 2: Breaking Down MaxPool2D Step by Step
### 124. Mode 2: Using a Trick to Find the Input and Output Shapes of Each of Our Layers
### 125. Model 2: Setting Up a Loss Function and Optimizer
### 126. Model 2: Training Our First CNN and Evaluating Its Results
### 127. Comparing the Results of Our Modelling Experiments
### 128. Making Predictions on Random Test Samples with the Best Trained Model
### 129. Plotting Our Best Model Predictions on Random Test Samples and Evaluating Them
### 130. Making Predictions and Importing Libraries to Plot a Confusion Matrix
### 131. Evaluating Our Best Models Predictions with a Confusion Matrix
### 132. Saving and Loading Our Best Performing Model
### 133. Recapping What We Have Covered Plus Exercises and Extra-Curriculum

    6min
### 134. What Is a Custom Dataset and What We Are Going to Cover
### 135. Importing PyTorch and Setting Up Device Agnostic Code
### 136. Downloading a Custom Dataset of Pizza, Steak and Sushi Images
### 137. Becoming One With the Data (Part 1): Exploring the Data Format
### 138. Becoming One With the Data (Part 2): Visualizing a Random Image
### 139. Becoming One With the Data (Part 3): Visualizing a Random Image with Matplotlib
### 140. Transforming Data (Part 1): Turning Images Into Tensors
### 141. Transforming Data (Part 2): Visualizing Transformed Images
### 142. Loading All of Our Images and Turning Them Into Tensors With ImageFolder
### 143. Visualizing a Loaded Image From the Train Dataset
### 144. Turning Our Image Datasets into PyTorch Dataloaders
### 145. Creating a Custom Dataset Class in PyTorch High Level Overview
### 146. Creating a Helper Function to Get Class Names From a Directory
### 147. Writing a PyTorch Custom Dataset Class from Scratch to Load Our Images
### 148. Compare Our Custom Dataset Class. to the Original Imagefolder Class
### 149. Writing a Helper Function to Visualize Random Images from Our Custom Dataset
### 150. Turning Our Custom Datasets Into DataLoaders
### 151. Exploring State of the Art Data Augmentation With Torchvision Transforms
### 152. Building a Baseline Model (Part 1): Loading and Transforming Data
### 153. Building a Baseline Model (Part 2): Replicating Tiny VGG from Scratch
### 154. Building a Baseline Model (Part 3):Doing a Forward Pass to Test Our Model Shapes
### 155. Using the Torchinfo Package to Get a Summary of Our Model
### 156. Creating Training and Testing loop Functions
### 157. Creating a Train Function to Train and Evaluate Our Models
### 158. Training and Evaluating Model 0 With Our Training Functions
### 159. Plotting the Loss Curves of Model 0
### 160. The Balance Between Overfitting and Underfitting and How to Deal With Each
### 161. Creating Augmented Training Datasets and DataLoaders for Model 1
### 162. Constructing and Training Model 1
### 163. Plotting the Loss Curves of Model 1
### 164. Plotting the Loss Curves of All of Our Models Against Each Other
### 165. Predicting on Custom Data (Part 1): Downloading an Image
### 166. Predicting on Custom Data (Part 2): Loading In a Custom Image With PyTorch
### 167. Predicting on Custom Data (Part3):Getting Our Custom Image Into the Right Format
### 168. Predicting on Custom Data (Part4):Turning Our Models Raw Outputs Into Prediction
### 169. Predicting on Custom Data (Part 5): Putting It All Together
### 170. Summary of What We Have Covered Plus Exercises and Extra-Curriculum

    6min
### 171. What Is Going Modular and What We Are Going to Cover
### 172. Going Modular Notebook (Part 1): Running It End to End
### 173. Downloading a Dataset
### 174. Writing the Outline for Our First Python Script to Setup the Data
### 175. Creating a Python Script to Create Our PyTorch DataLoaders
### 176. Turning Our Model Building Code into a Python Script
### 177. Turning Our Model Training Code into a Python Script
### 178. Turning Our Utility Function to Save a Model into a Python Script
### 179. Creating a Training Script to Train Our Model in One Line of Code
### 180. Going Modular: Summary, Exercises and Extra-Curriculum

    6min
### 181. Introduction: What is Transfer Learning and Why Use It
### 182. Where Can You Find Pretrained Models and What We Are Going to Cover
### 183. Installing the Latest Versions of Torch and Torchvision
### 184. Downloading Our Previously Written Code from Going Modular
### 185. Downloading Pizza, Steak, Sushi Image Data from Github
### 186. Turning Our Data into DataLoaders with Manually Created Transforms
### 187. Turning Our Data into DataLoaders with Automatic Created Transforms
### 188. Which Pretrained Model Should You Use
### 189. Setting Up a Pretrained Model with Torchvision
### 190. Different Kinds of Transfer Learning
### 191. Getting a Summary of the Different Layers of Our Model
### 192. Freezing the Base Layers of Our Model and Updating the Classifier Head
### 193. Training Our First Transfer Learning Feature Extractor Model
### 194. Plotting the Loss curves of Our Transfer Learning Model
### 195. Outlining the Steps to Make Predictions on the Test Images
### 196. Creating a Function Predict On and Plot Images
### 197. Making and Plotting Predictions on Test Images
### 198. Making a Prediction on a Custom Image
### 199. Main Takeaways, Exercises and Extra- Curriculum

    3min
### 200. What Is Experiment Tracking and Why Track Experiments
### 201. Getting Setup by Importing Torch Libraries and Going Modular Code
### 202. Creating a Function to Download Data
### 203. Turning Our Data into DataLoaders Using Manual Transforms
### 204. Turning Our Data into DataLoaders Using Automatic Transforms
### 205. Preparing a Pretrained Model for Our Own Problem
### 206. Setting Up a Way to Track a Single Model Experiment with TensorBoard
### 207. Training a Single Model and Saving the Results to TensorBoard
### 208. Exploring Our Single Models Results with TensorBoard
### 209. Creating a Function to Create SummaryWriter Instances
### 210. Adapting Our Train Function to Be Able to Track Multiple Experiments
### 211. What Experiments Should You Try
### 212. Discussing the Experiments We Are Going to Try
### 213. Downloading Datasets for Our Modelling Experiments
### 214. Turning Our Datasets into DataLoaders Ready for Experimentation
### 215. Creating Functions to Prepare Our Feature Extractor Models
### 216. Coding Out the Steps to Run a Series of Modelling Experiments
### 217. Running Eight Different Modelling Experiments in 5 Minutes
### 218. Viewing Our Modelling Experiments in TensorBoard
### 219. Loading the Best Model and Making Predictions on Random Images from the Test Set
### 220. Making a Prediction on Our Own Custom Image with the Best Model
### 221. Main Takeaways, Exercises and Extra- Curriculum

    4min
### 222. What Is a Machine Learning Research Paper?
### 223. Why Replicate a Machine Learning Research Paper?
### 224. Where Can You Find Machine Learning Research Papers and Code?
### 225. What We Are Going to Cover
### 226. Getting Setup for Coding in Google Colab
### 227. Downloading Data for Food Vision Mini
### 228. Turning Our Food Vision Mini Images into PyTorch DataLoaders
### 229. Visualizing a Single Image
### 230. Replicating a Vision Transformer - High Level Overview
### 231. Breaking Down Figure 1 of the ViT Paper
### 232. Breaking Down the Four Equations Overview and a Trick for Reading Papers
### 233. Breaking Down Equation 1
### 234. Breaking Down Equation 2 and 3
### 235. Breaking Down Equation 4
### 236. Breaking Down Table 1
### 237. Calculating the Input and Output Shape of the Embedding Layer by Hand
### 238. Turning a Single Image into Patches (Part 1: Patching the Top Row)
### 239. Turning a Single Image into Patches (Part 2: Patching the Entire Image)
### 240. Creating Patch Embeddings with a Convolutional Layer
### 241. Exploring the Outputs of Our Convolutional Patch Embedding Layer
### 242. Flattening Our Convolutional Feature Maps into a Sequence of Patch Embeddings
### 243. Visualizing a Single Sequence Vector of Patch Embeddings
### 244. Creating the Patch Embedding Layer with PyTorch
### 245. Creating the Class Token Embedding
### 246. Creating the Class Token Embedding - Less Birds
### 247. Creating the Position Embedding
### 248. Equation 1: Putting it All Together
### 249. Equation 2: Multihead Attention Overview
### 250. Equation 2: Layernorm Overview
### 251. Turning Equation 2 into Code
### 252. Checking the Inputs and Outputs of Equation
### 253. Equation 3: Replication Overview
### 254. Turning Equation 3 into Code
### 255. Transformer Encoder Overview
### 256. Combining equation 2 and 3 to Create the Transformer Encoder
### 257. Creating a Transformer Encoder Layer with In-Built PyTorch Layer
### 258. Bringing Our Own Vision Transformer to Life - Part 1: Gathering the Pieces
### 259. Bringing Our Own Vision Transformer to Life - Part 2: The Forward Method
### 260. Getting a Visual Summary of Our Custom Vision Transformer
### 261. Creating a Loss Function and Optimizer from the ViT Paper
### 262. Training our Custom ViT on Food Vision Mini
### 263. Discussing what Our Training Setup Is Missing
### 264. Plotting a Loss Curve for Our ViT Model
### 265. Getting a Pretrained Vision Transformer from Torchvision and Setting it Up
### 266. Preparing Data to Be Used with a Pretrained ViT
### 267. Training a Pretrained ViT Feature Extractor Model for Food Vision Mini
### 268. Saving Our Pretrained ViT Model to File and Inspecting Its Size
### 269. Discussing the Trade-Offs Between Using a Larger Model for Deployments
### 270. Making Predictions on a Custom Image with Our Pretrained ViT
### 271. PyTorch Paper Replicating: Main Takeaways, Exercises and Extra-Curriculum

    7min
### 272. What is Machine Learning Model Deployment - Why Deploy a Machine Learning Model
### 273. Three Questions to Ask for Machine Learning Model Deployment
### 274. Where Is My Model Going to Go?
### 275. How Is My Model Going to Function?
### 276. Some Tools and Places to Deploy Machine Learning Models
### 277. What We Are Going to Cover
### 278. Getting Setup to Code
### 279. Downloading a Dataset for Food Vision Mini
### 280. Outlining Our Food Vision Mini Deployment Goals and Modelling Experiments
### 281. Creating an EffNetB2 Feature Extractor Model
### 282. Create a Function to Make an EffNetB2 Feature Extractor Model and Transforms
### 283. Creating DataLoaders for EffNetB2
### 284. Training Our EffNetB2 Feature Extractor and Inspecting the Loss Curves
### 285. Saving Our EffNetB2 Model to File
### 286. Getting the Size of Our EffNetB2 Model in Megabytes
### 287. Collecting Important Statistics and Performance Metrics for Our EffNetB2 Model
### 288. Creating a Vision Transformer Feature Extractor Model
### 289. Creating DataLoaders for Our ViT Feature Extractor Model
### 290. Training Our ViT Feature Extractor Model and Inspecting Its Loss Curves
### 291. Saving Our ViT Feature Extractor and Inspecting Its Size
### 292. Collecting Stats About Our-ViT Feature Extractor
### 293. Outlining the Steps for Making and Timing Predictions for Our Models
### 294. Creating a Function to Make and Time Predictions with Our Models
### 295. Making and Timing Predictions with EffNetB2
### 296. Making and Timing Predictions with ViT
### 297. Comparing EffNetB2 and ViT Model Statistics
### 298. Visualizing the Performance vs Speed Trade-off
### 299. Gradio Overview and Installation
### 300. Gradio Function Outline
### 301. Creating a Predict Function to Map Our Food Vision Mini Inputs to Outputs
### 302. Creating a List of Examples to Pass to Our Gradio Demo
### 303. Bringing Food Vision Mini to Life in a Live Web Application
### 304. Getting Ready to Deploy Our App Hugging Face Spaces Overview
### 305. Outlining the File Structure of Our Deployed App
### 306. Creating a Food Vision Mini Demo Directory to House Our App Files
### 307. Creating an Examples Directory with Example Food Vision Mini Images
### 308. Writing Code to Move Our Saved EffNetB2 Model File
### 309. Turning Our EffNetB2 Model Creation Function Into a Python Script
### 310. Turning Our Food Vision Mini Demo App Into a Python Script
### 311. Creating a Requirements File for Our Food Vision Mini App
### 312. Downloading Our Food Vision Mini App Files from Google Colab
### 313. Uploading Our Food Vision Mini App to Hugging Face Spaces Programmatically
### 314. Running Food Vision Mini on Hugging Face Spaces and Trying it Out
### 315. Food Vision Big Project Outline
### 316. Preparing an EffNetB2 Feature Extractor Model for Food Vision Big
### 317. Downloading the Food 101 Dataset
### 318. Creating a Function to Split Our Food 101 Dataset into Smaller Portions
### 319. Turning Our Food 101 Datasets into DataLoaders
### 320. Training Food Vision Big: Our Biggest Model Yet!
### 321. Outlining the File Structure for Our Food Vision Big
### 322. Downloading an Example Image and Moving Our Food Vision Big Model File
### 323. Saving Food 101 Class Names to a Text File and Reading them Back In
### 324. Turning Our EffNetB2 Feature Extractor Creation Function into a Python Script
### 325. Creating an App Script for Our Food Vision Big Model Gradio Demo
### 326. Zipping and Downloading Our Food Vision Big App Files
### 327. Deploying Food Vision Big to Hugging Face Spaces
### 328. PyTorch Mode Deployment: Main Takeaways, Extra-Curriculum and Exercises

    6min
### 329. Introduction to PyTorch 2.0
### 330. What We Are Going to Cover and PyTorch 2 Reference Materials
### 331. Getting Started with PyTorch 2 in Google Colab
### 332. PyTorch 2.0 - 30 Second Intro
### 333. Getting Setup for PyTorch 2
### 334. Getting Info from Our GPUs and Seeing if They're Capable of Using PyTorch 2
### 335. Setting the Default Device in PyTorch 2
### 336. Discussing the Experiments We Are Going to Run for PyTorch 2
### 337. Introduction to PyTorch 2
### 338. Creating a Function to Setup Our Model and Transforms
### 339. Discussing How to Get Better Relative Speedups for Training Models
### 340. Setting the Batch Size and Data Size Programmatically
### 341. Getting More Potential Speedups with TensorFloat-32
### 342. Downloading the CIFAR10 Dataset
### 343. Creating Training and Test DataLoaders
### 344. Preparing Training and Testing Loops with Timing Steps for PyTorch 2.0 timing
### 345. Experiment 1 - Single Run without torch.compile
### 346. Experiment 2 - Single Run with torch.compile
### 347. Comparing the Results of Experiment 1 and 2
### 348. Saving the Results of Experiment 1 and 2
### 349. Preparing Functions for Experiment 3 and 4
### 350. Experiment 3 - Training a Non-Compiled Model for Multiple Runs
### 351. Experiment 4 - Training a Compiled Model for Multiple Runs
### 352. Comparing the Results of Experiment 3 and 4
### 353. Potential Extensions and Resources to Learn More

    6min
### 354. Special Bonus Lecture

    1min
### 355. Thank You!
### 356. Become An Alumni
### 357. Endorsements on LinkedIn
### 358. Learning Guideline
