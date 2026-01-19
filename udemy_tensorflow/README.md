## Complete Tensorflow 2 and Keras Deep Learning Bootcamp
- Instructor: Pierian Training

## Section 1: Course Overview, Installs, and Setup

### 1. Auto-Welcome Message

### 2. Course Overview
- No github repo

### 3. Course Setup and Installation

### 4. FAQ - Frequently Asked Questions

## Section 2: Course overview confirmation

### Quiz 1: PLEASE WATCH COURSE OVERVIEW LECTURE

## Section 3: NumPy Crash Course

### 5. Introduction to NumPy

### 6. NumPy Arrays

### 7. Numpy Index Selection

### 8. NumPy Operations

### 9. NumPy Exercises

### 10. Numpy Exercises - Solutions

## Sectin 4: Pandas Crash Course

### 11. Introduction to Pandas

### 12. Pandas Series

### 13. Pandas DataFrames - Part One

### 14. Pandas DataFrames - Part Two

### 15. Pandas Missing Data

### 16. GroupBy Operations

### 17. Pandas Operations

### 18. Data Input and Output

### 19. Pandas Exercises

### 20. Pandas Exercises - Solutions

## Section 5: Visualization Crash Course

### 21. Introduction to Python Visualization

### 22. Matplotlib Basics

### 23. Seaborn Basics
- https://seaborn.pydata.org

### 24. Data Visualization Exercises

### 25. Data Visualization Exercises - Solutions

## Section 6: Machine Learning Concepts Overview

### 26. What is Machine Learning?

### 27. Supervised Learning Overview
- Splitting data into 3 sets
  - Training data: model parameters
  - Validation data: hyperparameters
  - Test data: final performance metric

### 28. Overfitting
- Overfitting
  - Low error on training sets but higher error on test/validation sets
- Underfitting
  - Model doesn't capture the underlying trend of the data
  - Low variance but high bias
  - Model might be too simple

### 29. Evaluating Performance - Classification Error Metrics
- Accuracy: number of correct predictions made by the model divided by the total number of predictions
- Recall: ability of a model to find all the relevant cases within a dataset
  - The number of true positives / (the number of true positives + the number of false negatives)
- Precision: ability of a classification model to identify only the relevant data points
  - The number of true positives / (the number of true positives + the number of false positives)
- Recal expresses the abilty to find all relevant instances in a dataset while precision expresses the proportion of the relevant data
- F1 score: harmonic mean of precision and recall
  - It punishes extreme values
  - F1 = 2 \* (precision \* recall)/(precision + recall)

Confusion matrix |Predicted Positive |  Predicted Negative
----------------|-------------------|-----------
Actual Positive |  TP               | FN
Actual Negative |  FP               | TN

- When False Negative is good
  - Spam detection: A few spam mails (FN) reach your inbox than having legitimate emails (FP) blocked
  - Quality control if re-testing is expensive
  - Security when false alarm is expensive: Minimizing FPs (alerting on normal) is important than meassing a few actual threats (FN)
- When False Positive is good
  - Missing FN is far worse than FP
  - Detecing a rare but dangeerous disease whre missing a case (FN) is catastrophic

### 30. Evaluating Performance - Regression Error Metrics
- MAE
- MSE
- RMSE

### 31. Unsupervised Learning
- Clustering
- Anomaly detection
- Unsupervised learning
  - No historical data
  - Evaluation is much harder and more nuanced
  
## Section 7: Basic Artificial Neural Networks - ANNs

### 32. Introduction to ANN Section

### 33. Perceptron Model

### 34. Neural Networks

### 35. Activation Functions
- https://en.wikipedia.org/wiki/Activation_function

### 36. Multi-Class Classification Considerations
- Non-exclusive classes
  - A data point may have multiple classes/categories
- Mutually exclusive classes
  - Each data point has one class only
- One-hot encoding
  - Class values might be replaced as 0 or 1
  - Multiple classes will be a group of 0 and 1 (A=0, B=1, C=1, ...)


### 37. Cost Functions and Gradient Descent
- Cost function == loss function
  - How far from training data
- Deep learning
  - Cross-entropy is commonly used

  
### 38. Backpropagation

### 39. TensorFlow vs. Keras Explained

### 40. Keras Syntax Basics - Part One - Preparing the Data

### 41. Keras Syntax Basics - Part Two - Creating and Training the Model

### 42. Keras Syntax Basics - Part Three - Model Evaluation

### 43. Keras Regression Code Along - Exploratory Data Analysis

### 44. Keras Regression Code Along - Exploratory Data Analysis - Continued

### 45. Keras Regression Code Along - Data Preprocessing and Creating a Model

### 46. Keras Regression Code Along - Model Evaluation and Predictions

### 47. Keras Classification Code Along - EDA and Preprocessing

### 48. Keras Classification - Dealing with Overfitting and Evaluation

### 49. TensorFlow 2.0 Keras Project Options Overview

### 50. TensorFlow 2.0 Keras Project Notebook Overview

### 51. Keras Project Solutions - Exploratory Data Analysis

### 52. Keras Project Solutions - Dealing with Missing Data

### 53. Keras Project Solutions - Dealing with Missing Data - Part Two

### 54. Keras Project Solutions - Categorical Data

### 55. Keras Project Solutions - Data PreProcessing

### 56. Keras Project Solutions - Creating and Training a Model

### 57. Keras Project Solutions - Model Evaluation

### 58. Tensorboard

    18min

### 59. CNN Section Overview
### 60. Image Filters and Kernels
### 61. Convolutional Layers
### 62. Pooling Layers
### 63. MNIST Data Set Overview
### 64. CNN on MNIST - Part One - The Data
### 65. CNN on MNIST - Part Two - Creating and Training the Model
### 66. CNN on MNIST - Part Three - Model Evaluation
### 67. CNN on CIFAR-10 - Part One - The Data
### 68. CNN on CIFAR-10 - Part Two - Evaluating the Model
### 69. Downloading Data Set for Real Image Lectures
### 70. CNN on Real Image Files - Part One - Reading in the Data
### 71. CNN on Real Image Files - Part Two - Data Processing
### 72. CNN on Real Image Files - Part Three - Creating the Model
### 73. CNN on Real Image Files - Part Four - Evaluating the Model
### 74. CNN Exercise Overview
### 75. CNN Exercise Solutions

    9min

### 76. RNN Section Overview
### 77. RNN Basic Theory
### 78. Vanishing Gradients
### 79. LSTMS and GRU
### 80. RNN Batches
### 81. RNN on a Sine Wave - The Data
### 82. RNN on a Sine Wave - Batch Generator
### 83. RNN on a Sine Wave - Creating the Model
### 84. RNN on a Sine Wave - LSTMs and Forecasting
### 85. RNN on a Time Series - Part One
### 86. RNN on a Time Series - Part Two
### 87. RNN Exercise
### 88. RNN Exercise - Solutions
### 89. Bonus - Multivariate Time Series - RNN and LSTMs

    16min

### 90. Introduction to NLP Section
### 91. NLP - Part One - The Data
### 92. NLP - Part Two - Text Processing
### 93. NLP - Part Three - Creating Batches
### 94. NLP - Part Four - Creating the Model
### 95. NLP - Part Five - Training the Model
### 96. NLP - Part Six - Generating Text

    9min

### 97. Introduction to Autoencoders
### 98. Autoencoder Basics
### 99. Autoencoder for Dimensionality Reduction
### 100. Autoencoder for Images - Part One
### 101. Autoencoder for Images - Part Two - Noise Removal
### 102. Autoencoder Exercise Overview
### 103. Autoencoder Exercise - Solutions

    11min

### 104. GANs Overview
### 105. Creating a GAN - Part One- The Data
### 106. Creating a GAN - Part Two - The Model
### 107. Creating a GAN - Part Three - Model Training
### 108. DCGAN - Deep Convolutional Generative Adversarial Networks

    7min

### 109. Introduction to Deployment
### 110. Creating the Model
### 111. Model Prediction Function
### 112. Running a Basic Flask Application
### 113. Flask Postman API
### 114. Flask API - Using Requests Programmatically
### 115. Flask Front End
### 116. Live Deployment to the Web
