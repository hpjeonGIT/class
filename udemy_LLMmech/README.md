
## A deep understanding of AI large language model mechanisms



## Section 1: Introductions

### 1. [IMPORTANT] Prerequisites and how to succeed in this course
- Take nodes by hand!
- No lecture slides provided

### 2. Using the Udemy platform

### 3. Getting the course code, and the detailed overview
- Ref: https://github.com/mikexcohen/LLM_course

### 4. Do you need a Colab Pro subscription?

### 5. About the "CodeChallenge" videos
- HELPER vs SOLUTION

## Section 2: Part1: Tokenization and embeddings

### 6. Tokenizations and embeddings

## Section 3: Words to tokens to numbers

### 7. Why text needs to be numbered
- Key terms
  - Token: a piece of text that can be represented as an integer
    - By character: Hello -> H + e + l + l + o
    - By subword: Hello -> He + llo
    - By word: Hello -> Hello
  - Embedding
  - Subword
- Q: Why embeddings are more efficient than tokens?
- Text is converted into tokens 
  - Tokens must be converted into embeddings
  - LLMs modify the embeddings for classification and generation
- Embedding: a dense numeric representation of a token
  - More text can be represented using fewer numbers
  - Semantic relations across tokens can be represented
- Real embeddings are:
  - High dimensional - 768D
  - Not human readable
  - Modified dynamically during model calculations

### 8. Parsing text to numbered tokens
- Vocalbulary (aka lexicon): A unique set of tokens in a tokenizer
- Encoder: A function that maps text into integers
- Decoder: A function (or a lookup table) that maps integers into text
```py
text = [' All that we are is the result of what we have thought', 
        'To be or not to be that is the quesiton', 
        'Be yourself eevenone else is already taken']
import re
allwords = re.split('\s', ' '.join(text).lower())
vocab = sorted(set(allwords)) # set() for uniqueness
vocab  
# encoder/decoder
word2idx = {}
for i, word in enumerate(vocab):
  word2idx[word] = i
idx2word = {}
for i, word in enumerate(vocab):
  idx2word[i] = word
idx2word
   
```

### 9. CodeChallenge: Create and visualize tokens (part 1)
- Key topics
  - How to identify "target" and their contexts
  - What "one-hot encoding" means and how it looks
- Context is the tokens before and after the target token
```py
# find the context of "to"
targetId = word2idx['to']
for i, el in enumerate(allwords):
  if el == 'to':
    print(allwords[i-1],el,allwords[i+1])
# thought to be
# not to be    
```    

### 10. CodeChallenge: Create and visualize tokens (part 2)
- One-hot encoding: represent categorical data in a sparse matrix, with columns corresponding to category
  - New category adds a new column
```py
import numpy as np
word_matrix = np.zeros((len(allwords),len(vocab)))
for i, word in enumerate(allwords):
  word_matrix[i,word2idx[word]]  = 1
#print(word_matrix)
import matplotlib.pyplot as plt
plt.imshow(word_matrix)
```

### 11. Preparing text for tokenization
- How to import text from the web (gutenberg.org)
- Introduction to regexp to clean
- Download the full text of 'The Time Machine' from gutenberg.org


### 12. CodeChallenge: Tokenizing The Time Machine

### 13. Tokenizing characters vs. subwords vs. words

### 14. Byte-pair encoding algorithm

### 15. CodeChallenge: Byte-pair encoding to a desired vocab size

### 16. Exploring ChatGPT4's tokenizer

### 17. CodeChallenge: Token count by subword length (part 1)

### 18. CodeChallenge: Token count by subword length (part 2)

### 19. How many "r"s in strawberry?

### 20. CodeChallenge: Create your algorithmic rapper name :)

### 21. Tokenization in BERT

### 22. CodeChallenge: Character counts in BERT tokens

### 23. Translating between tokenizers

### 24. CodeChallenge: More on token translation

### 25. CodeChallenge: Tokenization compression ratios

### 26. Tokenization in different languages

### 27. CodeChallenge: Zipf's law in characters and tokens

### 28. Word variations in Claude tokenizer

    14min

### 29. Word2Vec vs. GloVe vs. GPT vs. BERT... oh my!
### 30. Exploring GloVe pretrained embeddings
### 31. CodeChallenge: Wikipedia vs. Twitter embeddings (part 1)
### 32. CodeChallenge: Wikipedia vs. Twitter embeddings (part 2)
### 33. Exploring GPT2 and BERT embeddings
### 34. CodeChallenge: Math with tokens and embeddings
### 35. Cosine similarity (and relation to correlation)
### 36. CodeChallenge: GPT2 cosine similarities
### 37. CodeChallenge: Unembeddings (vectors to tokens)
### 38. Position embeddings
### 39. CodeChallenge: Exploring position embeddings
### 40. Training embeddings from scratch
### 41. Create a data loader to train a model
### 42. Build a model to learn the embeddings
### 43. Loss function to train the embeddings
### 44. Train and evaluate the model
### 45. CodeChallenge: How the embeddings change
### 46. CodeChallenge: How stable are embeddings?

    18min

### 47. Large language models

    1min

### 48. Why build when you can download?
### 49. Model 1: Embedding (input) and unembedding (output)
### 50. Understanding nn.Embedding and nn.Linear
### 51. CodeChallenge: GELU vs. ReLU
### 52. Softmax (and temperature): math, numpy, and pytorch
### 53. Randomly sampling words with torch.multinomial
### 54. Other token sampling methods: greedy, top-k, and top-p
### 55. CodeChallenge: More softmax explorations
### 56. What, why, when, and how to layernorm
### 57. Model 2: Position embedding, layernorm, tied output, temperature
### 58. Temporal causality via linear algebra (theory)
### 59. Averaging the past while ignoring the future (code)
### 60. The "attention" algorithm (theory)
### 61. CodeChallenge: Code Attention manually and in Pytorch
### 62. Model 3: One attention head
### 63. The Transformer block (theory)
### 64. The Transformer block (code)
### 65. Model 4: Multiple Transformer blocks
### 66. Multihead attention: theory and implementation
### 67. Working on the GPU
### 68. Model 5: Complete GPT2 on the GPU
### 69. CodeChallenge: Time model5 on CPU and GPU
### 70. Inspecting OpenAI's GPT2
### 71. Summarizing GPT using equations
### 72. Visualizing nano-GPT
### 73. CodeChallenge: How many parameters? (part 1)
### 74. CodeChallenge: How many parameters? (part 2)
### 75. CodeChallenge: GPT2 trained weights distributions
### 76. CodeChallenge: Do we really need Q?

    22min

### 77. What is "pretraining" and is it necessary?
### 78. Introducing huggingface.co
### 79. The AdamW optimizer
### 80. CodeChallenge: SGD vs. Adam vs. AdamW
### 81. Train model 1
### 82. CodeChallenge: Add a test set
### 83. CodeChallenge: Train model 1 with GPT2's embeddings
### 84. CodeChallenge: Train model 5 with modifications
### 85. Create a custom loss function
### 86. CodeChallenge: Train a model to like "X"
### 87. CodeChallenge: Numerical scaling issues in DL models
### 88. Weight initializations
### 89. CodeChallenge: Train model 5 with weight inits
### 90. Dropout in theory and in Pytorch
### 91. Should you output logits or log-softmax(logits)?
### 92. The FineWeb dataset
### 93. CodeChallenge: Fine dropout in model 5 (part 1)
### 94. CodeChallenge: Fine dropout in model 5 (part 2)
### 95. CodeChallenge: What happens to unused tokens?
### 96. Optimization options

    5min

### 97. What does "fine-tuning" mean?
### 98. Fine-tune a pretrained GPT2
### 99. CodeChallenge: Gulliver's learning rates
### 100. On generating text from pretrained models
### 101. CodeChallenge: Maximize the "X" factor
### 102. Alice in Wonderland and Edgar Allen Poe (with GPT-neo)
### 103. CodeChallenge: Quantify the Alice/Edgar fine-tuning
### 104. CodeChallenge: A chat between Alice and Edgar
### 105. Partial fine-tuning by freezing attention weights
### 106. CodeChallenge: Fine-tuning and targeted freezing (part 1)
### 107. CodeChallenge: Fine-tuning and targeted freezing (part 2)
### 108. Parameter-efficient fine-tuning (PEFT)
### 109. CodeGen for code completion
### 110. CodeChallenge: Fine-tune codeGen for calculus
### 111. Fine-tuning BERT for classification
### 112. CodeChallenge: IMDB sentiment analysis using BERT
### 113. Gradient clipping and learning rate scheduler (part 1)
### 114. Gradient clipping and learning rate scheduler (part 2)
### 115. CodeChallenge: Clip, freeze, and schedule BERT
### 116. Saving and loading trained models
### 117. BERT decides: Alice or Edgar?
### 118. CodeChallenge: Evolution of Alice and Edgar (part 1)
### 119. CodeChallenge: Evolution of Alice and Edgar (part 2)
### 120. Why fine-tune when you can use AGI?

    7min


