# A deep understanding of AI large language model mechanisms

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
- Exercise 1: Get and prepare text
  - Parse the full book into words, vocabulary, and econder and decoder functions
```py
f = open("time_machine.txt",'r')
text = f.readlines()
f.close()
import re
allwords = re.split('\s', ' '.join(text).lower())
vocab = sorted(set(allwords)) # set() for uniqueness
# vocab  
# encoder/decoder
word2idx = {}
for i, word in enumerate(vocab):
  word2idx[word] = i
idx2word = {}
for i, word in enumerate(vocab):
  idx2word[i] = word
def incoder(word2idx,words):
  res = []
  for word in words:
    res.append(word2idx[word])
  return res
def decoder(idx2word,ids):
  res = []
  for idx in ids:
    res.append(idx2word[idx])
  return res
```
- Exercise 2: A random walk through the time machine
  - Generate 10 random integers and create a sentence from those tokens
  - Brown noise is the cumulative sum of random numbers
    - We use only -1 and +1
  - Generate 30 random token IDs as Brownian noise starting from a randomly selected token ID
    - Decode to tokens and enjoy the poetry  
```py
import random
rvalue = random.randint(0,len(vocab))
vsize = len(vocab)
alist = []
for n in range(30):
  rvalue += random.choice([-1,1])
  if rvalue < 0:
    rvalue += vsize
  if rvalue >= vsize:
    rvalue -= vsize
  alist.append(rvalue)
print(alist)  
print(decoder(idx2word,alist))
```
- Exercise 3: Distribution of word lengths
  - Count the number of characters in each word of the text
  - Make scatter plot and histogram
```py
import matplotlib.pyplot as plt
x = []
y = []
for i, word in enumerate(allwords):
  x.append(i)
  y.append(len(word))
plt.scatter(x,y)  
plt.show()
plt.hist(y,bins=30)
plt.show()
```
- Exercise 4: Encode an novel sentence
  - Find a new word compared with setence = "The space aliens came to Earth to steal watermelons and staplers"
- Exercise 5: Create a new encoder
  - When a token is not found, return `<|unk|>`
```py
def incoder(word2idx,words):
  res = []
  for word in words:
    if word in vocab:
      res.append(word2idx[word])
    else:
      res.append("<|unk|>")
  return res
```
- Instead of the above code, add `<|unk|>` into the last of word2idx and idx2word, having it as a vocabulary

### 13. Tokenizing characters vs. subwords vs. words
- Hello
  - Character: H, e, l, l, o
  - Subword: He llo
  - Word: Hello
- Character
  - Language agnostic
  - Handles misspellings
  - Compact
  - Long sequences
  - Little semantic meaning
  - Slow learning
  - Need more data
- Word
  - Semantic relations
  - Memory efficient
  - Faster learning
  - Human-interpretable
  - Large vocab
  - Language-dependent
  - OOV (Out-of-Vocab) difficulties  

### 14. Byte-pair encoding algorithm
- BPE is simple and widely used in real-world applications
- BPE motivation and theory
  - Tokenization efficiency can be improved with tokens that represent longer character sequences - word or subword
  - "Often appear together" is calculated emprically using large text datasets found online
- BPE algorithm 
  - Step 0: initialize a vocabulary comprising only characters (no pairs)
  - Step 1: Loop through the text and count the frequency of sequential pairs
    - For "love", sequential pairs are "lo", "ov", "ve"
  - Step 2: Find the most frequent pair and merge them to create a new token
  - Step 3: Add that new token to the vocabulary
  - Step 4: Loop through the text and replace the sequence with that new pair
    - Only the most frequent pair only
  - Step 5: Repeate steps 1-4 until the vocabl contains k tokens 
    - 100,000 tokens at GPT4
- Demo
  - Sample text = "like liker love lovely hug hugs hugging hearts"
  - Characters: ' ', a, e, g, h, ....
  - Token_pairs: 'li', 'ik', 'ke', 'e ', ' l', ...
  - Vocab:    
```py
import numpy as np
text = 'like liker love lovely hug hugs hugging hearts'
chars = list(set(text))
chars.sort()
for l in chars:
  print(f'{l} appears {text.count(l)} times.')
# make a vocabulary
vocab = {word:i for i, word in enumerate(chars)}  
'''
  appears 7 times.
a appears 1 times.
e appears 5 times.
g appears 5 times.
h appears 4 times.
i appears 3 times.
...
'''
origtext = list(text)
token_pairs = dict()
for i in range(len(origtext)-1):
  pair = origtext[i] + origtext[i+1]
  if pair in token_pairs:
    token_pairs[pair] += 1
  else:
    token_pairs[pair] = 1
token_pairs    
'''
'li': 2,
 'ik': 2,
 'ke': 2,
 'e ': 2,
 ' l': 3,
 'er': 1,
 ...
'''
mostFreqPair_idx = np.argmax(list(token_pairs.values()))
mostFreqPair_char = list(token_pairs.keys())[mostFreqPair_idx]
vocab[mostFreqPair_char] = max(vocab.values()) +1
vocab
'''
 ...
 't': 12,
 'u': 13,
 'v': 14,
 'y': 15,
 ' h': 16}
'''
# loops
newtext = []
i = 0
while i < (len(origtext)-1):
  if (origtext[i] + origtext[i+1]) == mostFreqPair_char:
    newtext.append(mostFreqPair_char)
    print(f'added "{mostFreqPair_char}"')
    i+=2
  else:
    newtext.append(origtext[i])
    i+=1
print(f'Original text: {origtext}')
print(f'Updated_text: {newtext}')
'''
added " h"
added " h"
added " h"
added " h"
Original text: ['l', 'i', 'k', 'e', ' ', 'l', 'i', 'k', 'e', 'r', ' ', 'l', 'o', 'v', 'e', ' ', 'l', 'o', 'v', 'e', 'l', 'y', ' ', 'h', 'u', 'g', ' ', 'h', 'u', 'g', 's', ' ', 'h', 'u', 'g', 'g', 'i', 'n', 'g', ' ', 'h', 'e', 'a', 'r', 't', 's']
Updated_text: ['l', 'i', 'k', 'e', ' ', 'l', 'i', 'k', 'e', 'r', ' ', 'l', 'o', 'v', 'e', ' ', 'l', 'o', 'v', 'e', 'l', 'y', ' h', 'u', 'g', ' h', 'u', 'g', 's', ' h', 'u', 'g', 'g', 'i', 'n', 'g', ' h', 'e', 'a', 'r', 't']
'''
token_pairs = dict()
for i in range(len(newtext)-1):
  pair = newtext[i] + newtext[i+1]
  if pair in token_pairs:
    token_pairs[pair] += 1
  else:
    token_pairs[pair] = 1
token_pairs    
'''
 ...
 ve': 2,
 'el': 1,
 'ly': 1,
 'y h': 1,
 ' hu': 3,
 'ug': 3,
 'g h': 2,
 'gs': 1,
 's h': 1,
 'gg': 1,
 ...
'''
```
- Additional steps to create professional tokenizers
  - Data selection
  - Cleaning and preprocessing (removing non-ASCII characters, excessive whitespace, formatting)
  - Subword initialization ("ing", "er", "ly") or constraints (e.g., code, url, numbers)
  - Experimenting with vocab size
  - Merge based on bytes not characters !!!
  - Handling rare words to reduce OOV problems
  - Post-training adjustments (removing or adding merges
  - Adding special tokens)
  - Coding tricks to decrease compute time

### 15. CodeChallenge: Byte-pair encoding to a desired vocab size
- Up to 25 tokens of vocab
```py

def update_vocab(token_pairs, vocab):
  mostFreqPair_idx = np.argmax(list(token_pairs.values()))
  mostFreqPair_char = list(token_pairs.keys())[mostFreqPair_idx]
  vocab[mostFreqPair_char] = max(vocab.values()) +1
  return vocab, mostFreqPair_char

def make_newtext(pretext, mostFreqPair_char):
  newtext = []
  i = 0
  while i < (len(pretext)-1):
    if (pretext[i] + pretext[i+1]) == mostFreqPair_char:
      newtext.append(mostFreqPair_char)
      print(f'added "{mostFreqPair_char}"')
      i+=2
    else:
      newtext.append(pretext[i])
      i+=1
  return newtext

def make_token_pairs(newtext):
  token_pairs = dict()
  for i in range(len(newtext)-1):
    pair = newtext[i] + newtext[i+1]
    if pair in token_pairs:
      token_pairs[pair] += 1
    else:
      token_pairs[pair] = 1
  return token_pairs    
#
import numpy as np
text = 'like liker love lovely hug hugs hugging hearts'
chars = list(set(text))
chars.sort()
# make a vocabulary
vocab = {word:i for i, word in enumerate(chars)}  
origtext = list(text)
token_pairs = make_token_pairs(origtext)
newtext = origtext
while len(vocab.values()) < 26:
  vocab, mostFreqPair_char = update_vocab(token_pairs, vocab)
  newtext = make_newtext(newtext, mostFreqPair_char)
  token_pairs = make_token_pairs(newtext)
  print("vocab size = ", len(vocab))
print(newtext)
'''
vocab size =  25
added "like"
vocab size =  26
['like', ' l', 'ike', 'r', ' love', ' love', 'l', 'y', ' hug', ' hug', 's', ' hug']
'''
```

### 16. Exploring ChatGPT4's tokenizer
- Many tokens with space-padded
- pip install tiktoken
- https://github.com/vnglst/gpt4-tokens/blob/main/decode-tokens.ipynb

### 17. CodeChallenge: Token count by subword length (part 1)
```py
import requests
import re
text = requests.get('https://www.gutenberg.org/files/35/35-0.txt').text
# split by punctuation
words = re.split(r'([,.:;—?_!"“()\']|--|\s)',text)
words = [item.strip() for item in words if item.strip()]
print(f'There are {len(words)} words.')
words[10000:10050]
# what happens if we just tokenize the raw (unprocessed) text?
import tiktoken
tokenizer = tiktoken.get_encoding('cl100k_base')
tmTokens = tokenizer.encode(text)
print(f'The text has {len(tmTokens):,} tokens and {len(words):,} words.')
# The text has 43,053 tokens and 37,786 words.
```

### 18. CodeChallenge: Token count by subword length (part 2)

### 19. How many "r"s in strawberry?
- Early ChatGPT couldn't answer correctly
```py
import tiktoken
tokenizer = tiktoken.get_encoding('cl100k_base')
tmTokens = tokenizer.encode('strawberry')
for t in tmTokens:
  print(f'{t:5d} = {tokenizer.decode([t])}')
'''
  496 = str
  675 = aw
15717 = berry
'''
r = tokenizer.encode('r')
print(f'{r} = {tokenizer.decode(r)}')
r in tmTokens # False
# let's decode:
s_string = tokenizer.decode(tmTokens)
s_string
r_string = tokenizer.decode(r)
s_string.count(r_string) # 3 is found
```

### 20. CodeChallenge: Create your algorithmic rapper name :)

### 21. Tokenization in BERT
- BERT is trained to find missing word in the word completion
  - Good at classification
```py
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dir(tokenizer)
print(tokenizer.vocab_size) # 30522
text = 'science is great'
res1 = tokenizer.convert_tokens_to_ids(text)
res3 = tokenizer.encode(text)
for i in res3:
  print(f'Token {i} is "{tokenizer.decode(i)}"')
'''
Token 101 is "[CLS]" # classification
Token 2671 is "science"
Token 2003 is "is"
Token 2307 is "great"
Token 102 is "[SEP]"  # sentence separation
'''
```  

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


