## Summary
- Title: Machine Learning: Natural Language Processing in Python (V2)
- Instructor: Lazy Programmer Inc.

## Section 1: Introduction

## Section 2: Getting Set Up

3. Get your hands dirty, pratical coding experience, data links
- https://github.com/lazyprogrammer/machine_learning_examples

4. How to use Github & Extra coding tips

5. Where to get the code, notebooks and data

6. How to succeed in this course

7. Temporary 403 Errors

## Section 3: Vector Models and Text preprocessing

8. Vector Models and Test preprocessing Intro

9. Basic definitions in NLP
- Tokens: words, punctation marks, question marks, ...
- Character: a-z letters, white space, \n, ...
- Vocabulary: set of all words. Reasonable subset of words
- Corpus: Collection of writings or recorded remarks used for linquistic analysis. Here, ML dataset
- N-gram: N consecutive items
  - data: 1-gram (unigram)
  - hello world: 2-gram (bigram)

10. What is a Vector?

11. Bag of words
- Text is sequential
- The specific sequence of words gives the meaning
- But many NLP approaches do not consider the word order
  - "Bag of Words" representations
    - Unordered text
  - In vector models and classic models
  - In many cases, can yield good results still
- In probabilistic model/deep learning, ordered text might be favored

12. Count vectorizer (theory)
- Bag of words approach
- No sequence but density of words
- This will result in sparse matrices

13. Tokenization
- More than split()
- Punctuation
  - '.' vs '?'
- Casing
  - Sentiment analysis or spam detection
  - 'cat' vs 'Cat' 
- Accents
- Character-based tokenization
  - No meanings in characters
  - In scikit, `CountVectorizer(analyzer="char")`
- Word based tokenization
  - 1 million words will generate 1Mx1M matrix
  - In scikit, `CountVectorizer(analyzer="word")`
- Subword based tokenization
  - Middle ground b/w word-based and character-based
  - "walking" -> "walk" + "ing"

14. Stopwords
- How to avoid high dimensionality?
- Stopwords are the words we wish to ignore
- `CountVectorizer(stop_words=list_of_defined_words)`
  - None is the default
- nltk has stopwords per each language

15. Stemming and Lemmatization
- Handling walk, walking, walks separately will yield high dimensionality
- Stemming & Lemmatization
  - Converts words to their root word
- Stemming:
  - Very crude
  - Chops off
  - Ex) "Replacement" -> "Replac"
  - Available algorithm: Porter Stemmer in nltk
- Lemmatization:
  - More sophisticated
  - Actual rules of language
    - Kind of lookup table
  - Ex) "Better" -> "Good", "was" -> "be"
  - From nltk
    - lemmmatizer.lemmatize("going") # returns "going"
    - lemmmatizer.lemmatize("going",pos=wordnet.VERB) # returns "go"

16. Stemming and Lemmatization Demo

17. Count Vectorizer(code)
- Xtrain is a sparse matrix
```python
vectorizer = CountVectorizer(stop_words='english')
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("train score=",model.score(Xtrain,Ytrain))
print("test score=",model.score(Xtest,Ytest))
```
- Lemmatization is very expensive

18. Vector Similarity
- How to find 'similar' documents
- Metric b/w vectors
  - Distance
  - Angle
  - Cosine distance = 1 - Cosine similarity

19. TF-IDF (theory)
- How to improve the count vectorizer
- TF-IDF is popular for document retrieval and text mining
- What's wrong with the count vectorizer?
  - Stop words
- How do we know our list of stopwords is correct?
  - Application dependent
- Ex) it, and, ...
- Term Frequency - Inverse Document Frequency
  - Term frequency/document frequency
  - tfidf(t,d) = tf(t,d) * idf(t)
    - tf(t,d) = # of times t appears in d (matrix form)
    - idf(t) = log (N/N(t))
```python
from sklearn.feature_extraction.text import TfidVectorizer
tfidf = TfidVectorizer()
Xtrain = tfidf.fit_transform(train_texts)
Xtest = tfidf.transform(test_texts)
```
- Term Frequency (TF) Variations
  - Binary (1 if appears, 0 otherwise)
  - Normalize the count: tf(t,d) = count(t,d)/sum(count(t,d))
  - Take the log: tf(t,d) = log(1+count(t,d))
- Inverse Document Frequency (IDF) Variations
  - Smooth IDF: idf(t) = log (N/(N(t)+1)) + 1
  - IDF Max: idf(t) = log (max N(t')/N(t))
  - Probabilistic IDF: idf(t) = log ((N-N(t))/N(t))
- Normalizing TF-IDF
  - tfidf(t,d) = tfidf(t,d)/|tfidf(*,d)|
  - L1 or L2 norm

20. (Interactive) Recommender Exercise Prompt

21. IF-IDF (Code)
- Download from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download
```python
#wget https://lazyprogrammer.me/course_files/nlp/tmdb_5000_movie.csv -> not working Feb 2023
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
df = pd.read_csv('tmdb_5000_movies.csv')
#df.head()
x = df.iloc[0]
x['genres']
x['keywords']
j = json.loads(x['genres'])
def genres_and_keywords_to_string(row):
  genres = json.loads(row['genres'])
  genres = ' '.join(''.join(j['name'].split()) for j in genres)
  keywords = json.loads(row['keywords'])
  keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
  return "%s %s" % (genres, keywords)
df['string'] = df.apply(genres_and_keywords_to_string,axis=1)
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(df['string'])
movie2idx = pd.Series(df.index,index= df['title'])
idx = movie2idx['Scream 3']
query = X[idx]
query.toarray()
scores = cosine_similarity(query,X)
scores = scores.flatten()
plt.plot(scores)
#(-scores).argsort()
recommended_idx = (-scores).argsort()[1:6]
def recommend(title):
  idx = movie2idx[title]
  if type(idx) == pd.Series:
    idx = idx.iloc[0]
  query = X[idx]
  scores = cosine_similarity(query, X)
  scores = scores.flatten()
  recommended_idx = (-scores).argsort()[1:6]
  return df['title'].iloc[recommended_idx]
print("Recommendatino for Scream 3", recommend('Scream 3'))
```

22. Word-to-Index Mapping
- Building document-term matrix
  - Row = document, column=term (size = #documents * #terms)
- Eample
  - document 1: I like cats
  - document 2: I love cats
  - document 3: I love dogs
  - Now there are 5 unique words
  - There will be 5 columns
  - 3x5 matrix
- Sample code
```python
current_idx = 0
word2idx = {}
for doc in documents:
  tokens = word_tokenize(doc)
  for token in tokens:
    if token not in word2idx.keys():
      word2idx[token] = current_idx
      current_idx += 1
```

23. How to Build TF-IDF From scatch
- Download bbc_text_cls.csv from https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv 
```py
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
nltk.download('punkt')
df = pd.read_csv('bbc_text_cls.csv')
#
idx = 0
word2idx = {}
tokenized_docs = []
for doc in df['text']:
  words = word_tokenize(doc.lower())
  doc_as_int = []
  for word in words:
    if word not in word2idx:
      word2idx[word] = idx
      idx += 1
    doc_as_int.append(word2idx[word])
  tokenized_docs.append(doc_as_int)
# reverse mapping
idx2word = {v:k for k,v in word2idx.items()}
N = len(df['text'])
V = len(word2idx)
# term-frequency matrix
tf = np.zeros((N,V))
# populate term-frequency counts
for i,doc_as_int in enumerate(tokenized_docs):
  for j in doc_as_int:
    tf[i,j] += 1
# compute IDF
document_freq = np.sum(tf >0, axis=0)
idf = np.log(N/document_freq)
tf_idf = tf*idf
np.random.seed(123)
# show the random top 5 terms
i = np.random.choice(N)
row = df.iloc[i]
print("Label:" , row['labels'])
print("Text:", row['text'].split("\n",1)[0])
print("Top 5 terms:")
scores = tf_idf[i]
indices = (-scores).argsort()
for j in indices[:5]:
  print(idx2word[j])
```

24. Neural Word Embeddings
- Umbrella of vector models
- A document becomes a sequence of vectors
  - Something more than 'bag of words'
- Sequences of Vectors
  - Prebuilt models for sequences: CNN, RNN, Transformers
- 2 methods to discuss
  - Word2vec by Google
    - Embeddings (or vectors) are stored in the weights of the neural network
    - The goal of training is, given an input word, predict whether an output word appears in its context
  - GloVe by Stanford
    - Doesn't use neural network
    - Like a recommender system
- What can we do with word embeddings or word vectors?
  - Vector but not sparse
  - Embeddings are dens and low-dimensional
- Word Analogies
  - Can do arithemtic on vectors
  - King - Man ~ Queen - Woman

25. Neural Word Embeddings Demo
- wget https://lazyprogrammer.me/course_files/nlp/GoogleNews-vectors-negative300.bin.gz
  - 1.5GB
- Use pre-trained model as training from scratch is too expensive (~$Millions)
```py
import gdown
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
def find_analogies(w1,w2,w3):
  r = word_vectors.most_similar(positive=[w1,w3],negative=[w2])
  print("%s - %s = %s - %s"%(w1,w2,r[0][0],w3))
find_analogies('king','man','worman')
find_analogies('france','paris','london')
def nearest_neighbors(w):
  r = word_vectors.most_similar(positive=[w])
  print("neighbors of: %s" % w)
  for word, score in r:
    print("\t%s"%word)
nearest_neighbors('king')
```

26. Vector Models & Text Preprocessing Summary
- How to turn text into vectors
  - Counting
  - TF-IDF
  - Vector similarity/recommender system
  - Word-to-index mapping
  - Neural word embeddings (word2vec, GloVe)
  - Word analogies
- Text pre-processing
  - Tokenization
  - Bag of words
  - Stopwords
  - Stemming and lemmatization
  
27. Text Summarization Preview
- Text summarization is more than just TF-IDF

28. How to do NLP in other languages
- How to apply NLP into other languages or bioinformatics
- Steps of a typical NLP analysis
  - Get the text (strings)
  - Tokenize the text
  - Stopwords, stemming/lemmatization
  - Map tokens to integers
    - Tabular ML works with numbers
    - A table of the format(documents x tokens)
    - Need to know which column goes with which token!
  - Convert test into count vectors / TF-IDF
  - Do ML task (recommend, detect spam, summarize, topic model, ...)
- Scikit-Learn's count vectorizer won't tokenize Japanese
  - Build a tokenizer from scratch or use any existing library (JapaneseTokenizer)

29. Suggestion Box

## Section 4: Probablistic Models

30. Probabilistic Models
- Markov model/N-gram language model
- Application: Article spineer (black hat SEO)
- Application: Cipher decryption/code-breaking
- Later: machine learning and deep learning, which apply both vector models and probabilty models

## Section 5: Markov Models

31. Markov models section introduction
- Example
  - Finance: the basis for the Black-Scholes formula
  - Reinforcement learning: Markov Decision Process (MDP)
  - Hidden Markov Model (speech recognition, computational biology)
  - Markov Chain Monte Carlo (MCMC): numerical approximation
- Applications
  - Building a text classifier using Bayes rule + Generating poetry
  - Supervised/unsupervised ML

32. The Markov Property
- A very restrictive assumption on the dependency structure of the join distribution
- Markov assumption
  - We assume the Markov property holds, even when it does not
  
33. The Markov model
- State distribution
  - p(s_t = 1), p(s_t = 2), ...
- State transitions
  - p(s_t = j | s_t-1 = i)
  - Probabilty that state at time t is j, given that the state at time t-1 was i
- State transition matrix
  - A_ij = p(s_t = j | s_t-1 = i)
  - MxM matrix
  - Basically time-depedent matrix but when A is not dependent on time, time-homogneous Markov process
- Summary
  - We want to model a sequence of states
  - State transition matrix A, initial state distribution pi
  - Steps
    - Find the probability of a sequence
    - Given a dataset, find A and pi (learning or training)

34. Probability smoothing and log-probabilities
- Probability of sequence
  - Due to multiplication, any zero will nullify it
  - Add-one smoothing:
    - A_ij = (count(i->j) + 1)/(count(i) + M)
    - pi_j = (count(s_1 = i) + 1) /(N+M)
  - Add epsilon smoothing
    - Instead of 1, use epsilon
- Computing the probability of a sequence
  - Many multiplication of small numbers
  - Solution: compute log probabilities instead
  - We don't need exact number as what we actually do is compare
  - A>B => log(A) > log(B)

35. Building a text classifier (theory)
- Text classification is an example of **supervised learning** but Markov models are **unsupervised**
  - No label in Markov model
- Bayes' Rule
  - p(author|poem) = p(poem|author)p(author)/p(poem)
- Recap
  - We train a separate Markov model for each class
  - Each model gives us p(x|class=k) for all k
  - General form of decision rule using Bayes' rule: k* = argmax_k p(class=k|x)
  - Posterior can be simplified since we don't need its actual value
  - Maximum a posteriori (MAP): k* = argmax_k log(p(x|class=k)) + log (p(class=k))
  - Maximum likelihood: k* = argmax_k log(p(x|class=k))

36. Building a text classifier (exercise prompt)
- 2 poems by 2 authors
- Build a classifier that can distinguish b/w 2 author
- Compute train and test accuracy
- Check for class imbalance, F1 score when imbalanced
- Details
  - Save each line as a list
  - Save the labels
  - Train-test split
  - Create a mapping from unique word to unique integer index
  - Tokenize each line
  - Assign each unique word a unique integer index
  - Convert each line of text into integer lists
  - Train a Markov model for each class
  - Use smoothing (add-epsilon or add-one)
  - Consier if you need A and pi or log(A) and log(pi)
  - Write a function to compute the posterior for each class
  - Take the argmax over the posteriors to get the predicted class
  - Make predictions for both train and test sets
  - Compute accuracy for train/test
  - Check for class imbalance
  - If imbalanced, check confusion matrix and F1-score
  
37. Building a Text classifier (code pt 1)
- Input data at
  - wget -nc https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/edgar_allan_poe.txt
  - wget -nc https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt
```py
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
input_files = [ 'edgar_allan_poe.txt','robert_frost.txt']
input_texts = []
labels = []
for label, f in enumerate (input_files):
  print(f"{f} corresponds to label {label}")
  for line in open(f):
    line = line.rstrip().lower()
    if line:
      line = line.translate(str.maketrans('','',string.punctuation))
      input_texts.append(line)
      labels.append(label)
train_text, test_text, Ytrain, Ytest = train_test_split(input_texts,labels)
idx = 1
word2idx = {'<unk>':0}
# populate word2idx
for text in train_text:
  tokens = text.split()
  for token in tokens:
    if token not in word2idx:
      word2idx[token] = idx
      idx += 1
# convert data into integer format
train_text_int = []
test_text_int = []
for text in train_text:
  tokens = text.split()
  line_as_int = [word2idx[token] for token in tokens]
  train_text_int.append(line_as_int)
for text in test_text:
  tokens = text.split()
  line_as_int = [word2idx.get(token,0) for token in tokens]
  test_text_int.append(line_as_int)
# Initialize A matrix and pi vector
V = len(word2idx)
A0 = np.ones((V,V))
pi0 = np.ones(V)
A1 = np.ones((V,V))
pi1 = np.ones(V)
# compute counts for A and pi
def compute_counts(text_as_int, A, pi):
  for tokens in text_as_int:
    last_idx = None
    for idx in tokens:
      if last_idx is None:
        pi[idx] += 1
      else:
        A[last_idx,idx] += 1
      # update last idx
      last_idx = idx
compute_counts([t for t,y in zip(train_text_int,Ytrain) if y==0], A0, pi0)
compute_counts([t for t,y in zip(train_text_int,Ytrain) if y==1], A1, pi1)
# normalize A and pi
A0  /= A0.sum(axis=1, keepdims=True)
pi0 /= pi0.sum()
A1  /= A1.sum(axis=1, keepdims=True)
pi1 /= pi1.sum()
logA0  = np.log(A0)
logpi0 = np.log(pi0)
logA1  = np.log(A1)
logpi1 = np.log(pi1)
# compute priors
count0 = sum(y==0 for y in Ytrain)
count1 = sum(y==1 for y in Ytrain)
total = len(Ytrain)
p0 = count0/total
p1 = count1/total
logp0 = np.log(p0)
logp1 = np.log(p1)
# buidl a classifier
class Classifier:
  def __init__(self,logAs, logpis, logpriors):
    self.logAs = logAs
    self.logpis = logpis
    self.logpriors = logpriors
    self.K = len(logpriors)
  def _compute_log_likelihood(self,input_,class_):
    logA = self.logAs[class_]
    logpi = self.logpis[class_]
    last_idx = None
    logprob = 0
    for idx in input_:
      if last_idx is None:
        logprob += logpi[idx]
      else:
        logprob += logA[last_idx,idx]
      last_idx = idx
    return logprob
  def predict(self,inputs):
    predictions = np.zeros(len(inputs))
    for i, input_ in enumerate(inputs):
      posteriors = [self._compute_log_likelihood(input_,c) + self.logpriors[c] for c in range(self.K)]
      pred = np.argmax(posteriors)
      predictions[i] =pred
    return predictions
clf = Classifier([logA0,logA1], [logpi0, logpi1], [logp0, logp1])
Ptrain = clf.predict(train_text_int)
print(f"Train acc: {np.mean(Ptrain == Ytrain)}")
Ptest = clf.predict(test_text_int)
print(f"Test acc: {np.mean(Ptest == Ytest)}")
from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(Ytrain, Ptrain)
print(cm)
cm_test = confusion_matrix(Ytest, Ptest)
print(cm_test)
print(f1_score(Ytrain,Ptrain))
print(f1_score(Ytest, Ptest))
```

38. Building a Text classifier (code pt 2)

39. Language Model (Theory)
- Using Markov models to generate text
  - Classifying text: supervised learning
  - Generating text: unsupervised learning
- Bayes classifier
  - Discriminate model: p(y|x)
    - Logistic regression
    - Neural networks
  - Generative model: p(x|y)
    - Can be used to generate text
- Random sampling
  - May use different probabilistic distribution than pure random
- Problems with the Markov assumption 
  - The next word depends only on a single preceding word
  - Solution: extending the Markov model
    - Instead of depending on only one past state, depends on two: MxMxM
    - Third order will be MxMxMxM matrices

40. Language Model (Exercise Prompt)
- Why use dictionary? Not numpy array?
  - Sparse matrix

41. Language Model (Code pt 1)
- Markov Models
  - Markov Model Classifier / Poetry Generator
  - wget -nc https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/edgar_allan_poe.txt
  - wget -nc https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt
- We want dictionary of first word and second word with probabilty
  - Ex) ('I', 'am'):['happy':0.5, 'hungry':0.3, ...]
```py
import numpy as np
import string
np.random.seed(1234)
initial = {} # start of a phrase
first_order = {} # second word only
second_order = {}
def remove_punctuation(s):
  table = str.maketrans('','',string.punctuation)
  return [w.translate(table) for w in s]
def add2dict(d,k,v):
  if k not in d:
    d[k] = []
  d[k].append(v) # value will be a list
for line in open('robert_frost.txt'):
  tokens = remove_punctuation(line.rstrip().lower().split())
  T = len(tokens)
  for i in range(T):
    t = tokens[i]
    if i==0:
      initial[t] = initial.get(t,0.) +1
    else:
      t_1 = tokens[i-1]
      if i== T-1:
        add2dict(second_order,(t_1,t), 'END')
      if i == 1:
        add2dict(first_order, t_1,t)
      else:
        t_2 = tokens[i-2]
        add2dict(second_order,(t_2,t_1),t)
# normalize the distributions
initial_total = sum(initial.values())
for t,c in initial.items():
  initial[t] = c/initial_total
# convert [cat,cat,dog,dog,dog, ...] into {cat:0.2, dog:0.3, ... }
def list2pdict(ts):
  d = {}
  n = len(ts)
  for t in ts:
    d[t] = d.get(t,0.)+1
  for t,c in d.items():
    d[t] = c/n
  return d
for t_1,ts in first_order.items():
  # replace list with dictionary of probabilities
  first_order[t_1] = list2pdict(ts)
for k,ts in second_order.items():
  second_order[k] = list2pdict(ts)
def sample_word(d):
  p0 = np.random.random()
  cumulative = 0
  for t,p in d.items():
    cumulative += p
    if p0 < cumulative:
      return t
  assert(False) # shouldn't reach this stage
def generate():
  for i in range(4): # generates 4 lines
    sentence = []
    w0 = sample_word(initial)
    sentence.append(w0)
    w1 = sample_word(first_order[w0])
    sentence.append(w1)
    while True:
      w2 = sample_word(second_order[(w0,w1)])
      if w2 == 'END':
        break
      sentence.append(w2)
      w0 = w1
      w1 = w2
    print(' '.join(sentence))
generate()
```
- Sample results:
```
i went to bed alone and left me
might just as empty
but it isnt as if and thats not all the money goes so fast
you couldnt call it living for it aint
```

42. Language Model (code pt 2)
- How big the vocabulary size?
- How many values are stored in the dictionaries?

43. Markov Models Section Summary
- Basic idea: predicts the future from the past
  - Predict x(t) using x(t-1)
  - But can add more t-2, t-3, ...
  - Time series analysis: forecast the next value from previous values
  - Autoregressive text models with more complex architectures

## Section 6: Article Spinner

44. Article Spinning - Problem Description
- Ex: Blog
  - How to get readers?
  - Gets high ranks from search engine: search engine optimization
  - Automatic content writer
    - Just copy & paste will be detected by search egnine
    - spin content: replaces some words and pass away from duplicate check
      - Prior to ML advent: human intervention required
- The goal isn't to build an ariticle spinning product

45. Article Spinning - N-gram appraoch
- First order Markov model: p(w_t|w_t-1)
- Second order Markov model: p(w_t|w_t-1, w_t-2)
- Predicting the Middle word: p(w_t|w_t-1,w_t+1)

46. Article Spinner Exercise Prompt
- BBC News data
  - Business articles only
  - Wikipedia at https://dumps.wikimedia.org
- Build the model: VxVxV matrix, same as 2nd order language model but different ordering of dimensions
- Spinning
  - Which words to replace? How often?
  - Every word or 2 words in a row?
  - How to find if a word can be replaced?

47. Article Spinner in Python (pt 1)
```py
import numpy as np
import pandas as pd
import textwrap
import nltk
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')
df = pd.read_csv('bbc_text_cls.csv')
df.head()
labels = set(df['labels'])
label = 'business'
texts = df[df['labels'] == label]['text']
probs = {} #key: (w(t-1),w(t+1)), value: {w(t):count(w(t))}
for doc in texts:
  lines = doc.split('\n')
  for line in lines:
    tokens = word_tokenize(line)
    for i in range(len(tokens)-2):
      t_0 = tokens[i]
      t_1 = tokens[i+1]
      t_2 = tokens[i+2]
      key = (t_0,t_2)
      if key not in probs:
        probs[key] = {}
      if t_1 not in probs[key]:
        probs[key][t_1] = 1
      else:
        probs[key][t_1] += 1
# normalize probabilities
for key, d in probs.items():
  # d should represent a distribution
  total = sum(d.values())
  for k,v in d.items():
    d[k] = v/total
def spin_document(doc):
  lines = doc.split("\n")
  output = []
  for line in lines:
    if line:
      new_line = spin_line(line)
    else:
      new_line = line
    output.append(new_line)
  return "\n".join(output)
detokenizer = TreebankWordDetokenizer()
def sample_word(d):
  p0 = np.random.random()
  cumulative = 0
  for t,p in d.items():
    cumulative += p
    if p0 < cumulative:
      return t
  assert (False) # shouldn't reach here
def spin_line(line):
  tokens = word_tokenize(line)
  i = 0
  output = [tokens[0]]
  while i< (len(tokens)-2):
    t_0 = tokens[i]
    t_1 = tokens[i+1]
    t_2 = tokens[i+2]
    key = (t_0,t_2)
    p_dist = probs[key]
    if len(p_dist) > 1 and np.random.random() < 0.3:
      middle = sample_word(p_dist)
      output.append(t_1)
      output.append("<"+middle +">")
      output.append(t_2)
      i += 2
    else:
      output.append(t_1)
      i += 1
  # append the final token
  if i == len(tokens) - 2:
    output.append(tokens[-1])
  return detokenizer.detokenize(output)
np.random.seed(1234)
i = np.random.choice(texts.shape[0])
doc = texts.iloc[i]
new_doc = spin_document(doc)
print(textwrap.fill(new_doc,replace_whitespace=False, fix_sentence_endings=True))
```

48. Article Spinner in Python (pt 2)
- Grammatical error, missing quotes, wrong period found

49. Case Study: Article Spinning Gone Wrong

## Section 7: Cipher Decryption

50. Section Introduction
- Probabilistic language modeling
- Genetic algorithm
- What is a cipher?
  - Encode/decode a message
- Language modeling
  - What is the probability of this sentence?
- Generic algorithm/evolutionary algorithm
  - Optimization based on biological evolution
  
51. Ciphers
- Mapping alphabets in different order
  - Encryption/decryption

52. Language Models (Review)
- If decryption is done, the results will have very inappropriate form or have low probability
- Bigram probability
  - p(A|C) = (# of times "CA" appears in the dataset) / (# of times "C" appears in the dataset)
  - p(AB) = p(B|A)*p(A)
  - p(ABC) = p(C|AB)*p(B|A)*p(A) = p(C|B)*p(B|A)*p(A)
- Add one smoothing
  - Avoid zero when any p() is zero
  - p(x_t|x_t-1) = (cout{x_t-1->x_t} + 1) / (count{x_t-1} + V)
- Practical issue
  - Probabilities are small
  - 100 character long sentence ~ 10^-100
  - Use Log-likelihood
    - log p(x1,x2) = log p(x1) + log p(x2|x1)

53. Genetic Algorithms
- Brute force mapping counts = 26! = 4e26
- Type of mutations
  - Substition
  - Insertion
  - Deletion
  - But cannot be used for cypher decryption or character mapping 

54. Code Preparation
- Generate a random substitution cipher
- Read in Moby Dick, create a character level language model
- Encoding/decoding functions
- Genetic algorithm

55. Code pt 1
56. Code pt 2
57. Code pt 3
58. Code pt 4
59. Code pt 5
60. Code pt 6
- Download: https://lazyprogrammer.me/course_files/moby_dick.txt
```py
import numpy as np
import matplotlib.pyplot as plt
import string
import random
import re
import requests
import os
import textwrap
## creating substitution cipher
letters1 = list(string.ascii_lowercase)
letters2 = list(string.ascii_lowercase)
true_mapping = {}
random.shuffle(letters2)
for k,v in zip(letters1,letters2):
    true_mapping[k] = v
## language model
M = np.ones((26,26))
pi = np.zeros(26)
def update_transition(ch1,ch2):
    i = ord(ch1) - 97 # ord() converts a sigle unicode character into its integer equivalent. We map 'a' into 0
    j = ord(ch2) - 97
    M[i,j] += 1
def update_pi(ch):
    i = ord(ch) - 97
    pi[i] += 1
def get_word_prob(word):
    i = ord(word[0]) - 97
    logp = np.log(pi[i])
    for ch in word[1:]:
        j = ord(ch) - 97
        logp += np.log(M[i,j])
        i = j
    return logp
def get_sequence_prob(words):
    if type(words) == str:
        words = words.split()
    logp = 0
    for word in words:
        logp += get_word_prob(word)
    return logp
if not os.path.exists('moby_dick.txt'):
    print("Downloading moby dick ...")
    r = requests.get('https://lazyprogrammer.me/course_files/moby_dick.txt')
    with open('moby_dick.txt','w') as f:
        f.write(r.content.decode())
regex = re.compile('[^a-zA-Z]')
for line in open('moby_dick.txt'):
    line = line.rstrip()
    if line:
        line = regex.sub(' ',line) # replaces all non-alpha char with space
        tokens = line.lower().split()
        for token in tokens:
            ch0 = token[0]
            update_pi(ch0)
            for ch1 in token[1:]:
                update_transition(ch0,ch1)
                ch0 = ch1
# normlize
pi /= pi.sum()
M /= M.sum(axis=1,keepdims=True)
original_message = '''Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest meon shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and'''
##
def encode_message(msg):
    msg = msg.lower()
    msg = regex.sub(' ',msg)
    coded_msg = []
    for ch in msg:
        coded_ch = ch
        if ch in true_mapping:
            coded_ch = true_mapping[ch]
        coded_msg.append(coded_ch)
    return ''.join(coded_msg)
encoded_message = encode_message(original_message)
def decode_message(msg,word_map):
    decoded_msg = []
    for ch in msg:
        decoded_ch = ch
        if ch in word_map:
            decoded_ch = word_map[ch]
        decoded_msg.append(decoded_ch)
    return ''.join(decoded_msg)
## run an evolutionary algorithm to decode the message
dna_pool = []
for _ in range(20):
    dna = list(string.ascii_lowercase)
    random.shuffle(dna)
    dna_pool.append(dna)
def evolve_offspring(dna_pool, n_children):
    offspring = []
    for dna in dna_pool:
        for _ in range(n_children):
            copy = dna.copy()
            j = np.random.randint(len(copy))
            k = np.random.randint(len(copy))
            # switch
            tmp = copy[j]
            copy[j] = copy[k]
            copy[k] = tmp
            offspring.append(copy)
    return offspring + dna_pool
# main loop
num_iters = 1000
scores = np.zeros(num_iters)
best_dna = None
best_map = None
best_score = float('-inf')
for i in range(num_iters):
    if i> 0:
        dna_pool = evolve_offspring(dna_pool,3)
    dna2score = {}
    for dna in dna_pool:
        current_map = {}
        for k,v in zip(letters1,dna):
            current_map[k] = v
        decoded_message = decode_message(encoded_message, current_map)
        score = get_sequence_prob(decoded_message)
        dna2score[''.join(dna)] = score
        if score > best_score:
            best_dna = dna
            best_map = current_map
            best_score = score
        scores[i] = np.mean(list(dna2score.values()))
        # keep the best 5 dna
        sorted_dna = sorted(dna2score.items(),key = lambda x: x[1],reverse=True)
        dna_pool = [list(k) for k,v in sorted_dna[:5]]
        if i%200 == 0:
            print("iter: ", i, "score:", scores[i], " best so far: ", best_score)
decoded_message = decode_message(encoded_message,best_map)
print("LL of decoded message:", get_sequence_prob(decoded_message))
print("LL of true message:", get_sequence_prob(regex.sub(' ',original_message.lower())))
for true, v in true_mapping.items():
    pred = best_map[v]
    if true != pred:
        print("true: %s, pred: %s"%(true,pred))
# print the final decoded message
print("Decoded message:\n", textwrap.fill(decoded_message))
print("\nTrue message: \n", original_message)
```
- Not exact. Run multiple times and analyze

61. Cipher decryption - additional discussion

62. Section Conclusion
- Bisection model may work good but the uni-gram/bi-gram model may be inappropriate
  - Trigram?

## Section 8: Machine Learning Models

63. Machine Learning Models (Introduction)
- So far:
  - Vector based models
  - Probability based models
- ML model can be vector or probability based, or both

|supervised | unsupervised|
|---|---|
| spam detection | Topic modeling|
| Sentiment analysis | Latent semantic analysis|
| | Text summarization (but can be supervised) |

## Section 9: Spam Detection

64. Spam Detection - Problem Description

65. Naive Bayes Intuition
- Bayes' rule:
  - P(Y|X) = P(X|Y)P(Y)/\sum P(X|y)P(y) = P(X,Y)/P(X)
  - Input X= email, Output Y= category (spam or not)
- Naive Bayes
  - Multiple inputs are indepedent each other

66. Spam Detection - Exercise Problem
- wget https://lazyprogrammer.me/course_files/spam.csv

67. Aside: Class Imbalance, ROC, AUC, and F1 Score (pt1)
- Class Imbalance
- Binary Classification
  - TP: True Positive. Predicted Positive while actually Positive
  - TN: True Negative. Predicted negative while actually Negative
  - FP: False Positive. Predicted Positive while actually Negative
  - FN: False Negative. Predicted Negative while actually Positive
- In medical/life science
  - Sensitivity = TP/(TP+FN), higher/better
  - Specificity = TN/(TN+FP), higher/better
- Recall = sensitivity = True Positive rate. Or # of docs found/# docs we must have found
- Precision = Positive Predictive Value = TP/(TP+FP). Or # docs correctly retrieved/# docs retrieved
- F1-score: Harmonic mean of precision and recall
  - F1 = 2 (precsion*recall)/(precision + recall)

68. Aside: Class Imbalance, ROC, AUC, and F1 Score (pt2)
- ROC curve: TP vs FP
  - AUC: Area under curve

69. Spam Detection In Python
```py
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],axis=1)
df.columns = ["labels","data"]
df.head()
df['labels'].hist() # compares histogram
df['b_labels'] = df['labels'].map({'ham':0, 'spam':1})
Y = df['b_labels'].to_numpy()
df_train, df_test, Ytrain,Ytest = train_test_split(df['data'], Y, test_size=0.33)
featurizer = CountVectorizer(decode_error='ignore')
Xtrain = featurizer.fit_transform(df_train)
Xtest = featurizer.transform(df_test)
model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("train acc:", model.score(Xtrain,Ytrain))
print("test acc:", model.score(Xtest,Ytest))
Ptrain = model.predict(Xtrain)
Ptest = model.predict(Xtest)
print("train F1:", f1_score(Ytrain, Ptrain))
print("test F1:", f1_score(Ytest, Ptest))
cm = confusion_matrix(Ytrain,Ptrain)
cm # array([[3211,   13],  [  14,  495]])
def plot_cm(cm):
    classes = ['ham','spam']
    df_cm = pd.DataFrame(cm,index=classes,columns=classes)
    ax = sn.heatmap(df_cm,annot=True,fmt='g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
plot_cm(cm) # visualization of confusion matrix
def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]  ['data']:
        msg = msg.lower()
        words += msg +' '
    wordcloud = WordCloud(width=600,height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
visualize('spam')
visualize('ham')
# Now see what went wrong
X = featurizer.transform(df['data'])
df['predictions']  = model.predict(X)
# things that must be spam
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)
# things should not be spam
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in sneaky_spam:
    print(msg)
```

## Section 10: Sentiment Analysis

70. Sentiment Analysis - Problem Description
- Unlike image classification, there is an ordering to the classes
- Application
  - Reputation management
  - Customer support

71. Logistic Regression Intuition (pt 1)
- Logistic regression model is related into vector model instead of probability model
- Activation: ax(x) = w1x1 + w2x2 + b
  - w1,w2: weights
  - b: bias
  - a(x) = 0: on the line
  - a(x) < 0: one side of the line
  - a(x) > 0: other side of the line
  - Vector notation: a(x) = \sum_i w_i x_i + b = w^T x + b
- Sigmoid: logistic function
  - \sigma(x) = 1/(1+\exp(-x)): b/w [0,1]
  - p(y=1|x) = \sigma(w^T x + b)
- Naive Bayes model is generative while Logistic Regression is discriminative  

72. Multiclass Logistic Regression (pt 2)
- Sometimes called as multinomial logistic regression or maximum entropy classifier
- Multiclass logistic regression
  - We have K classes
    - K weight vectors: w1, w2, ... wk
    - K bias: b1, b2, ... bk
    - K activations: a1 = w1^T x + b1, a2 = w2^T x + b2, ... ak = wk^T x + bk
  - For each of N samples, we get K probabilities that sum to 1
  - The overall output is a matrix of NxK
  - Class predictions using probabilities?
    - Find the max

73. Logisitc Regression Training and Interpretation (pt3)
- For multiclass case
  - Interpret the weight as a matrix
  - a(x) = W^Tx + b
  - If W[d,k] is large and positive, it makes the kth activation more positive
  - if W[d,k] is large and negative, it makes the kth activation more negative

74. Sentiment Analysis - Exercise Prompt
- Use a vectorization strategy of your choice (counting, TF-IDF)
- Options: Tokenization, lemmatization, normalization
- Classifier: Use Logistic Regression
- Score function returns accurayc
- Check for class imbalance
  - When unbalanced, check metrics like AUC/F1-score + plot confusion matrix

75. Sentiment Analysis in Python (pt 1)
- wget -nc https://lazyprogrammer.me/course_files/AirlineTweets.csv
```py
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
np.random.seed(1)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
df_ = pd.read_csv('AirlineTweets.csv')
df_.head()
df = df_[['airline_sentiment','text']].copy()
df['airline_sentiment'].hist() # too many negative. Imbalance found
target_map = {'positive':1, 'negative':0, 'neutral':2}
df['target'] = df['airline_sentiment'].map(target_map)
df_train,df_test = train_test_split(df)
vectorizer = TfidfVectorizer(max_features=2000)
X_train = vectorizer.fit_transform(df_train['text'])
X_test = vectorizer.transform(df_test['text'])
Y_train = df_train['target']
Y_test = df_test['target']
model = LogisticRegression(max_iter=500)
model.fit(X_train,Y_train)
print("Train acc:", model.score(X_train, Y_train))
print("Test acc:", model.score(X_test, Y_test))
Pr_train = model.predict_proba(X_train)
Pr_test = model.predict_proba(X_test)
print("Train AUC:", roc_auc_score(Y_train, Pr_train, multi_class='ovo'))
print("Test AUC:", roc_auc_score(Y_test, Pr_test, multi_class='ovo'))
## plotting confusion matrix
P_train = model.predict(X_train)
P_test = model.predict(X_test)
cm = confusion_matrix(Y_train, P_train, normalize='true')
cm
def plot_cm(cm):
    classes = ['negative','positive','neutral' ]
    df_cm = pd.DataFrame(cm,index=classes,columns=classes)
    ax = sn.heatmap(df_cm,annot=True,fmt='g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
plot_cm(cm)
```

76. Sentiment Analysis in Python (pt 2)
- How to improve the accuracy/AUC
  - Using Positive and Negative only
```py
binary_target_list = [target_map['positive'],target_map['negative']]
df_b_train = df_train[df_train['target'].isin(binary_target_list)]
df_b_test = df_test[df_test['target'].isin(binary_target_list)]
X_train = vectorizer.fit_transform(df_b_train['text'])
X_test = vectorizer.transform(df_b_test['text'])
Y_train = df_b_train['target']
Y_test = df_b_test['target']
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)
print("Train acc:", model.score(X_train,Y_train))
print("Test acc:", model.score(X_test,Y_test))
Pr_train=model.predict_proba(X_train)[:,1]
Pr_test = model.predict_proba(X_test)[:,1]
print("Train AUC:", roc_auc_score(Y_train, Pr_train))
print("Test AUC:", roc_auc_score(Y_test, Pr_test))
model.coef_
plt.hist(model.coef_[0], bins=30)
word_index_map = vectorizer.vocabulary_
word_index_map
threshold = 2
# find extreme words
print("Most positive words:")
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold:
        print(word,weight)
#southwestair 2.860075665821313
#thank 8.070503978065155
#great 5.208733896653555
#best 3.6368642824846105
print("Most negative words:")
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight < -threshold:
        print(word,weight)
#hours -3.180120172892713
#not -4.237871164989555
#delayed -2.7045270961885732
#hour -2.069684629978255
#but -2.2210484580160723
#cancelled -2.6770621926840525
```
- Exercise: print the most-wrong tweets for both classes
  - Find a negative review where p(y=1|x) is closest to 1
  - Find a positive review where p(y=1|x) is closest to 0
- Set class_weight = 'balanced'

## Section 11: Text Summarization

77. Text Summarization Section Introduction
- Ex: Summary from Search engine
- 2 Types of summarization
  - Extractive: text taken from the original document. Relatively easy
  - Abstractive: may contain novel sequences of text not necessarily taken from the input. Hard
- Section outline
  - Method 1: Using the knowledge of vector-based methods (TF-IDF)
  - Method 2: TextRank, based on Google's PageRank

78. Text Summarization Using Vectors
- Text summarization with TF-IDF
  - Split the document into sentences
  - Score each sentence
  - Rank each sentence by those scores
  - Summary = top scoring sentences
  - This is an extractive method
  - No training data required
- Scoring each sentence
  - Score = average (non-zero TF-IDF values)
  - Important words will have a larger score
  - Why mean, not sum?
    - The sum would be biased toward longer sentences
- What to do with the scores
  - Sort the scores, pick the sentences with the highest score
  - Multiple options
    - Top N sentences
    - Top N words
    - Top N characters
    - Mixing

79. Text Summarization Exercise Prompt
- BBC news data
- Split the article into sentences (nltk.sent_tokenize)
- Compute TF-IDF matrix from list of sentences
- Score each sentence by taking the average of non-zero TF-IDF values
- Sort each sentence by score
- Print the top scoring sentences as the summary

80. Text Summarization in Python
- wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv
```py
import numpy as np
import pandas as pd
import textwrap
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
df = pd.read_csv('bbc_text_cls.csv')
doc = df[df.labels == 'business']['text'].sample(random_state=42)
def wrap(x):    
    return textwrap.fill(x,replace_whitespace=False, fix_sentence_endings=True)
print(wrap(doc.iloc[0]))
## tokenizing sentences
sents = nltk.sent_tokenize(doc.iloc[0].split("\n",1)[1])
featurizer = TfidfVectorizer(stop_words=stopwords.words('english'),norm='l1',)
X = featurizer.fit_transform(sents)
def get_sentence_score(tfidf_row):
    x = tfidf_row[tfidf_row != 0]
    return x.mean()
scores = np.zeros(len(sents))
for i in range(len(sents)):
    score = get_sentence_score(X[i,:])
    scores[i] = score
sort_idx = np.argsort(-scores)
print("Generated_summary:")
for i in sort_idx[:5]:
    print(wrap("%.2f: %s"%(scores[i],sents[i])))
# check the title of the document
doc.iloc[0].split("\n",1)[0]
#Generated_summary:
#0.14: A number of retailers have already reported poor figures for December.
#0.13: However, reports from some High Street retailers highlight the weakness of the sector.
#0.12: The ONS revised the annual 2004 rate of growth down from the
# Let's define a sinle function doing summary
def summarize(text):
    sents = nltk.sent_tokenize(text)
    X = featurizer.fit_transform(sents)
    scores = np.zeros(len(sents))
    for i in range(len(sents)):
        score = get_sentence_score(X[i,:])
        scores[i] = score
    sort_idx = np.argsort(-scores)
    for i in sort_idx[:5]:
        print(wrap("%.2f: %s"%(scores[i],sents[i])))
doc = df[df.labels == 'entertainment']['text'].sample(random_state=123)
summarize(doc.iloc[0].split("\n",1)[1])
#0.11: The Black Eyed Peas won awards for best R 'n' B video and sexiest video, both for Hey Mama.
#0.10: The ceremony was held at the Luna Park fairground in Sydney Harbour and was hosted by the Osbourne family.
#0.10: Goodrem, Green Day and the Black Eyed Peas took home two awards each.
#0.10: Other winners included Green Day, voted best group, and the Black Eyed Peas.
#0.10: The VH1 First Music Award went to Cher honouring her achievements within the music industry.
# check the title of the document
doc.iloc[0].split("\n",1)[0]
```

81. TextRank Intuition
- Recap of TF-IDF 
  - Split document into sentences
  - Compute TF-IDF matrix (sentences * terms)
  - Score each sentence
  - Take the top scoring sentences
- TextRank is an alternative method of scoring each sentence
- Google PageRank
  - Random walk
  - More linked webpage has higher probability
  - Probabilities on each web page becomes stationary after long time
  - More popular page has high score, less popular web site has low score
- Applying PageRank to TextRank
  - We want to score each sentence, as we score each webpage
  - What's the equivalent of a link from one webpage to another?
  - Number of links from one sentence to another is the **cosine similarty** b/w their TF-IDF vectors
  
82. TextRank - How it Really Works(Advanced)

83. TextRank Exercise Prompt

84. TextRank in Python (Advanced)

85. Text Summarization in Python - The easy way (Beginner)

86. Text Summarization Section Summary

## Section 12: Topic Modeling

## Section 13: Latent Semantic Analysis

## Section 14: Deep Learning 

## Section 15: The Neuron

## Section 16: Forward Aritficial Neural Networks

## Section 17: Convolutional Neural Networks

## Section 18: Recurrent Neural Networks

## Section 19: Setting Up Your Environment FAQ

## Section 20: Extra Help with Python Coding for Beginners FAQ

## Section 21: Effective Learning Strategies for Machine Learning

## Section 22: Appendix/FAQ Finale

Vector Models and Text Preprocessing
Count Vectorizer
# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv

TFIDF Recommender System
# https://www.kaggle.com/tmdb/tmdb-movie-metadata
!wget https://lazyprogrammer.me/course_files/nlp/tmdb_5000_movies.csv

Word Embeddings Demo
# Slower but always guaranteed to work
!wget -nc https://lazyprogrammer.me/course_files/nlp/GoogleNews-vectors-negative300.bin.gz

# You are better off just downloading this from the source
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# https://code.google.com/archive/p/word2vec/
# !gdown https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM

Markov Models
Markov Model Classifier / Poetry Generator
!wget -nc https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/edgar_allan_poe.txt
!wget -nc https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt

Article Spinner
# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv

Cipher Decryption
https://lazyprogrammer.me/course_files/moby_dick.txt
# is an edit of https://www.gutenberg.org/ebooks/2701
# (I removed the front and back matter)

Test text (note: you can use any text you like):

I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.

Spam Detection
# https://www.kaggle.com/uciml/sms-spam-collection-dataset
!wget https://lazyprogrammer.me/course_files/spam.csv

Sentiment Analysis
# https://www.kaggle.com/crowdflower/twitter-airline-sentiment
!wget -nc https://lazyprogrammer.me/course_files/AirlineTweets.csv

Text Summarization
# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv

Topic Modeling
# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv

Latent Semantic Analysis
!wget -nc https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/nlp_class/all_book_titles.txt

The Neuron
# https://www.kaggle.com/crowdflower/twitter-airline-sentiment
!wget -nc https://lazyprogrammer.me/course_files/AirlineTweets.csv

ANN
TF2 ANN with TFIDF
# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv

CNN
# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv

RNN
RNN Text Classification
# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv

NER TF2
# conll 2003
!wget -nc https://lazyprogrammer.me/course_files/nlp/ner_train.pkl
!wget -nc https://lazyprogrammer.me/course_files/nlp/ner_test.pkl


