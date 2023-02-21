## Summary
- Title: Machine Learning: Natural Language Processing in Python (V2)
- Instructor: Lazy Programmer Inc.

## Section 1: Introduction

## Section 2: Getting Set Up

3. Get your hands dirty, pratical coding experience, data links
- https://github.com/lazyprogrammer/machine_learning_examples

## Section 3: Vector Models and Text preprocessing

6. Basic definitions in NLP
- Tokens: words, punctation marks, question marks, ...
- Character: a-z letters, white space, \n, ...
- Vocabulary: set of all words. Reasonable subset of words
- Corpus: Collection of writings or recorded remarks used for linquistic analysis. Here, ML dataset
- N-gram: N consecutive items
  - data: 1-gram (unigram)
  - hello world: 2-gram (bigram)

8. Bag of words
- Text is sequential
- The specific sequence of words gives the meaning
- But many NLP approaches do not consider the word order
  - "Bag of Words" representations
    - Unordered text
  - In vector models and classic models
  - In many cases, can yield good results still
- In probabilistic model/deep learning, ordered text might be favored

9. Count vectorizer (theory)
- Bag of words approach
- No sequence but density of words
- This will result in sparse matrices

10. Tokenization
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

11. Stopwords
- How to avoid high dimensionality?
- Stopwords are the words we wish to ignore
- `CountVectorizer(stop_words=list_of_defined_words)`
  - None is the default
- nltk has stopwords per each language

12. Stemming and Lemmatization
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

13. Stemming and Lemmatization Demo

14. Count Vectorizer(code)
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

15. Vector Similarity
- How to find 'similar' documents
- Metric b/w vectors
  - Distance
  - Angle
  - Cosine distance = 1 - Cosine similarity

16. TF-IDF (theory)
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

17. (Interactive) Recommender Exercise Prompt

18. IF-IDF (Code)
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

19. Word-to-Index Mapping
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

20. How to Build TF-IDF From scatch
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

21. Neural Word Embeddings
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

22. Neural Word Embeddings Demo
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

23. Vector Models & Text Preprocessing Summary
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
  
24. Text Summarization Preview
- Text summarization is more than just TF-IDF

25. How to do NLP in other languages
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

26. Suggestion Box

## Section 4: Probablistic Models

27. Probabilistic Models
- Markov model/N-gram language model
- Application: Article spineer (black hat SEO)
- Application: Cipher decryption/code-breaking
- Later: machine learning and deep learning, which apply both vector models and probabilty models

## Section 5: Markov Models

28. Markov models section introduction
- Example
  - Finance: the basis for the Black-Scholes formula
  - Reinforcement learning: Markov Decision Process (MDP)
  - Hidden Markov Model (speech recognition, computational biology)
  - Markov Chain Monte Carlo (MCMC): numerical approximation
- Applications
  - Building a text classifier using Bayes rule + Generating poetry
  - Supervised/unsupervised ML

29. The Markov Property

30. The Markov model

## Section 6: Article Spinner

## Section 7: Cipher Decryption

## Section 8: Machine Learning Models

## Section 9: Spam detection

## Section 10: Sentiment Analysis

## Section 11: Text Summarization

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


