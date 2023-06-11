## Title: Data Science: Transformers for Natural Language Processing
- By Lazyprogrammer.me

## Welcome

1. Introduction
2. Outline

## Geting Setup

3. Where to get the code and data - instant access
- colab notebook: https://deeplearningcourses.com/notebooks/_j335VY5pKEEWmluSiccog
- https://github.com/lazyprogrammer/machine_learning_examples

4. How to use Github & Extra coding tips (optional)
5. Are you beginner, intermediate, or advanced? All are OK!

## Beginner's Corner

6. Beginner's Corner Section Introduction
- Sentiment analysis
- Embeddings and nearest neighbor search
- Named entity recognition (many-to-many)
- Text generation (autoregrssive)
- Masked language model (article spinning)
- Text summarization (sequence-to-sequence)
- Language translation
- Question answering
- Zero-shot classification
  - Classify text given an arbitrary set of labels

7. From RNNs to Attention and Transformers - Intuition
- The attention mechanism allows neural networks to learn very long-range dependencies in sequences
  - Longer range than LSTM
  - Attention was created for RNN but transformers use attention only, while doing away with the recurrent part
- Transformers are big and slow
  - But computationa can be done in parallel
    - Unlike RNN, which does sequentially
- Types of tasks
  - Many to one: ex) spam detection
  - Many to many: ex) speech tagging
- More tasks
  - None of above mentioned tasks
  - Like language translation
  - Problem
    - Input sequence length != target sequence length
    - Sequence rule may break (adjective may come later)
  - Solution
    - Seq2Seq
      - Encoder digests input, compressing the input
      - Decoder decompression data from encoder
      - Attention in Seq2Seq
        - For each OUTPUT token, we want to know which INPUT tokens to pay attention to
        - Encoder is bi-directional
        - Decoder is uni-directional
        - Context vector: input to decoder
![attn_seq2seq.png](./attntn_seq2seq.png)
      - Interpreting Attention weights
        - Indexed by t and t' = \alpha(t,t')
        - There are Tx*Ty weights
        - Larget weight at location (t,t') means that output t was dependent on input t'
    - Attention is All You Need
      - Get rid of RNN, keeps attention
        - RNN is slow and sequentional
      - Now can be parallelized
      - RNN has an issue of vanishing gradients
      - With attention, very long sequences can be handled, and every input is connected to every output
        - Computing cost is N^2

8. Sentiment Analysis
- Negative, neutral, positive
- Classification task
- Bag of words can work good
  - But ordering and word relationships are lost
- Hugging Face pipeline
```py
from transformers import pipeline
classifier = pipeline("sentiment-analysis") # loading a pretrained model
classsifier("this is a good movie") #=> label will be printed. List of multiple sentences can be used
```

9. Sentiment Analysis in Python
```py
!pip install transformers
# https://www.kaggle.com/crowdflower/twitter-airline-sentiment
!wget -nc https://lazyprogrammer.me/course_files/AirlineTweets.csv
from transformers import pipeline
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
# Basic usage
classifier = pipeline("sentiment-analysis")
# Will download a model if not exists yet
type(classifier)
# Output is a dictionary
classifier("This is such a great movie!")
# Multiple inputs passed in as a list
classifier([
  "This course is just what I needed.",
  "I can't understand any of this. Instructor kept telling me to meet the \
    prerequisites. What are prerequisites? Why does he keep saying that?"
])
import torch
torch.cuda.is_available()
# Use the GPU
classifier = pipeline("sentiment-analysis", device=0) 
# when gpu device is 0
df_ = pd.read_csv('AirlineTweets.csv')
# it has predefined labels but we use pre-trained model anyway
df = df_[['airline_sentiment', 'text']].copy()
df['airline_sentiment'].hist()
df = df[df.airline_sentiment != 'neutral'].copy()
target_map = {'positive': 1, 'negative': 0}
df['target'] = df['airline_sentiment'].map(target_map)
texts = df['text'].tolist()
predictions = classifier(texts) # may take > 2min. Note that this is prediction, not training
probs = [d['score'] if d['label'].startswith('P') else 1 - d['score'] \
         for d in predictions]
preds = [1 if d['label'].startswith('P') else 0 for d in predictions]
preds = np.array(preds)
print("acc:", np.mean(df['target'] == preds))
cm = confusion_matrix(df['target'], preds, normalize='true')
cm
# Scikit-Learn is transitioning to V1 but it's not available on Colab
# The changes modify how confusion matrices are plotted
def plot_cm(cm):
  classes = ['negative', 'positive']
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)
  ax = sn.heatmap(df_cm, annot=True, fmt='g')
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Target")
plot_cm(cm)
f1_score(df['target'], preds)
f1_score(1 - df['target'], 1 - preds)
roc_auc_score(df['target'], probs)
roc_auc_score(1 - df['target'], 1 - np.array(probs))
```

10. Text Generation
11. Text Generation in Python
12. Masked Language Modeling (Article Spinner)
13. Masked Lanauage Modeling (Article Spinner) in Python
14. Named Entity Recognition (NER)
15. Named Entity Recognition (NER) in Python
16. Text Summarization
17. Text Summarization in Python
18. Neural Machine Translation
19. Neural Machine Translation in Python
20. Question Answering
21. Question Answering in Python
22. Zero-Shot Classification
23. Zero-Shot Classification in Python
24. Beginner's Corner Section Summary
25. Beginner Q&A: Can we use GPT-4 for everything?
26. Suggestion Box

## Fine Tuning (Intermediate)

## Named Entity Recognition (NER) and POS Tagging (Intermediate)
