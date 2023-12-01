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
- How to predict future
  - Autoregressive time series: predicts next value using past values (ARIMA)
  - Autoregressive language model: language is a time series of categorical objects
    - Markov model
- History of language models
  - Markov assumption: x(t+1) depends only on x(t), x(t) only  on x(t-1), ...
  - Markov models are not scalable O(V^N)
- Use of autoregressive language models
  - Generates poetry
  - Use cases: writing emails/creative writing, github copilot
  - Code preparation
  ```py
  from transformers import pipeline
  gen = pipeline("text-generation") # uses GPT-2
  prompt = "Neural networks with attention have been used with great success"
  get(prompt)
  ```

11. Text Generation in Python
```py
!wget -nc https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt
!pip install transformers
from transformers import pipeline, set_seed
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
lines = [line.rstrip() for line in open('robert_frost.txt')]
lines = [line for line in lines if len(line) > 0] # removing empty lines
gen = pipeline("text-generation") # will use gpt2 and may download a new model
set_seed(1234)
lines[0]
gen(lines[0])
pprint(_) # pretty print
pprint(gen(lines[0], max_length=20))
pprint(gen(lines[0], num_return_sequences=3, max_length=20))
def wrap(x):
  return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)
out = gen(lines[0], max_length=30)
print(wrap(out[0]['generated_text']))
prev = 'Two roads diverged in a yellow wood, including one that blocked the' + \
  ' road leading to another intersection in the middle of the city.'
out = gen(prev + '\n' + lines[2], max_length=60)
print(wrap(out[0]['generated_text']))
prev = 'Two roads diverged in a yellow wood, including one that blocked the' + \
  ' road leading to another intersection in the middle of the city.\n' + \
  'And be one traveler, long I stood in front of the burning wreckage. ' + \
  'That\'s what I had written on Twitter a few minutes ago.'
out = gen(prev + '\n' + lines[4], max_length=90)
print(wrap(out[0]['generated_text']))
# Exercise: think of ways you'd apply GPT-2 and text generation in the real
# world and as a business / startup.
# This model is not trained for poem anyway
prompt = "Neural networks with attention have been used with great success"  + \
  " in natural language processing."
out = gen(prompt, max_length=300)
print(wrap(out[0]['generated_text']))
```
- Results don't make sense but are quite readable
- Try bakery, coffee, waffle, and so on

12. Masked Language Modeling (Article Spinner)
- Similar to autoregressive but also incoming texts affect as well
  - Bi-directions
- Text generation pipeline or autoregressive language model is done by GPT
- Masked language modeling is done by BERT
  - Bidirectional Encoder Representations from Transformers
- Article spinning
  - SEO = techniques to improve search engine rankings
  - Creates contents which users want
  - Article spinning: changes words which are appropriate, avoiding plagiarism
- Code preparation
```py
from transformers import pipeline
mlm = pipeline("fill-mask")
mlm("The cat <mask> over the box")
```
- Autoencoding language model
  - Neural nets trying to reproduce their input
  - Recommender systems, pretraining
  - Denoising autoencoder

13. Masked Lanauage Modeling (Article Spinner) in Python
```py
# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv
!pip install transformers
import numpy as np
import pandas as pd
import textwrap
from pprint import pprint
from transformers import pipeline
df = pd.read_csv('bbc_text_cls.csv')
labels = set(df['labels'])
labels
# Pick a label
label = 'business'
texts = df[df['labels'] == label]['text']
texts.head()
np.random.seed(1234)
i = np.random.choice(texts.shape[0])
doc = texts.iloc[i]
print(textwrap.fill(doc, replace_whitespace=False, fix_sentence_endings=True))
mlm = pipeline('fill-mask')
mlm('Bombardier chief to leave <mask>')
text = 'Shares in <mask> and plane-making ' + \
  'giant Bombardier have fallen to a 10-year low following the departure ' + \
  'of its chief executive and two members of the board.'
mlm(text)
```

14. Named Entity Recognition (NER)
- Identify people, places, and companies in the documents
- Parts of speech tagging (many-to-many)
- Data is highly imbalanced (mostly tagged as 'O')
- IOB format
  - B-PER (Beginning of a person chunk)
  - I-PER (Inside of a person chunk)
  - O (Outside)
```py
from transformers import pipeline
ner = pipeline("ner", aggregation_strategy='simple', device=0)
```

15. Named Entity Recognition (NER) in Python
```py
!pip install transformers
from transformers import pipeline
ner = pipeline("ner", aggregation_strategy='simple', device=0)
import pickle
!wget -nc https://lazyprogrammer.me/course_files/nlp/ner_train.pkl
!wget -nc https://lazyprogrammer.me/course_files/nlp/ner_test.pkl
with open('ner_train.pkl', 'rb') as f:
  corpus_train = pickle.load(f)
with open('ner_test.pkl', 'rb') as f:
  corpus_test = pickle.load(f)
corpus_test
inputs = []
targets = []
for sentence_tag_pairs in corpus_test:
  tokens = []
  target = []
  for token, tag in sentence_tag_pairs:
    tokens.append(token)
    target.append(tag)
  inputs.append(tokens)
  targets.append(target)
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()
detokenizer.detokenize(inputs[9])
ner(detokenizer.detokenize(inputs[9]))
def compute_prediction(tokens, input_, ner_result):
  # map hugging face ner result to list of tags for later performance assessment
  # tokens is the original tokenized sentence
  # input_ is the detokenized string
  predicted_tags = []
  state = 'O' # keep track of state, so if O --> B, if B --> I, if I --> I
  current_index = 0
  for token in tokens:
    # find the token in the input_ (should be at or near the start)
    index = input_.find(token)
    assert(index >= 0)
    current_index += index # where we are currently pointing to
    # print(token, current_index) # debug
    # check if this index belongs to an entity and assign label
    tag = 'O'
    for entity in ner_result:
      if current_index >= entity['start'] and current_index < entity['end']:
        # then this token belongs to an entity
        if state == 'O':
          state = 'B'
        else:
          state = 'I'
        tag = f"{state}-{entity['entity_group']}"
        break
    if tag == 'O':
      # reset the state
      state = 'O'
    predicted_tags.append(tag)
    # remove the token from input_
    input_ = input_[index + len(token):]
    # update current_index
    current_index += len(token)
  # sanity check
  # print("len(predicted_tags)", len(predicted_tags))
  # print("len(tokens)", len(tokens))
  assert(len(predicted_tags) == len(tokens))
  return predicted_tags
input_ = detokenizer.detokenize(inputs[9])
ner_result = ner(input_)
ptags = compute_prediction(inputs[9], input_, ner_result)
from sklearn.metrics import accuracy_score, f1_score
accuracy_score(targets[9], ptags)
for targ, pred in zip(targets[9], ptags):
  print(targ, pred)
# get detokenized inputs to pass into ner model
detok_inputs = []
for tokens in inputs:
  text = detokenizer.detokenize(tokens)
  detok_inputs.append(text)
# 17 min on CPU, 3 min on GPU
ner_results = ner(detok_inputs)
predictions = []
for tokens, text, ner_result in zip(inputs, detok_inputs, ner_results):
  pred = compute_prediction(tokens, text, ner_result)
  predictions.append(pred)
# https://stackoverflow.com/questions/11264684/flatten-list-of-lists
def flatten(list_of_lists):
  flattened = [val for sublist in list_of_lists for val in sublist]
  return flattened
# flatten targets and predictions
flat_predictions = flatten(predictions)
flat_targets = flatten(targets)
accuracy_score(flat_targets, flat_predictions)
#0.9916563354782848
f1_score(flat_targets, flat_predictions, average='macro')
```

16. Text Summarization
- 2 Types of summarization
  - Extractive vs abstractive
  - Extractive summaries consists of text taken from the original document
  - Abstractive summarys can contain novel sequences of words
  - seq2seq transformers enables the abstractive summarization
```py
summarizer = pipeline("summarization")
summarizer(my_long_text)
```

17. Text Summarization in Python
```py
# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv
!pip install transformers
import pandas as pd
import numpy as np
import textwrap
from transformers import pipeline
df = pd.read_csv('bbc_text_cls.csv')
doc = df[df.labels == 'business']['text'].sample(random_state=42)
def wrap(x):
  return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)
print(wrap(doc.iloc[0]))
summarizer = pipeline("summarization") # will download a new model, distilbart
summarizer(doc.iloc[0].split("\n", 1)[1])
def print_summary(doc):
  result = summarizer(doc.iloc[0].split("\n", 1)[1])
  print(wrap(result[0]['summary_text']))
print_summary(doc)
doc = df[df.labels == 'entertainment']['text'].sample(random_state=123)
print(wrap(doc.iloc[0]))
print_summary(doc) # some redundancy or error found. 
```

18. Neural Machine Translation
- Convert phrases from one language to another
```py
translator = pipeline('translation', 
  model = 'Helsink-NLP/opus-mt-en-es')
translator("I loke eggs and ham")
```
- Find models at huggingface.co/models
- Translation evaluation
  - BLEU (Bilingual Evaluation Understudy) score is the most popular metric
    - 3 facts:
    - Prediction is compared with multiple reference texts
    - Is a value b/w 0 and 1
    - Looks at precision of n-grams (n=1,2,3,4)

19. Neural Machine Translation in Python
```py
!wget -nc http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
!unzip -nq spa-eng.zip
!head spa-eng/spa.txt
# compile eng-spa translations
eng2spa = {}
for line in open('spa-eng/spa.txt'):
  line = line.rstrip()
  eng, spa = line.split("\t")
  if eng not in eng2spa:
    eng2spa[eng] = []
  eng2spa[eng].append(spa)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
tokenizer.tokenize('¿Qué me cuentas?'.lower())
tokens = tokenizer.tokenize('¿Qué me cuentas?'.lower())
sentence_bleu([tokens], tokens)
sentence_bleu([['hi']], ['hi']) # 4-gram warning as there is only one token
smoother = SmoothingFunction()
sentence_bleu(['hi'], 'hi', smoothing_function=smoother.method4)
sentence_bleu(['hi there'.split()], 'hi there'.split())
sentence_bleu(['hi there friend'.split()], 'hi there friend'.split())
entence_bleu([[1,2,3,4]], [1,2,3,4])
eng2spa_tokens = {}
for eng, spa_list in eng2spa.items():
  spa_list_tokens = []
  for text in spa_list:
    tokens = tokenizer.tokenize(text.lower())
    spa_list_tokens.append(tokens)
  eng2spa_tokens[eng] = spa_list_tokens
!pip install transformers sentencepiece transformers[sentencepiece]
from transformers import pipeline
translator = pipeline("translation",
                      model='Helsinki-NLP/opus-mt-en-es', device=0)
translator("I like eggs and ham")
[{'translation_text': 'Me gustan los huevos y el jamón.'}]
eng_phrases = list(eng2spa.keys())
len(eng_phrases)
# 100,000 => we reduce as 1000 as it is too much
eng_phrases_subset = eng_phrases[20_000:21_000]
# 27 min for 10k phrases on GPU
translations = translator(eng_phrases_subset)
translations[0]
scores = []
for eng, pred in zip(eng_phrases_subset, translations):
  matches = eng2spa_tokens[eng]
  # tokenize translation
  spa_pred = tokenizer.tokenize(pred['translation_text'].lower())
  score = sentence_bleu(matches, spa_pred)
  scores.append(score)
import matplotlib.pyplot as plt
plt.hist(scores, bins=50);
import numpy as np
np.mean(scores)
np.random.seed(1)
def print_random_translation():
  i = np.random.choice(len(eng_phrases_subset))
  eng = eng_phrases_subset[i]
  print("EN:", eng)
  translation = translations[i]['translation_text']
  print("ES Translation:", translation)
  matches = eng2spa[eng]
  print("Matches:", matches)
print_random_translation()
print_random_translation()
```

20. Question Answering
- SQuAD (Stanford Question Answering Dataset)
  - Extractive question answering dataset
  - There is NO database of knowledge and the model has NOT memorized any facts about the world, it just picks the right part of the input
```py
from transformers import pipeline
qa = pipeline("question-answering")
ctx = "Today, I made a peanut butter sandwich"
q = "What did I put in my sandwich?"
qa(context = ctx, question = q)
```

21. Question Answering in Python
```py
!pip install transformers
from transformers import pipeline
qa = pipeline("question-answering")
context = "Today I went to the store to purchase a carton of milk."
question = "What did I buy?"
qa(context=context, question=question)
context = "Out of all the colors, I like blue the best."
question = "What is my favorite color?"
qa(context=context, question=question)
context = "Albert Einstein (14 March 1879 – 18 April 1955) was a " + \
  "German-born theoretical physicist, widely acknowledged to be one of the " + \
  "greatest physicists of all time. Einstein is best known for developing " + \
  "the theory of relativity, but he also made important contributions to " + \
  "the development of the theory of quantum mechanics. Relativity and " + \
  "quantum mechanics are together the two pillars of modern physics."
question = "When was Albert Einstein born?"
qa(context=context, question=question)
question = "What is peanut butter made of?"
qa(context=context, question=question)
# Check the score number
```

22. Zero-Shot Classification
- Zero-shot image classification is a computer vision task to classify images into one of several classes, without any prior training or knowledge of the classes.
  - Ref: https://huggingface.co/tasks/zero-shot-image-classification
```py
from transformers import pipeline
clf = piepeline("zero-shot-classification", device=0)
clf("This is a great movie", candidate_labels=["positive","negative"])
```

23. Zero-Shot Classification in Python
```py
# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv
!pip install transformers
from transformers import pipeline
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import textwrap
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
classifier = pipeline("zero-shot-classification", device=0)
classifier("This is a great movie", candidate_labels=["positive", "negative"])
# https://en.wikipedia.org/wiki/AMP-activated_protein_kinase
text = "Due to the presence of isoforms of its components, there are 12 " + \
  "versions of AMPK in mammals, each of which can have different tissue " + \
  "localizations, and different functions under different conditions. " + \
  "AMPK is regulated allosterically and by post-translational " + \
  "modification, which work together."
classifier(text, candidate_labels=["biology", "math", "geology"])
df = pd.read_csv('bbc_text_cls.csv')
len(df)
df.sample(frac=1).head()
labels = list(set(df['labels']))
print(textwrap.fill(df.iloc[1024]['text']))
df.iloc[1024]['labels']
classifier(df.iloc[1024]['text'], candidate_labels=labels)
# Takes about 55min
preds = classifier(df['text'].tolist(), candidate_labels=labels)
predicted_labels = [d['labels'][0] for d in preds]
df['predicted_labels'] = predicted_labels
print("Acc:", np.mean(df['predicted_labels'] == df['labels']))
# Convert prediction probs into an NxK matrix according to
# original label order
N = len(df)
K = len(labels)
label2idx = {v:k for k,v in enumerate(labels)}
probs = np.zeros((N, K))
for i in range(N):
  # loop through labels and scores in corresponding order
  d = preds[i]
  for label, score in zip(d['labels'], d['scores']):
    k = label2idx[label]
    probs[i, k] = score
int_labels = [label2idx[x] for x in df['labels']]
int_preds = np.argmax(probs, axis=1)
cm = confusion_matrix(int_labels, int_preds, normalize='true')
# Scikit-Learn is transitioning to V1 but it's not available on Colab
# The changes modify how confusion matrices are plotted
def plot_cm(cm):
  df_cm = pd.DataFrame(cm, index=labels, columns=labels)
  ax = sn.heatmap(df_cm, annot=True, fmt='.2g')
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Target")
plot_cm(cm)
f1_score(df['labels'], predicted_labels, average='micro')
roc_auc_score(int_labels, probs, multi_class='ovo')
```

24. Beginner's Corner Section Summary
- First task: sentiment analysis using Hugging face pipeline
  - Bag of words can perform well but loses information from word ordering
    - "Not interesting" will not negate "interesting"
  - Converting text into vectors/embeddings
- Many-to-many task
- Autoregressive/causal language model
- Autoencoding/masked language model (article spinner)
- Sequence-to-sequence tasks
- Summarization
- Translation
- Question Answering
- Zero-shot classification

25. Beginner Q&A: Can we use GPT-4 for everything?
- GPT-4 has been shown to be a general purpose AI
  - Not only a language model, it is tuned to follow instructions
  - Can classify text
  - Can summarize text
  - Can translate b/w languages
  - Can write/debug code
  - Can describe the contents of an image (multi-modal)
- Lessons from history
  - Recall
    - LSTM was very popular in 2010s
    - "Unreasonable Effectiveness of RNNs", Karpathy 2015
    - Folks used LSTM to predict stocks/time series
      - Not worked well
      - Transformers either
  - Hype train
- Pipelines are model-agnostic
  - Eventually pipelines will use GPT-4 in the backend

26. Suggestion Box

## Fine Tuning (Intermediate)

27. Fine-Tuning Section Introduction
- In the previous section, we used the model for inference/prediction
  - No training
- This section covers training (not just predict), and requires the understanding of the components of the pipeline
- Section Outline
  - Review text preprocessing (converts text into numbers)
  - Tokenization, token to integer mapping, padding
  - Instead of training from scratch, we do transfer learning/fine tuning

28. Text Preprocessing and Tokenization Review
- Steps
  - Tokenize
  - Map tokens into integers
  - Padding/truncation
- Tokenization
  - More than string split
  - Punctuation, question mark, ...
- Character level tokenization
  - "cat" => "c", "a", "t"
  - Use cases: name generators, language translation
- Subword tokenization
  - Split words into multiple tokens
  - run vs running vs runs
  - How do we choose subword boundaries?
  - Different models use different tokenization schemes
- Padding
  - Don't want to pad everything to be as long as the longest document
  - We should pad dynamically, relative to the current batch
  - Truncation
    - Transformers have a maximum sequence length: BERT limit=512 tokens, GPT-2 limit=1024 tokes

29. Models and Tokenizers
- What does a pipeline actually do?
  - Text processing -> tokenized data -> Model -> Numerical predictions -> Post processing
- In hugging face, tokenizer does all text processing in addition to tokenization
```py
from transformers import AutoTokenizer
checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer("hello world")
```
- CLS/SEP in transformers
- Multiple inputs
  - Needs padding/truncation
  - For padding, attention mask will be zero
- Using the model
```py
from transformers import AutoMaodelForSequenceClassification
# must be the same checkpoint as tokenizer
model = AutoModelForSequenceClassfication.from_pretrained(checkpoint)  
```
- Making predictions
```py
model_inputs = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
outputs = model(**model_inputs)
```
- Double asterisks (**): converts a dictionary into named arguments
  - Values of each key are sent as arguments

30. Models and Tokenizers in Python
```py
!pip install transformers
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer
#PreTrainedTokenizerFast(name_or_path='bert-base-uncased', vocab_size=30522, model_max_len=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})
tokenizer("hello world")
#{'input_ids': [101, 7592, 2088, 102], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
tokens = tokenizer.tokenize("hello world")
tokens
#['hello', 'world']
ids = tokenizer.convert_tokens_to_ids(tokens)
ids
#[7592, 2088]
tokenizer.convert_ids_to_tokens(ids)
#['hello', 'world']
tokenizer.decode(ids)
#'hello world'
ids = tokenizer.encode("hello world")
ids
#[101, 7592, 2088, 102]
tokenizer.convert_ids_to_tokens(ids)
#['[CLS]', 'hello', 'world', '[SEP]']
tokenizer.decode(ids)
#'[CLS] hello world [SEP]'
model_inputs = tokenizer("hello world")
model_inputs
#{'input_ids': [101, 7592, 2088, 102], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
data = [
  "I like cats.",
  "Do you like cats too?",
]
tokenizer(data)
#{'input_ids': [[101, 1045, 2066, 8870, 1012, 102], [101, 2079, 2017, 2066, 8870, 2205, 1029, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]}
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**model_inputs)
# this will fail as list object doesn't fit
model_inputs = tokenizer("hello world", return_tensors='pt')
model_inputs
#{'input_ids': tensor([[ 101, 7592, 2088,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1]])}
# the default was to create a binary classifier!
outputs = model(**model_inputs)
outputs
#SequenceClassifierOutput([('logits',
#                           tensor([[ 0.0272, -0.7987]], grad_fn=<AddmmBackward0>))])
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=3)
outputs = model(**model_inputs)
outputs
#SequenceClassifierOutput([('logits',
#                           tensor([[-0.0963,  0.2918, -0.1460]], grad_fn=<AddmmBackward0>))])
outputs.logits
#tensor([[-0.0963,  0.2918, -0.1460]], grad_fn=<AddmmBackward0>)
outputs['logits']
#tensor([[-0.0963,  0.2918, -0.1460]], grad_fn=<AddmmBackward0>)
outputs[0]
#tensor([[-0.0963,  0.2918, -0.1460]], grad_fn=<AddmmBackward0>)
outputs.logits.detach().cpu().numpy()
#array([[-0.09628202,  0.29179913, -0.14602423]], dtype=float32)
data = [
  "I like cats.",
  "Do you like cats too?",
]
model_inputs = tokenizer(data, return_tensors='pt')
model_inputs
# this will fail as size doesn't match. padding/truncation are necessary
model_inputs = tokenizer(
    data, padding=True, truncation=True, return_tensors='pt')
model_inputs
#{'input_ids': tensor([[ 101, 1045, 2066, 8870, 1012,  102,    0,    0],
#        [ 101, 2079, 2017, 2066, 8870, 2205, 1029,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0],
#        [1, 1, 1, 1, 1, 1, 1, 1]])}
model_inputs['input_ids']
#tensor([[ 101, 1045, 2066, 8870, 1012,  102,    0,    0],
#        [ 101, 2079, 2017, 2066, 8870, 2205, 1029,  102]])
model_inputs['attention_mask']
#tensor([[1, 1, 1, 1, 1, 1, 0, 0],
#        [1, 1, 1, 1, 1, 1, 1, 1]])
outputs = model(**model_inputs)
outputs
#SequenceClassifierOutput([('logits', tensor([[ 0.1343,  0.0503, -0.1545],
#                                   [ 0.1249,  0.0649, -0.1077]], grad_fn=<AddmmBackward0>))])
```

31. Transfer Learning & Fine-Tuning (pt1)
- Transfer learning
  - Not specific for transformer but general ML technique
  - Pretrained body -> Chop off one or more layers -> add one or more new layers -> train
    - Only new layers need to be trained

32. Transfer Learning & Fine-Tuning (pt2)
- Fine-tuning: adjust the parameters slowly and carefully a bit further to improve performance
- Example
  - Chop off the final layer and add new one
  - Train the new layer (transfer learning)
    - Pretrained body is fixed while new head is updated
  - Fine tuning
    - Pretrained body and new head are updated little
- Greedy Layer-Wise Pretraining

33. Transfer Learning & Fine-Tuning (pt3)
- Pretraining tasks
  - Supervised learning is not ideal
  - We use self-supervised learning or unsupervised learning
- Recap of possible pretraining tasks
  - Causal Language model (Autoregressive LM, GPT-like)
  - Masked language model (Autoencoding LM, BERT-like)
- Is pretraining practical?
  - LeNET of CNN has 60k parameters
  - GPT-3 has 175B parameters
  - Hence use the petrained model from big tech

34. Fine-Tuning Sentiment Analysis and the GLUE Benchmark
- Still need to learn:
  - How datawsets are represented
  - Trainer, TrainingArguments objects
  - Computing metrics
  - Saving and using the trained model
```py
from datasets import load_dataset
raw_datasets = load_dataset("amazon_polarity")  
raw_datasets = load_dataset("glue", "sst2") # data set/task set
```
- Glue benchmark
  - For NLP, and consists of multiple datasets/tasks
  - We will look at the SST-2 task (sentiment analysis)

35. Fine-Tuning Sentiment Analysis in Python
```py
!pip install transformers datasets
from datasets import load_dataset
import numpy as np
# https://huggingface.co/datasets/amazon_polarity
# takes a long time to process, you may want to try it yourself
dataset = load_dataset("amazon_polarity")
raw_datasets = load_dataset("glue", "sst2")
raw_datasets['train']
raw_datasets['train'][50000:50003]
raw_datasets['train'].features
from transformers import AutoTokenizer
# checkpoint = "bert-base-uncased"
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences = tokenizer(raw_datasets['train'][0:3]['sentence'])
from pprint import pprint
pprint(tokenized_sentences)
def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
from transformers import TrainingArguments
training_args = TrainingArguments(
  'my_trainer',
  evaluation_strategy='epoch',
  save_strategy='epoch',
  num_train_epochs=1,
)
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2)
!pip install torchinfo
from torchinfo import summary
# summary(model, input_size=(16,512), dtypes=['torch.IntTensor'], device='cpu')
summary(model)
params_before = []
for name, p in model.named_parameters():
  params_before.append(p.detach().cpu().numpy())
from transformers import Trainer
from datasets import load_metric
metric = load_metric("glue", "sst2")
metric.compute(predictions=[1, 0, 1], references=[1, 0, 0])
def compute_metrics(logits_and_labels):
  # metric = load_metric("glue", "sst2")
  logits, labels = logits_and_labels
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
from transformers import pipeline
newmodel = pipeline('text-classification', model='my_saved_model', device=0)
import json
config_path = 'my_saved_model/config.json'
with open(config_path) as f:
  j = json.load(f)
j['id2label'] = {0: 'negative', 1: 'positive'}
with open(config_path, 'w') as f:
  json.dump(j, f, indent=2)
!cat my_saved_model/config.json
newmodel = pipeline('text-classification', model='my_saved_model', device=0)
newmodel('This movie is great!')
params_after = []
for name, p in model.named_parameters():
  params_after.append(p.detach().cpu().numpy())
for p1, p2 in zip(params_before, params_after):
  print(np.sum(np.abs(p1 - p2)))
```

36. Fine-Tuning Transformers with Custom Dataset
- pip install transformers datasets
- wget -nc https://lazyprogrammer.me/course_files/AirlineTweets.csv
- pip install torchinfo
```py
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
df_ = pd.read_csv('AirlineTweets.csv')
df = df_[['airline_sentiment', 'text']].copy()
df['airline_sentiment'].hist() # note that data are unbalanced
target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
df['target'] = df['airline_sentiment'].map(target_map)
df2 = df[['text', 'target']]
df2.columns = ['sentence', 'label']
df2.to_csv('data.csv', index=None)
from datasets import load_dataset
raw_dataset = load_dataset('csv', data_files='data.csv')
split = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)
checkpoint = 'distilbert-base-cased'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)
tokenized_datasets = split.map(tokenize_fn, batched=True)
from transformers import AutoModelForSequenceClassification, \
  Trainer, TrainingArguments
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=3)
from torchinfo import summary
summary(model)
training_args = TrainingArguments(
  output_dir='training_dir',
  evaluation_strategy='epoch',
  save_strategy='epoch',
  num_train_epochs=3,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=64,
)
def compute_metrics(logits_and_labels):
  logits, labels = logits_and_labels
  predictions = np.argmax(logits, axis=-1)
  acc = np.mean(predictions == labels)
  f1 = f1_score(labels, predictions, average='macro')
  return {'accuracy': acc, 'f1': f1}
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
from transformers import pipeline
savedmodel = pipeline('text-classification',
                      model='training_dir/checkpoint-1282',
                      device=0)
split['test']
test_pred = savedmodel(split['test']['sentence'])
test_pred
def get_label(d):
  return int(d['label'].split('_')[1])
test_pred = [get_label(d) for d in test_pred]
print("acc:", accuracy_score(split['test']['label'], test_pred))
# acc: 0.8372040072859745
print("f1:", f1_score(split['test']['label'], test_pred, average='macro'))
#f1: 0.7828187972003485
def plot_cm(cm):
  classes = ['negative', 'positive', 'neutral']
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)
  ax = sn.heatmap(df_cm, annot=True, fmt='g')
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Target")
cm = confusion_matrix(split['test']['label'], test_pred, normalize='true')
plot_cm(cm)
```

37. Hugging Face AutoConfig

38. Fine-Tuning With Multiple Inputs (Textual Entailment)

39. Fine-Tuning transformers with Multiple Inputs in Python

40. Fine-Tuning Section Summary

## Named Entity Recognition (NER) and POS Tagging (Intermediate)

## Seq2Seq and Neural Machine Translation (Intermediate) 

## Question-Answering (Advanced) 

## Transformers and Attention Theory (Advanced) 

86. Theory section introduction
- Preparation
  - CNN/RNN
  - Stacking convolutional filters to make a convolution layer
- Outline
  - Basic self-attention
  - Scaled dot-product attention
  - Attention mask
  - Multi-head attention
  - More layers -> transformer block
  - Encoders (eg, BERT)
  - Decoders (eg, GPT)
  - Seq2Seq encoder-decoders

87. Basic Self-Attention
- How attention in Seq2Seq RNN works
- Attention weights from softmax

88. Self-Attention & Scaled Dot-Product Attention
- Self-attention
  - No learnable parameters!
- Not so fast
  - Attention(Q,K,V) = softmax(QK^T/\sqrt(d_k))V
- Why self-attention?
  - Different from seq2seq RNNs - for each output token, we wanted to know which input token to pay attention to

89. Attention Efficiency
- Every word must have an attention weight with every word
- T^2 computation since we must compute T^2 weights
- Still superior to RNNs
  - Can be parallelized: q_i can be computed independently of q_j
  - RNN is sequential
- Self attention can handle variable length sequences like RNN

90. Attention Mask
- Tokenizer always gives back input id and mask
- Why attention mask?
  - We use padding to make all sequences in batch have same T
  - Decoder training (autoaggressive)
- For padding, we want it to have zero weight
- Attention mask is applied before softmax, not after
  - `don't care` is -inf to yield exp(-inf) = 0

91. Multi-Head Attention
- Scaled dot-product attention vs Multi-head attention
  - Attenion(Q,K,V) = softmax(QK^T/\sqrt(d_k))V
  - MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
- Parallelization
  - Concatentates scaled dot-product attention and can be done in parallel

92. Transformer Block
- Multi-head attention is a building block in the Transformer block
- Stacking Transformer blocks gives us a Transformer
- Feedforward ANN
- Layer normalization
- Skip connections (aka residual connections)
  - More layers degrade performance

93. Positional Encodings (07:16)
- Does order matter in attention? No!

94. Encoder Architecture (06:23)
- BERT is an example of an encoder-only network

95. Decoder Architecture (10:58)
- Decoder pretraining objective: predict next token in a sequence
- Architecture is almost the same as encoder
- Main challenge is how we make the decoder predict the next token
- Causal self-attention
  - Transformer block uses self-attention with masking
- Many-to-many transformer
  - Every token will attend to every other token
  - Masking the attention weights ensures that each output only pays attention to the past

96. Encoder-Decoder Architecture (08:31)
97. BERT (04:52)
98. GPT (06:45)
99. GPT-2 (06:30)
100. GPT-3 (05:14)
101. ChatGPT (06:33)
102. GPT-4 (03:00)
103. Theory Section Summary (04:50)

## Implement Transformers From Scratch (Advanced) 
