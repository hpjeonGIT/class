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

93. Positional Encodings
- Does order matter in attention? No!

94. Encoder Architecture
- BERT is an example of an encoder-only network

95. Decoder Architecture
- Decoder pretraining objective: predict next token in a sequence
- Architecture is almost the same as encoder
- Main challenge is how we make the decoder predict the next token
- Causal self-attention
  - Transformer block uses self-attention with masking
- Many-to-many transformer
  - Every token will attend to every other token
  - Masking the attention weights ensures that each output only pays attention to the past

96. Encoder-Decoder Architecture
- V, K from encoder, Q from decoder
- V is size of T_input x d_v, K is size of T_input x d_k
- Q has size T_output x d_k
- Attention weights have shape of T_output x T_input
- Training now involves 2 inputs (each for encoder/decoder)
- One training sample is now a triple: encoder input, decoder input, decoder target
  - Teacher forcing: true target is passed in

97. BERT
- Bidirectional Encoder Representations from Transformers
- BERT-base: 12 blocks, 768 hidden size, 12 attention heads
- BERT-large: 24 blocks, 1024 hidden size, 16 attention heads
- Unsupervised pretraining + fine-tuning
- Pretraining tasks: masked language modeling + next sentence prediction

98. GPT
- Generative Pretrained Transformer
- Decoder only
- Pretrain with unsupervised task
- Reads documents from left to right, not 'bidrectional'
- All transfomer blocks use causal self-attention
- Fine-tuning with Auxiliary Language Model
  - L3(C) = L2(C) + \Lambda \* L1(C)
  - Overall Loss = Task-specific loss +  Balancing constant \* Language model loss
- 110 million parameters

99. GPT-2
- Surface level: more data, more parameters
- No fine tuning for downstream tasks
- Prompting
  - GPT-2 simply tries to guess "what comes next"
  - Prompt engineering: how to talk to an AI
- 1.5 billion parameters
- 1024 context size
- Still left-to-right

100. GPT-3
- More data/more parameters
- 176B parameters
- Doubles context size (1024 -> 2048)
- No fine-tuning
  - Inner loop
- GPT-2 primarily used zero-shot 
- GPT-3 uses zero-shot, one-host, few-shot

101. ChatGPT
- Based on GPT-3.5
- Used human trainer
  - Reinforcement learning from human feedback (RLHF)
  - Reward model comprised of ranked responses
  - Proximal Policy Optimization (PPO)
- Limitations
  - Can still make stuff up
  - Can decline to answer
  - Sensitive to input phrasing
  - Does not clarify ambiguous queries
  - DAN jailbreak - Do Anything Now
- Parameter count:1.3B, 6B, 175B
- Context size: 4096

102. GPT-4
- Multimodal: text and images
- Context size: 8192 tokens
  - Another version has 32k tokens
- Alignment and AI safety
  - Improved factuality and adherence to the desired behavior

103. Theory Section Summary
- 2 key components
  - Self-attention for capturing long-range dependencies
  - Unsupervised pretraining to leverage massive amounts of data
- Extend your knowledge
  - Longformer
  - Sparse transformer

## Implement Transformers From Scratch (Advanced) 

104. Implementation Section Introduction
- No numpy
- Will use PyTorch
- Will not implement tokenizer from scratch
  - We use HuggingFace
- No premade layers
- Section outline
  - 3 models: encoder, decoder, encoder-decoder

105. Encoder Implementation Plan & Outline
- Building modules in PyTorch
- HuggingFace tokenizer API

106. How to Implement Multihead Attention From Scratch
```py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
import numpy as np
import matplotlib.pyplot as plt
class MultiHeadAttention(nn.Module):
  def __init__(self, d_k, d_model, n_heads):
    super().__init__()
    # Assume d_v = d_k
    self.d_k = d_k
    self.n_heads = n_heads
    self.key = nn.Linear(d_model, d_k * n_heads)
    self.query = nn.Linear(d_model, d_k * n_heads)
    self.value = nn.Linear(d_model, d_k * n_heads)
    # final linear layer
    self.fc = nn.Linear(d_k * n_heads, d_model)
  def forward(self, q, k, v, mask=None):
    q = self.query(q) # N x T x (hd_k)
    k = self.key(k)   # N x T x (hd_k)
    v = self.value(v) # N x T x (hd_v)
    # Attention(Q,K,V) = softmax(QK^T/\sqrt(d_k))V
    N = q.shape[0]
    T = q.shape[1]
    # change the shape to: # instead of 3D tensor, we make 4D
    # (N, T, h, d_k) -> (N, h, T, d_k)
    # in order for matrix multiply to work properly
    q = q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
    k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
    v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
    # compute attention weights
    # (N, h, T, d_k) x (N, h, d_k, T) --> (N, h, T, T)
    attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
    if mask is not None:
      attn_scores = attn_scores.masked_fill(
          mask[:, None, None, :] == 0, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)    
    # compute attention-weighted values
    # (N, h, T, T) x (N, h, T, d_k) --> (N, h, T, d_k)
    A = attn_weights @ v
    # reshape it back before final linear layer
    A = A.transpose(1, 2) # (N, T, h, d_k)
    A = A.contiguous().view(N, T, self.d_k * self.n_heads) # (N, T, h*d_k)
    # projection
    return self.fc(A)
```    

107. How to Implement the Transformer Block From Scratch
```py
class TransformerBlock(nn.Module):
  def __init__(self, d_k, d_model, n_heads, dropout_prob=0.1):
    super().__init__()
    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)
    self.mha = MultiHeadAttention(d_k, d_model, n_heads)
    self.ann = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Linear(d_model * 4, d_model),
        nn.Dropout(dropout_prob),
    )
    self.dropout = nn.Dropout(p=dropout_prob)
  def forward(self, x, mask=None):
    x = self.ln1(x + self.mha(x, x, x, mask))
    x = self.ln2(x + self.ann(x))
    x = self.dropout(x)
    return x
```    

108. How to Implement Positional Encoding From Scratch
```py
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout_prob)
    position = torch.arange(max_len).unsqueeze(1)
    exp_term = torch.arange(0, d_model, 2) # damping solutions
    div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
  def forward(self, x):
    # x.shape: N x T x D
    x = x + self.pe[:, :x.size(1), :]
    return self.dropout(x)
```    

109. How to Implement Transformer Encoder From Scratch
```py
class Encoder(nn.Module):
  def __init__(self,
               vocab_size,
               max_len,
               d_k,
               d_model,
               n_heads,
               n_layers,
               n_classes,
               dropout_prob):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
    transformer_blocks = [
        TransformerBlock(
            d_k,
            d_model,
            n_heads,
            dropout_prob) for _ in range(n_layers)]
    self.transformer_blocks = nn.Sequential(*transformer_blocks)
    self.ln = nn.LayerNorm(d_model)
    self.fc = nn.Linear(d_model, n_classes)  
  def forward(self, x, mask=None):
    x = self.embedding(x)
    x = self.pos_encoding(x)
    for block in self.transformer_blocks:
      x = block(x, mask)
    # many-to-one (x has the shape N x T x D)
    x = x[:, 0, :]
    x = self.ln(x)
    x = self.fc(x)
    return x
model = Encoder(20_000, 1024, 16, 64, 4, 2, 5, 0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
x = np.random.randint(0, 20_000, size=(8, 512))
x_t = torch.tensor(x).to(device)

mask = np.ones((8, 512))
mask[:, 256:] = 0
mask_t = torch.tensor(mask).to(device)
y = model(x_t, mask_t)
y.shape
```
110. Train and Evaluate Encoder From Scratch
- pip install transformers datasets
```py
from transformers import AutoTokenizer, DataCollatorWithPadding
checkpoint = 'distilbert-base-cased' # no-segment embedding
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
from datasets import load_dataset
raw_datasets = load_dataset("glue", "sst2")
def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
from torch.utils.data import DataLoader
train_loader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,  # for train only, not for validation
    batch_size=32,
    collate_fn=data_collator
)
valid_loader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=32,
    collate_fn=data_collator
)
# check how it works
for batch in train_loader:
  for k, v in batch.items():
    print("k:", k, "v.shape:", v.shape)
  break
set(tokenized_datasets['train']['labels'])
tokenizer.vocab_size
# 28996
model = Encoder(
    vocab_size=tokenizer.vocab_size,
    max_len=tokenizer.max_model_input_sizes[checkpoint],
    d_k=16,
    d_model=64,
    n_heads=4,
    n_layers=2,
    n_classes=2,
    dropout_prob=0.1,
)
model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
from datetime import datetime
# A function to encapsulate the training loop
def train(model, criterion, optimizer, train_loader, valid_loader, epochs):
  train_losses = np.zeros(epochs)
  test_losses = np.zeros(epochs)
  for it in range(epochs):
    model.train()
    t0 = datetime.now()
    train_loss = 0
    n_train = 0
    for batch in train_loader:
      # move data to GPU
      batch = {k: v.to(device) for k, v in batch.items()}
      # zero the parameter gradients
      optimizer.zero_grad()
      # Forward pass
      outputs = model(batch['input_ids'], batch['attention_mask'])
      loss = criterion(outputs, batch['labels'])      
      # Backward and optimize
      loss.backward()
      optimizer.step()
      train_loss += loss.item()*batch['input_ids'].size(0)
      n_train += batch['input_ids'].size(0)
    # Get average train loss
    train_loss = train_loss / n_train    
    model.eval()
    test_loss = 0
    n_test = 0
    for batch in valid_loader:
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(batch['input_ids'], batch['attention_mask'])
      loss = criterion(outputs, batch['labels'])
      test_loss += loss.item()*batch['input_ids'].size(0)
      n_test += batch['input_ids'].size(0)
    test_loss = test_loss / n_test
    # Save losses
    train_losses[it] = train_loss
    test_losses[it] = test_loss    
    dt = datetime.now() - t0
    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
      Test Loss: {test_loss:.4f}, Duration: {dt}')  
  return train_losses, test_losses
train_losses, test_losses = train(
    model, criterion, optimizer, train_loader, valid_loader, epochs=4)
# Accuracy
model.eval()
n_correct = 0.
n_total = 0.
for batch in train_loader:
  # Move to GPU
  batch = {k: v.to(device) for k, v in batch.items()}
  # Forward pass
  outputs = model(batch['input_ids'], batch['attention_mask'])
  # Get prediction
  # torch.max returns both max and argmax
  _, predictions = torch.max(outputs, 1)
  # update counts
  n_correct += (predictions == batch['labels']).sum().item()
  n_total += batch['labels'].shape[0]
train_acc = n_correct / n_total
n_correct = 0.
n_total = 0.
for batch in valid_loader:
  # Move to GPU
  batch = {k: v.to(device) for k, v in batch.items()}
  # Forward pass
  outputs = model(batch['input_ids'], batch['attention_mask'])
  # Get prediction
  # torch.max returns both max and argmax
  _, predictions = torch.max(outputs, 1)
  # update counts
  n_correct += (predictions == batch['labels']).sum().item()
  n_total += batch['labels'].shape[0]
test_acc = n_correct / n_total
print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
```

111. How to Implement Causal Self-Attention From Scratch
```py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
import numpy as np
import matplotlib.pyplot as plt
class CausalSelfAttention(nn.Module):
  # similar to multihead attention loop but we need masks
  def __init__(self, d_k, d_model, n_heads, max_len):
    super().__init__()
    # Assume d_v = d_k
    self.d_k = d_k
    self.n_heads = n_heads
    self.key = nn.Linear(d_model, d_k * n_heads)
    self.query = nn.Linear(d_model, d_k * n_heads)
    self.value = nn.Linear(d_model, d_k * n_heads)
    # final linear layer
    self.fc = nn.Linear(d_k * n_heads, d_model)
    # causal mask
    # make it so that diagonal is 0 too
    # this way we don't have to shift the inputs to make targets
    # Lower Triangle along diagonal component is 1
    cm = torch.tril(torch.ones(max_len, max_len))
    self.register_buffer(
        "causal_mask",
        cm.view(1, 1, max_len, max_len)
    )
  def forward(self, q, k, v, pad_mask=None):
    q = self.query(q) # N x T x (hd_k)
    k = self.key(k)   # N x T x (hd_k)
    v = self.value(v) # N x T x (hd_v)
    N = q.shape[0]
    T = q.shape[1]
    # change the shape to:
    # (N, T, h, d_k) -> (N, h, T, d_k)
    # in order for matrix multiply to work properly
    q = q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
    k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
    v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
    # compute attention weights
    # (N, h, T, d_k) x (N, h, d_k, T) --> (N, h, T, T)
    attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
    if pad_mask is not None:
      attn_scores = attn_scores.masked_fill(
          pad_mask[:, None, None, :] == 0, float('-inf'))
    attn_scores = attn_scores.masked_fill(
        self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)
    # compute attention-weighted values
    # (N, h, T, T) x (N, h, T, d_k) --> (N, h, T, d_k)
    A = attn_weights @ v
    # reshape it back before final linear layer
    A = A.transpose(1, 2) # (N, T, h, d_k)
    A = A.contiguous().view(N, T, self.d_k * self.n_heads) # (N, T, h*d_k)
    # projection
    return self.fc(A)
class TransformerBlock(nn.Module):
  def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
    super().__init__()
    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)
    self.mha = CausalSelfAttention(d_k, d_model, n_heads, max_len)
    self.ann = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Linear(d_model * 4, d_model),
        nn.Dropout(dropout_prob),
    )
    self.dropout = nn.Dropout(p=dropout_prob)
  def forward(self, x, pad_mask=None):
    x = self.ln1(x + self.mha(x, x, x, pad_mask))
    x = self.ln2(x + self.ann(x))
    x = self.dropout(x)
    return x
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout_prob)

    position = torch.arange(max_len).unsqueeze(1)
    exp_term = torch.arange(0, d_model, 2)
    div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
  def forward(self, x):
    # x.shape: N x T x D
    x = x + self.pe[:, :x.size(1), :]
    return self.dropout(x)
```

112. How to Implement a Transformer Decoder (GPT) From Scratch 
```py
class Decoder(nn.Module):
  def __init__(self,
               vocab_size,
               max_len,
               d_k,
               d_model,
               n_heads,
               n_layers,
               dropout_prob):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
    transformer_blocks = [
        TransformerBlock(
            d_k,
            d_model,
            n_heads,
            max_len,
            dropout_prob) for _ in range(n_layers)]
    self.transformer_blocks = nn.Sequential(*transformer_blocks)
    self.ln = nn.LayerNorm(d_model)
    self.fc = nn.Linear(d_model, vocab_size)  
  def forward(self, x, pad_mask=None):
    x = self.embedding(x)
    x = self.pos_encoding(x)
    for block in self.transformer_blocks:
      x = block(x, pad_mask)
    x = self.ln(x)
    x = self.fc(x) # many-to-many
    return x
model = Decoder(20_000, 1024, 16, 64, 4, 2, 0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
x = np.random.randint(0, 20_000, size=(8, 512)) # random mask
x_t = torch.tensor(x).to(device)
y = model(x_t)
y.shape
mask = np.ones((8, 512))
mask[:, 256:] = 0
mask_t = torch.tensor(mask).to(device)
y = model(x_t, mask_t)
y.shape
```

113. How to Train a Causal Language Model From Scratch
- pip install transformers datasets
```py
from transformers import AutoTokenizer, DataCollatorWithPadding
checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
from datasets import load_dataset
# we'll use the same dataset, just ignore the labels
raw_datasets = load_dataset("glue", "sst2")
def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence", "idx", "label"])
from torch.utils.data import DataLoader
train_loader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator
)
# HW: write valid_loader
# check how it works
for batch in train_loader:
  for k, v in batch.items():
    print("k:", k, "v.shape:", v.shape)
  break
model = Decoder(
    vocab_size=tokenizer.vocab_size,
    max_len=tokenizer.max_model_input_sizes[checkpoint],
    d_k=16,
    d_model=64,
    n_heads=4,
    n_layers=2,
    dropout_prob=0.1,
)
model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) # why ignore_index? To ignore padding tokens
optimizer = torch.optim.Adam(model.parameters())
from datetime import datetime
# A function to encapsulate the training loop
def train(model, criterion, optimizer, train_loader, epochs):
  train_losses = np.zeros(epochs)
  for it in range(epochs):
    model.train()
    t0 = datetime.now()
    train_loss = []
    for batch in train_loader:
      # move data to GPU
      batch = {k: v.to(device) for k, v in batch.items()}
      # zero the parameter gradients
      optimizer.zero_grad()
      # shift targets backwards
      targets = batch['input_ids'].clone().detach()
      targets = torch.roll(targets, shifts=-1, dims=1) # removes CLS token (?)
      targets[:, -1] = tokenizer.pad_token_id 
      # Forward pass
      outputs = model(batch['input_ids'], batch['attention_mask'])
      # outputs are N x T x V
      # but PyTorch expects N x V x T
      # print("outputs:", outputs)
      # print("targets:", targets)
      loss = criterion(outputs.transpose(2, 1), targets)
      # N, T, V = outputs.shape
      # loss = criterion(outputs.view(N * T, V), targets.view(N * T))        
      # Backward and optimize
      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())
    # Get train loss and test loss
    train_loss = np.mean(train_loss)
    # Save losses
    train_losses[it] = train_loss    
    dt = datetime.now() - t0
    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Duration: {dt}')  
  return train_losses
train_losses = train(
    model, criterion, optimizer, train_loader, epochs=15)
valid_loader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=1,
    collate_fn=data_collator
)
model.eval()
for batch in valid_loader:
  # move data to GPU
  batch = {k: v.to(device) for k, v in batch.items()}
  outputs = model(batch['input_ids'], batch['attention_mask'])
  break
outputs.shape
prediction_ids = torch.argmax(outputs, axis=-1)
tokenizer.decode(prediction_ids[0])
tokenizer.decode(torch.concat((batch['input_ids'][0, :5], prediction_ids[:, 4])))
# generate something
prompt = "it's"
tokenized_prompt = tokenizer(prompt, return_tensors='pt')
tokenized_prompt
outputs = model(
    tokenized_prompt['input_ids'][:, :-1].to(device),
    tokenized_prompt['attention_mask'][:, :-1].to(device))
outputs.shape
prediction_ids = torch.argmax(outputs[:, -1, :], axis=-1)
tokenizer.decode(prediction_ids[0])
# generate something
prompt = "it's a"
tokenized_prompt = tokenizer(prompt, return_tensors='pt')
# prepare inputs + get rid of SEP token at the end
input_ids = tokenized_prompt['input_ids'][:, :-1].to(device)
mask = tokenized_prompt['attention_mask'][:, :-1].to(device)
for _ in range(20):
  outputs = model(input_ids, mask)
  prediction_id = torch.argmax(outputs[:, -1, :], axis=-1)
  input_ids = torch.hstack((input_ids, prediction_id.view(1, 1)))
  mask = torch.ones_like(input_ids)
  if prediction_id == tokenizer.sep_token_id:
    break
tokenizer.decode(input_ids[0])
# reults will be different at everytime
```

114. Implement a Seq2Seq Transformer From Scratch for Language Translation (pt 1)
```py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
import numpy as np
import matplotlib.pyplot as plt
class MultiHeadAttention(nn.Module):
  def __init__(self, d_k, d_model, n_heads, max_len, causal=False):
    super().__init__()
    # Assume d_v = d_k
    self.d_k = d_k
    self.n_heads = n_heads
    self.key = nn.Linear(d_model, d_k * n_heads)
    self.query = nn.Linear(d_model, d_k * n_heads)
    self.value = nn.Linear(d_model, d_k * n_heads)
    # final linear layer
    self.fc = nn.Linear(d_k * n_heads, d_model)
    # causal mask
    # make it so that diagonal is 0 too
    # this way we don't have to shift the inputs to make targets
    self.causal = causal
    if causal:
      cm = torch.tril(torch.ones(max_len, max_len))
      self.register_buffer(
          "causal_mask",
          cm.view(1, 1, max_len, max_len)
      )
  def forward(self, q, k, v, pad_mask=None):
    q = self.query(q) # N x T x (hd_k)
    k = self.key(k)   # N x T x (hd_k)
    v = self.value(v) # N x T x (hd_v)
    N = q.shape[0]
    T_output = q.shape[1]
    T_input = k.shape[1]
    # change the shape to:
    # (N, T, h, d_k) -> (N, h, T, d_k)
    # in order for matrix multiply to work properly
    q = q.view(N, T_output, self.n_heads, self.d_k).transpose(1, 2)
    k = k.view(N, T_input, self.n_heads, self.d_k).transpose(1, 2)
    v = v.view(N, T_input, self.n_heads, self.d_k).transpose(1, 2)
    # compute attention weights
    # (N, h, T, d_k) x (N, h, d_k, T) --> (N, h, T, T)
    attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
    if pad_mask is not None:
      attn_scores = attn_scores.masked_fill(
          pad_mask[:, None, None, :] == 0, float('-inf'))
    if self.causal:
      attn_scores = attn_scores.masked_fill(
          self.causal_mask[:, :, :T_output, :T_input] == 0, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)    
    # compute attention-weighted values
    # (N, h, T, T) x (N, h, T, d_k) --> (N, h, T, d_k)
    A = attn_weights @ v
    # reshape it back before final linear layer
    A = A.transpose(1, 2) # (N, T, h, d_k)
    A = A.contiguous().view(N, T_output, self.d_k * self.n_heads) # (N, T, h*d_k)
    # projection
    return self.fc(A)
class EncoderBlock(nn.Module):
  def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
    super().__init__()
    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)
    self.mha = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False) # note that causal is False
    self.ann = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Linear(d_model * 4, d_model),
        nn.Dropout(dropout_prob),
    )
    self.dropout = nn.Dropout(p=dropout_prob)  
  def forward(self, x, pad_mask=None):
    x = self.ln1(x + self.mha(x, x, x, pad_mask))
    x = self.ln2(x + self.ann(x))
    x = self.dropout(x)
    return x
class DecoderBlock(nn.Module):
  def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
    super().__init__()
    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)
    self.ln3 = nn.LayerNorm(d_model)
    self.mha1 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=True)
    self.mha2 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
    self.ann = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Linear(d_model * 4, d_model),
        nn.Dropout(dropout_prob),
    )
    self.dropout = nn.Dropout(p=dropout_prob)  
  def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
    # self-attention on decoder input
    x = self.ln1(
        dec_input + self.mha1(dec_input, dec_input, dec_input, dec_mask))
    # multi-head attention including encoder output
    x = self.ln2(x + self.mha2(x, enc_output, enc_output, enc_mask))
    x = self.ln3(x + self.ann(x))
    x = self.dropout(x)
    return x
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout_prob)
    position = torch.arange(max_len).unsqueeze(1)
    exp_term = torch.arange(0, d_model, 2)
    div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
  def forward(self, x):
    # x.shape: N x T x D
    x = x + self.pe[:, :x.size(1), :]
    return self.dropout(x)
class Encoder(nn.Module):
  def __init__(self,
               vocab_size,
               max_len,
               d_k,
               d_model,
               n_heads,
               n_layers,
              #  n_classes,
               dropout_prob):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
    transformer_blocks = [
        EncoderBlock(
            d_k,
            d_model,
            n_heads,
            max_len,
            dropout_prob) for _ in range(n_layers)]
    self.transformer_blocks = nn.Sequential(*transformer_blocks)
    self.ln = nn.LayerNorm(d_model)
    # self.fc = nn.Linear(d_model, n_classes)  
  def forward(self, x, pad_mask=None):
    x = self.embedding(x)
    x = self.pos_encoding(x)
    for block in self.transformer_blocks:
      x = block(x, pad_mask)
    # many-to-one (x has the shape N x T x D)
    # x = x[:, 0, :]
    x = self.ln(x)
    # x = self.fc(x)
    return x
class Decoder(nn.Module):
  def __init__(self,
               vocab_size,
               max_len,
               d_k,
               d_model,
               n_heads,
               n_layers,
               dropout_prob):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
    transformer_blocks = [
        DecoderBlock(
            d_k,
            d_model,
            n_heads,
            max_len,
            dropout_prob) for _ in range(n_layers)]
    self.transformer_blocks = nn.Sequential(*transformer_blocks)
    self.ln = nn.LayerNorm(d_model)
    self.fc = nn.Linear(d_model, vocab_size)  
  def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
    x = self.embedding(dec_input)
    x = self.pos_encoding(x)
    for block in self.transformer_blocks:
      x = block(enc_output, x, enc_mask, dec_mask)
    x = self.ln(x)
    x = self.fc(x) # many-to-many
    return x
class Transformer(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
  def forward(self, enc_input, dec_input, enc_mask, dec_mask):
    enc_output = self.encoder(enc_input, enc_mask)
    dec_output = self.decoder(enc_output, dec_input, enc_mask, dec_mask)
    return dec_output
# test it
encoder = Encoder(vocab_size=20_000,
                  max_len=512,
                  d_k=16,
                  d_model=64,
                  n_heads=4,
                  n_layers=2,
                  dropout_prob=0.1)
decoder = Decoder(vocab_size=10_000,
                  max_len=512,
                  d_k=16,
                  d_model=64,
                  n_heads=4,
                  n_layers=2,
                  dropout_prob=0.1)
transformer = Transformer(encoder, decoder)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
encoder.to(device)
decoder.to(device)
xe = np.random.randint(0, 20_000, size=(8, 512))
xe_t = torch.tensor(xe).to(device)
xd = np.random.randint(0, 10_000, size=(8, 256))
xd_t = torch.tensor(xd).to(device)
maske = np.ones((8, 512))
maske[:, 256:] = 0
maske_t = torch.tensor(maske).to(device)
maskd = np.ones((8, 256))
maskd[:, 128:] = 0
maskd_t = torch.tensor(maskd).to(device)
out = transformer(xe_t, xd_t, maske_t, maskd_t)
out.shape
```

115. Implement a Seq2Seq Transformer From Scratch for Language Translation (pt 2)
- wget -nc https://lazyprogrammer.me/course_files/nlp3/spa.txt
- pip install transformers datasets sentencepiece sacremoses
```py
import pandas as pd
df = pd.read_csv('spa.txt', sep="\t", header=None)
df.head()
df = df.iloc[:30_000] # full data takes too long. Cost increases as quadratic
df.columns = ['en', 'es']
df.to_csv('spa.csv', index=None)
from datasets import load_dataset
raw_dataset = load_dataset('csv', data_files='spa.csv')
split = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)
from transformers import AutoTokenizer
model_checkpoint = "Helsinki-NLP/opus-mt-en-es"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
en_sentence = split["train"][0]["en"]
es_sentence = split["train"][0]["es"]
inputs = tokenizer(en_sentence)
targets = tokenizer(text_target=es_sentence)
tokenizer.convert_ids_to_tokens(targets['input_ids'])
max_input_length = 128
max_target_length = 128
def preprocess_function(batch):
  model_inputs = tokenizer(
    batch['en'], max_length=max_input_length, truncation=True)
  # Set up the tokenizer for targets
  labels = tokenizer(
    text_target=batch['es'], max_length=max_target_length, truncation=True)
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs
tokenized_datasets = split.map(
  preprocess_function,
  batched=True,
  remove_columns=split["train"].column_names,
) # mask is missing. We will add it later
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer)
batch = data_collator([tokenized_datasets["train"][i] for i in range(0, 5)])
batch.keys()
# check values of attention_mask, input_ids, labels
from torch.utils.data import DataLoader
train_loader = DataLoader(
  tokenized_datasets["train"],
  shuffle=True,
  batch_size=32,
  collate_fn=data_collator
)
valid_loader = DataLoader(
  tokenized_datasets["test"],
  batch_size=32,
  collate_fn=data_collator
)
# check how it works
for batch in train_loader:
  for k, v in batch.items():
    print("k:", k, "v.shape:", v.shape)
  break
encoder = Encoder(vocab_size=tokenizer.vocab_size + 1,
                  max_len=512,
                  d_k=16,
                  d_model=64,
                  n_heads=4,
                  n_layers=2,
                  dropout_prob=0.1)
decoder = Decoder(vocab_size=tokenizer.vocab_size + 1,
                  max_len=512,
                  d_k=16,
                  d_model=64,
                  n_heads=4,
                  n_layers=2,
                  dropout_prob=0.1)
transformer = Transformer(encoder, decoder)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
encoder.to(device)
decoder.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(transformer.parameters())
```

116. Implement a Seq2Seq Transformer From Scratch for Language Translation (pt 3)
```py
from datetime import datetime
# A function to encapsulate the training loop
def train(model, criterion, optimizer, train_loader, valid_loader, epochs):
  train_losses = np.zeros(epochs)
  test_losses = np.zeros(epochs)
  for it in range(epochs):
    model.train()
    t0 = datetime.now()
    train_loss = []
    for batch in train_loader:
      # move data to GPU (enc_input, enc_mask, translation)
      batch = {k: v.to(device) for k, v in batch.items()}
      # zero the parameter gradients
      optimizer.zero_grad()
      enc_input = batch['input_ids']
      enc_mask = batch['attention_mask']
      targets = batch['labels']
      # shift targets forwards to get decoder_input
      dec_input = targets.clone().detach()
      dec_input = torch.roll(dec_input, shifts=1, dims=1)
      dec_input[:, 0] = 65_001
      # also convert all -100 to pad token id
      dec_input = dec_input.masked_fill(
          dec_input == -100, tokenizer.pad_token_id)
      # make decoder input mask
      dec_mask = torch.ones_like(dec_input)
      dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id, 0)
      # Forward pass
      outputs = model(enc_input, dec_input, enc_mask, dec_mask)
      loss = criterion(outputs.transpose(2, 1), targets)        
      # Backward and optimize
      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())
    # Get train loss and test loss
    train_loss = np.mean(train_loss)
    model.eval()
    test_loss = []
    for batch in valid_loader:
      batch = {k: v.to(device) for k, v in batch.items()}
      enc_input = batch['input_ids']
      enc_mask = batch['attention_mask']
      targets = batch['labels']
      # shift targets forwards to get decoder_input
      dec_input = targets.clone().detach()
      dec_input = torch.roll(dec_input, shifts=1, dims=1)
      dec_input[:, 0] = 65_001
      # change -100s to regular padding
      dec_input = dec_input.masked_fill(
          dec_input == -100, tokenizer.pad_token_id)
      # make decoder input mask
      dec_mask = torch.ones_like(dec_input)
      dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id, 0)
      outputs = model(enc_input, dec_input, enc_mask, dec_mask)
      loss = criterion(outputs.transpose(2, 1), targets)
      test_loss.append(loss.item())
    test_loss = np.mean(test_loss)
    # Save losses
    train_losses[it] = train_loss
    test_losses[it] = test_loss    
    dt = datetime.now() - t0
    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
      Test Loss: {test_loss:.4f}, Duration: {dt}')  
  return train_losses, test_losses
train_losses, test_losses = train(
    transformer, criterion, optimizer, train_loader, valid_loader, epochs=15)
# try it out
input_sentence = split['test'][10]['en']
input_sentence
dec_input_str = '<s>'
dec_input = tokenizer(text_target=dec_input_str, return_tensors='pt')
dec_input
# We'll ignore the final 0 ( </s> )
enc_input.to(device)
dec_input.to(device)
output = transformer(
    enc_input['input_ids'],
    dec_input['input_ids'][:, :-1],
    enc_input['attention_mask'],
    dec_input['attention_mask'][:, :-1],
)
enc_output = encoder(enc_input['input_ids'], enc_input['attention_mask'])
enc_output.shape
dec_output = decoder(
    enc_output,
    dec_input['input_ids'][:, :-1],
    enc_input['attention_mask'],
    dec_input['attention_mask'][:, :-1],
)
dec_output.shape
torch.allclose(output, dec_output)
dec_input_ids = dec_input['input_ids'][:, :-1]
dec_attn_mask = dec_input['attention_mask'][:, :-1]
for _ in range(32):
  dec_output = decoder(
      enc_output,
      dec_input_ids,
      enc_input['attention_mask'],
      dec_attn_mask,
  )
  # choose the best value (or sample)
  prediction_id = torch.argmax(dec_output[:, -1, :], axis=-1)
  # append to decoder input
  dec_input_ids = torch.hstack((dec_input_ids, prediction_id.view(1, 1)))
  # recreate mask
  dec_attn_mask = torch.ones_like(dec_input_ids)
  # exit when reach </s>
  if prediction_id == 0:
    break
tokenizer.decode(dec_input_ids[0])
def translate(input_sentence):
  # get encoder output first
  enc_input = tokenizer(input_sentence, return_tensors='pt').to(device)
  enc_output = encoder(enc_input['input_ids'], enc_input['attention_mask'])
  # setup initial decoder input
  dec_input_ids = torch.tensor([[65_001]], device=device)
  dec_attn_mask = torch.ones_like(dec_input_ids, device=device)
  # now do the decoder loop
  for _ in range(32):
    dec_output = decoder(
        enc_output,
        dec_input_ids,
        enc_input['attention_mask'],
        dec_attn_mask,
    )
    # choose the best value (or sample)
    prediction_id = torch.argmax(dec_output[:, -1, :], axis=-1)
    # append to decoder input
    dec_input_ids = torch.hstack((dec_input_ids, prediction_id.view(1, 1)))
    # recreate mask
    dec_attn_mask = torch.ones_like(dec_input_ids)
    # exit when reach </s>
    if prediction_id == 0:
      break  
  translation = tokenizer.decode(dec_input_ids[0, 1:])
  print(translation)
translate("How are you?")
# ¿Cómo estáis?
```

117. Implementation Section Summary

