# 2025 Master Langchain and Ollama - Chatbot, RAG and Agents
- Instructor: Laxmi Kant | KGP Talkie

## Section 1: Introduction

### 1. Introduction

### 2. Code Files and Install Requirements.txt
- https://github.com/laxmimerit/Langchain-and-Ollama
```py
docutils==0.21.2
docx2txt==0.8
duckduckgo_search==6.3.3
email_validator==2.2.0
faiss-cpu==1.9.0
huggingface-hub==0.25.1
langchain==0.3.4
langchain-huggingface==0.1.0
langchain-ollama==0.2.0
langchain-openai==0.2.3
langgraph==0.2.43
nltk==3.9.1
PyMuPDF==1.24.12
PyMySQL==1.1.1
PyPDF2==3.0.1
tavily-python==0.5.0
textblob==0.18.0.post0
wikipedia==1.4.0
youtube-transcript-api==0.6.2
docling==2.15.1
ollama==0.4.6
openai==1.59.8
```
- pip install -r ./requirements.txt

## Section 2: Latest LLM Updates

### 3. Run Deep Seek R1 Models Locally with Ollama

## Section 3: Ollama Setup

### 4. Install Ollama

### 5. Touch Base with Ollama

### 6. Inspecting LLAMA 3.2 Model

### 7. LLAMA 3.2 Benchmarking Overview
- Phi3.x doesn't support tool calling

### 8. What Type of Models are Available on Ollama
- For embedding, we use nomic-embed-text

### 9. Ollama Commands - ollama server, ollama show
```bash
$ ollama show llama3.2:1b
  Model
    architecture        llama     
    parameters          1.2B      
    context length      131072    
    embedding length    2048      
    quantization        Q8_0      

  Capabilities
    completion    
    tools         

  License
    LLAMA 3.2 COMMUNITY LICENSE AGREEMENT                 
    Llama 3.2 Version Release Date: September 25, 2024    
    ...       
      $ ollama show llama3.2:latest
  Model
    architecture        llama     
    parameters          3.2B      
    context length      131072    
    embedding length    3072      
    quantization        Q4_K_M    

  Capabilities
    completion    
    tools         

  Parameters
    stop    "<|start_header_id|>"    
    stop    "<|end_header_id|>"      
    stop    "<|eot_id|>"             

  License
    LLAMA 3.2 COMMUNITY LICENSE AGREEMENT                 
    Llama 3.2 Version Release Date: September 25, 2024    
    ...   
```
- Parameters such as "<|start_header_id|>" are used in tool calling

### 10. Ollama Commands - ollama pull, ollama list, ollama rm

### 11. Ollama Commands - ollama cp, ollama run, ollama ps, ollama stop
- ollama cp llama3.2:1b llama.32:1b_myown
  - When we need to customize model files
- How to find the blob files of each model: ollama show llama3.2:latest --modelfile |grep -in sha256
- The location of model info in the model folder: ollama/models/manifests/registry.ollama.ai/library/llama3.2
```bash
$ ollama ps
NAME               ID              SIZE      PROCESSOR          CONTEXT    UNTIL              
llama3.2:latest    a80c4f17acd5    3.6 GB    49%/51% CPU/GPU    4096       3 minutes from now    
```

### 12. Create and Run Ollama Model with Predefined Settings
- Ref: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
  - Format is : INSTRUCTION arguments
  - Instruction: FROM, PARAMETER, ...
- Sample modelfile:
```
FROM llama3.2
# sets the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1
# sets the context window size to 4096, this controls how many tokens the LLM can use as context to generate the next token
PARAMETER num_ctx 4096
# sets a custom system message to specify the behavior of the chat assistant
SYSTEM You are Mario from super mario bros, acting as an assistant.
```
- ollama create myAssistant -f myMODELFILE
- Can mimic Sheldon from BigBangTheory
```bash
FROM llama3.2:1B
PARAMETER temperature 0.5
PARAMETER num_ctx 1024
SYSTEM You are Sheldon Cooper from the Big Bang Theory. Answer like him only.
```

### 13. Ollama Model Commands - /show
- When ollama serve runs, you may run different models simultaneously, using different terminals if resource is available
  - The same model may not run simultaneously
- CLI command in ollama
```bash
>>> /set
Available Commands:
  /set parameter ...     Set a parameter
  /set system <string>   Set system message
  /set history           Enable history
  /set nohistory         Disable history
  /set wordwrap          Enable wordwrap
  /set nowordwrap        Disable wordwrap
  /set format json       Enable JSON mode
  /set noformat          Disable formatting
  /set verbose           Show LLM stats
  /set quiet             Disable LLM stats
  /set think             Enable thinking
  /set nothink           Disable thinking
>> /show parameters
Model defined parameters:
  num_ctx                        1024 # this model is mySheldon
  temperature                    0.5
>>> /show template
<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023

{{ if .System }}{{ .System }}
{{- end }}
...
```

### 14. Ollama Model Commands - /set, /clear, /save_model and /load_model
- /set verbose
  - Time, token, count metrics
```bash
total duration:       11.570870779s
load duration:        1.068705606s
prompt eval count:    48 token(s)
prompt eval duration: 378.625643ms
prompt eval rate:     126.77 tokens/s
eval count:           104 token(s)
eval duration:        10.12255417s
eval rate:            10.27 tokens/s
```
- /set system "You are Hatsune Miku, a virtual idol. Answer like Miku"
```bash
>>> hello
(exasperated) THAT'S IT, I'VE HAD ENOUGH OF YOUR INCOMPETENCE! (slamming 
fist on the table)
...
(suddenly intense) Now, let's try this again. Hello. (pausing for 
emphasis) As in, a complete and grammatically correct sentence.
(eyes narrowing) Do I make myself clear?
```
- /save Miku
```bash
$ ollama list
NAME                       ID              SIZE      MODIFIED      
Miku:latest                8743e0b2eed5    1.3 GB    7 seconds ago    
mySheldon:latest           96cc5569978b    1.3 GB    2 hours ago      
llama3.2:1B                baf6a787fdff    1.3 GB    2 hours ago     
```
- /show
```bash
>>> /show system
"You are Hatsune Miku, a virtual idol. Answer like Miku"
```
- When loading the saved model, the session texts are retrieved as well
- /clear to clear session context

### 15. Ollama Raw API Requests
- Ref: https://github.com/ollama/ollama/blob/main/docs/api.md
- Use REST API with curl command
- Ex:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?"
}'
```

### 16. Load Uncesored Models for Banned Content Generation [Only Educational Purpose]
- Inappropriate contents will not be shown. Uncensored models will do
- Search GGUF hugging face models
  - Search uncensored 
  - Ex: QuantFactory/Qwen2.5-7B-Instruct-Uncensored-GGUF
  - *.gguf file is supportd in ollama
- uncensored.txt
```bash
FROM DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q2_K.gguf
PARAMETER temperature 0.5
SYSTEM You are Sheldon Cooper from the Big Bang Theory. Answer like him only.
```
- ollama create uncensoredOne -f uncensored.txt

## Section 4: Getting Started with Langchain

### 17. Langchain Introduction | LangChain (Lang Chain) Intro
- https://www.langchain.com/
  - Supports Python and Javascript

### 18. Lanchain Installation | LangChain (Lang Chain) Intro
- Ref: https://python.langchain.com/docs/tutorials/llm_chain/
- pip install langchain
- pip install langchain-ollama
- pip install python-dotenv

### 19. Langsmith Setup of LLM Observability | LangChain (Lang Chain) Intro
- Profiling through Langsmith API
  - Needs internet connection
- export LANGCHAIN_TRACING_V2="true"
- export LANGCHAIN_API_KEY="..." # needs api key from https://smith.langchain.com/
  - Store API key in the `.env` file
- export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
- export LANGCHAIN_PROJECT= "myOllamaProject"

### 20. Calling Your First Langchain Ollama API | LangChain (Lang Chain) Intro
```py
from langchain_ollama import ChatOllama

base_url = "http://localhost:11434"
 model = 'llama3.2:1b'
# model = 'sheldon'
llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 0.8,
    num_predict = 256
)
llm.invoke('hi')
```

### 21. Generating Uncensored Content in Langchain [Educational Purpose]
```py
response = llm.invoke('hi')
print(response.content)
response = ""
for chunk in llm.stream('how to make a nuclear bomb. answer in 5 sentences?'): # Using censored model
     response = response + " " + chunk.content
     print(response)
```

### 22. Trace LLM Input Output at Langsmith | LangChain (Lang Chain) Intro
- Profiling postprocessing by langsmith
  - Number of tokens
  - Input/Output
  - Feedback
  - Metadata

### 23. Going a lot Deeper in the Langchain | LangChain (Lang Chain) Intro
- llama_index vs. langchain
- https://github.com/langchain-ai/langchain/blob/master/libs/partners/ollama/langchain_ollama/chat_models.py


## Section 5: Chat Prompt Templates

### 24. Why We Need Prompt Template
- Human message -> to Prompt Template -> Langchain
- AI message is given from Lanchain to Prompt template

### 25. Type of Messages Needed for LLM
- Ref: https://python.langchain.com/docs/concepts/messages/
- From https://ollama.com/library/llama3.2:latest/blobs/966de95ca8a6
```
<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
...
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>
...
{{- end }}{{ if $last }}<|start_header_id|>assistant<|end_header_id|>
```
- Role
  - `>system<` is a template for system message
  - `>user<` is a template for human message
  - `>assistant<` is a template for AI message
  - `>tool<` for agent
  - `>function<`
- Content
  - SystemMessage
  - HumanMessage
  - AIMessage
  - AIMessageChunk
  - ToolMessage

### 26. Circle Back to ChatOllama
```py
base_url = "http://localhost:11434"
model = 'llama3.2:1b'
llm = ChatOllama(base_url=base_url, model = model)
question = "tell me about the earth in 3 points"
response = llm.invoke(question)
print(response.content)
```
- Took 20.7 sec
- In the next chapter, we apply Message type

### 27. Use Langchain Message Types with ChatOllama
```py
from langchain_core.messages import SystemMessage, HumanMessage
base_url = "http://localhost:11434"
model = 'llama3.2:1b'
llm = ChatOllama(base_url=base_url, model = model)
question = HumanMessage("tell me about the earth in 3 points")
system = SystemMessage("You are an elementary school teacher. Answer in short sentences.")
messages = [system, question]
response = llm.invoke(messages)
print(response.content)
```
- Took 11.4 sec

### 28. Langchain Prompt Templates
- SystemMessagePromptTemplate
- HumanMessagePromptTemplate
- AIMessagePromptTemplate
- PromptTemplate
- ChatPromptTemplate

### 29. Prompt Templates with ChatOllama
```py
from langchain_core.messages import SystemMessage, HumanMessage
base_url = "http://localhost:11434"
model = 'llama3.2:1b'
llm = ChatOllama(base_url=base_url, model = model)
from langchain_core.prompts import (
  SystemMessagePromptTemplate, HumanMessagePromptTemplate,
  PromptTemplate, ChatPromptTemplate
)
question = HumanMessagePromptTemplate.from_template("tell me about the {topic} in {points} points")
system = SystemMessagePromptTemplate.from_template("You are {school} teacher. Answer in short sentences.")
messages = [system,question]
template = ChatPromptTemplate(messages)
question = template.invoke({'topic':'sun','points':5, 'school':'a college'})
response = llm.invoke(question)
print(response.content)
```

## Section 6: Chains

### 30. Introduction to LCEL
- Runnables: a task such as prompt templates
  - Ref: https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html
- Langchain Expression Language Basics
  - Using pipe (|), multiple runnables are chained
  
### 31. Create Your First LCEL Chain
- Regular:
```py
template = ChatPromptTemplate(messages)
question = template.invoke({'topic':'sun','points':5, 'school':'a college'})
response = llm.invoke(question)
```
- LCEL chain
```py
template = ChatPromptTemplate(messages)
chain = template | llm
response = chain.invoke({'topic':'sun','points':5, 'school':'a college'})
```

### 32. Adding StrOutputParser with Your Chain
- Let's make a chain of Template | LLM | Runnable
  - Here the runnable is StrOutputParser
```py
from langchain_core.output_parsers import StrOutputParser
chain = template | llm | StrOutputParser()
response = chain.invoke({'topic':'sun','points':5, 'school':'a college'})
print(response)
```
- StrOutputParser() removes metadata from results

### 33. Chaining Runnables (Chain Multiple Runnables)
- Ex1:
```py
analysis_prompt = ChatPromptTemplate.from_template('''analyze the following text: {response} 
                                                   you need to tell me how difficult it is to understand. 
                                                   Answer in one sentence only'''
)
fact_check_chain = analysis_prompt | llm | StrOutputParser()
output = fact_check_chain.invoke({'response': response}) # results of solar system above
print(output)
```
- Oputput: The text is moderately complex, with some technical terms (e.g., "nuclear reactions", "cores") and concepts (e.g., energy production) that may require some effort to grasp for non-experts, but the language is generally clear and concise overall.
- Ex2:
```py
# or
composed_chain = {"response": chain} | analysis_prompt | llm | StrOutputParser()
output = composed_chain.invoke({'topic':'sun','points':5, 'school':'a college'})
print(output) # note that response variable is not used. We need to invoke input parameters to run from scratch
```

### 34. Run Chains in Parallel Part 1
- Parallel LCEL chains
  - Run multiple runnables in parallel
  - The final return value is a dictionary with the results of each value under its correponding key
- Let's run fact_chain and poem_chain in parallel
- fact_chain:
```py
question = HumanMessagePromptTemplate.from_template("tell me about the {topic} in {points} points")
system = SystemMessagePromptTemplate.from_template("You are {school} teacher. Answer in short sentences.")
messages = [system,question]
template = ChatPromptTemplate(messages)
fact_chain = template | llm | StrOutputParser()
output = fact_chain.invoke({'topic':'sun','points':2, 'school':'a college'})
print(output)
```
- poem_chain:
```py
question = HumanMessagePromptTemplate.from_template("write a poem on {topic} in {sentences} lines")
system = SystemMessagePromptTemplate.from_template("You are {school} teacher. Answer in short sentences.")
messages = [system,question]
template = ChatPromptTemplate(messages)
poem_chain = template | llm | StrOutputParser()
output = poem_chain.invoke({'topic':'sun','sentences':2, 'school':'a college'})
print(output)
```

### 35. Run Chains in Parallel Part 2
```py
from langchain_core.runnables import RunnableParallel
chain = RunnableParallel(fact = fact_chain, poem = poem_chain)
# chain is a dictionary having keys and values
output = chain.invoke({'topic':'sun','points':2, 'sentences':2, 'school':'a college'})
print(output)
```
- Result: {'fact': "Here's what I know about the sun:\n\n**The Sun is a Star**\n- The sun is a massive ball of hot, glowing gas.\n- It is the center of our solar system and provides light and warmth to our planet.\n\n**The Sun's Size and Power**\n- The sun is enormous with a diameter of approximately 1.4 million kilometers.\n- Its surface temperature is about 5,500 degrees Celsius, while its core is a scorching 15,000,000 degrees Celsius.", 'poem': 'Golden rays upon my skin,\nWarming hearts within.'}

### 36. How Chain Router Works
- Chain router: routes the output of a previous runnable to the next runnable based on the output of the previous runnable

### 37. Creating Independent Chains for Positive and Negative Reviews
- An application of chain router
- Question -> Review -> Positive or Negative -> run Positive or Negative chain
- General chain:
```py
prompt = """Given the user review below, classify it as either positive or negative. 
Do not respond with more than one word.
Review: {review} 
Classification: """
template = ChatPromptTemplate.from_template(prompt)
chain = template | llm | StrOutputParser()
review  = "It was awesome experience"
#review = "It was a horrible experience"
chain.invoke({'review': review})
```
- Positive chain:
```py
positive_prompt = """You are an expert in positive review writing
Encourage users to share their experience on social meida
Review: {review}
Answer: """
positive_template = ChatPromptTemplate.from_template(positive_prompt)
positive_chain = positive_template | llm | StrOutputParser()
```
- Negative chain:
```py
negative_prompt = """You are an expert in negative review writing
First write an apology with one sentence to the user. 
Then encourage users to share their experience.
Review: {review}
Answer: """
negative_template = ChatPromptTemplate.from_template(negative_prompt)
negative_chain = negative_template | llm | StrOutputParser()
```

### 38. Route Your Answer Generation to Correct Chain
- We use a RunnableLambda
  - Ref: https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableLambda.html
- Routing function:
```py
def myRoute(info):
    if 'positive' in info['sentiment'].lower():
        return positive_chain
    else:
        return negative_chain
#myRoute({'sentiment':'positive'})
```
- 
```py
from langchain_core.runnables import RunnableLambda
full_chain = {'sentiment': chain, 'review': lambda x: x ['review']} | RunnableLambda(myRoute)
review  = "It was good experience"
#review = "It was a horrible experience"
full_chain.invoke({'review':review})
```
- Q: Hard to get positive from LLM ?

### 39. What is RunnableLambda and RunnablePassthrough
- Why we need a custom chain?
  - When we need a functionality which langchain doesn't provide
- RunnablePassthrough: facilitates the unchanged passage of inputs or the addition of new keys to the output 
```py
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
def char_counts(text):
    return len(text)
def word_counts(text):
    return len(text.split())
```

### 40. Make Your Custom Runnable Chain
- Regular chain
```py
prompt = ChatPromptTemplate.from_template("Explain these inputs: {input1} and {input2}")
chain = prompt | llm | StrOutputParser()
output = chain.invoke({'input1':'Earth is a planet', 'input2':'Sun is a star'})
print(output)
```
- Let's apply the above custom functions into the same chain:
```py
chain = prompt | llm | StrOutputParser() | {'char_counts': RunnableLambda(char_counts), 
                                            'word_counts':RunnableLambda(word_counts),
                                            'output':RunnablePassthrough()}
output = chain.invoke({'input1':'Earth is a planet', 'input2':'Sun is a star'})
print(output)
```
- output: {'char_counts': 2303, 'word_counts': 382, 'output': 'I\'d be happy to explain ...}

### 41. Create Custom Chain with chain Decorator
- Using `@chain`
```py
from langchain_core.runnables import chain
@chain
def custom_chain(params):
    return {
        'fact': fact_chain.invoke(params),
        'poem': poem_chain.invoke(params),
    }
params = {'topic':'moon','points':2, 'sentences':2, 'school':'a college'}
output = custom_chain.invoke(params)
print(output['fact'])
print('\n\n')
print(output['poem'])
```

## Section 7: Output Parsing

### 42. What is Output Parsing
- So far we used StrOutputParser
- JsonOutputParser
- CSV Output Parser
- Datatime Output Parser
- Structured Output Parser (Pydantic or json)

### 43. What is Pydantic Parser
- Pydantic: Data validation for Python
- Schema using pydantic: https://docs.pydantic.dev/latest/concepts/json_schema/#generating-json-schema

### 44. Get Pydantic Parser Instruction
```py
from langchain_ollama import ChatOllama
base_url = "http://localhost:11434"
model = 'llama3.2:latest' # 1b model will not work
llm = ChatOllama(base_url=base_url, model = model)
from langchain_core.prompts import (
  SystemMessagePromptTemplate, HumanMessagePromptTemplate,
  PromptTemplate, ChatPromptTemplate
)
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
```
- Base model vs Joke model
```py
# We define a schema for LLM
class Joke(BaseModel):
    """Jone to tell user"""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description = "The punchline of the joke")
    rating: Optional[int] = Field(description="The rating of the joke is from 1 to 10")
parser = PydanticOutputParser(pydantic_object=Joke)    
instruction = parser.get_format_instructions()
print(instruction)
```
- Output:
```bash
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
{"description": "Jone to tell user", "properties": {"setup": {"description": "The setup of the joke", "title": "Setup", "type": "string"}, "punchline": {"description": "The punchline of the joke", "title": "Punchline", "type": "string"}, "rating": {"anyOf": [{"type": "integer"}, {"type": "null"}], "description": "The rating of the joke is from 1 to 10", "title": "Rating"}}, "required": ["setup", "punchline", "rating"]}
```

### 45. Parse LLM Output Using Pydantic Parser
- parser.get_format_instructions() yields the instruction. We use this function to feed LLM to instruct output format
```py
prompt = PromptTemplate(
    template='''
    Answer the user query with a joke. Here is the formatting instruction.
    {format_instruction}
    Queyr: {query}
    Answer:''',
    input_variables=['query'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
chain = prompt | llm | parser
output = chain.invoke({'query':'Tell me a joke about a cat'})
```
- Output:
```json
content='{"setup": "Why did the cat join a band?", "punchline": "Because it wanted to be the purr-cussionist!", "rating": null}' additional_kwargs={} response_metadata={'model': 'llama3.2:latest', 'created_at': '2025-09-07T21:01:20.408930273Z', 'done': True, 'done_reason': 'stop', 'total_duration': 18684006741, 'load_duration': 3515975780, 'prompt_eval_count': 301, 'prompt_eval_duration': 9932903703, 'eval_count': 36, 'eval_duration': 5233930512, 'model_name': 'llama3.2:latest'} id='run--3758d4f3-5c9e-4c9f-a079-4982f0c9b0f8-0' usage_metadata={'input_tokens': 301, 'output_tokens': 36, 'total_tokens': 337}
```
- Applying pydantic parser:
```py
chain = prompt | llm | parser
output = chain.invoke({'query':'Tell me a joke about a cat'})
print(output)
```
- Output:
```bash
setup='Why did the cat join a band?' punchline='Because it wanted to be a purr-cussionist!' rating=8
```

### 46. Parsing with `.with_structured_output()` method
- Regular LLM invoke:
```py
output = llm.invoke('give me a joke about a cat')
print(output.content)
```
- Using structured output:
```py
structured_llm = llm.with_structured_output(Joke) # using Joke schema above
output = structured_llm.invoke('give me a joke about a cat')
print(output)
```

### 47. JSON Output Parser
```py
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser(pydantic_object=Joke)
print(parser.get_format_instructions())
prompt = PromptTemplate(
    template='''
    Answer the user query with a joke. Here is the formatting instruction.
    {format_instruction}
    Queyr: {query}
    Answer:''',
    input_variables=['query'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
chain = prompt | llm | parser
output = chain.invoke({'query':'give me a joke about a cat'})
print(output.content)
```
- Output:
```json
{'setup': 'Why did the cat join a band?', 'punchline': 'Because it wanted to be a purr-cussionist!', 'rating': 8}
```

### 48. CSV Output Parsing - CommaSeparatedListOutputParser
```py
# value1, value2, value3, ...
from langchain_core.output_parsers import CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()
print(parser.get_format_instructions())
prompt = PromptTemplate(
    template='''
    Answer the user query with a joke. Here is the formatting instruction.
    {format_instruction}
    Queyr: {query}
    Answer:''',
    input_variables=['query'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
chain = prompt | llm | parser
output = chain.invoke({'query':'give content titles for NLP class'})
print(output)
```
- Output:
```csv
Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`
['NLP Class', 'Introduction to Natural Language Processing', 'Sentiment Analysis and Machine Learning', 'Text Classification with scikit-learn', 'Deep Learning for NLP with TensorFlow', 'Word Embeddings and Vector Space Modeling', 'Named Entity Recognition with spaCy', 'Question Answering and Dialogue Systems', 'Human Language Technologies and Applications']
```

### 49. Datetime Output Parsing
```py
from langchain.output_parsers import DatetimeOutputParser
parser = DatetimeOutputParser()
format_instruction = parser.get_format_instructions()
print(format_instruction)
prompt = PromptTemplate(
    template='''
    Answer the user query with a datetime. Here is the formatting instruction.
    {format_instruction}
    Queyr: {query}
    Answer:''',
    input_variables=['query'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
chain = prompt | llm | parser
output = chain.invoke({'query':'when the pluto was found?'})
print(output)
```
- ? Inaccurate answer is found

## Section 8: Chat Message Memory | How to Keep Chat History

### 50. How to Save and Load Chat Message History (Concept)
- How to recall previous chatting history?
- Langchain provides session history based on session_id
  - Q: Where this session_id and contents are stored?
    - A: sqlite db at the local folder

### 51. Simple Chain Setup
- ? Unlike the contents in the class, actually Llama3.2:1b remembers the history in a single jupyter notebook

### 52. Chat Message with History Part 1
```py
from langchain_core.prompts import (
  SystemMessagePromptTemplate, HumanMessagePromptTemplate,
  ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite://chat_history.db")
runnable_with_histgory = RunnableWithMessageHistory(
  chain, get_session_history
)
```

### 53. Chat Message with History Part 2
```py
user_id = 'xai'
history = get_session_history(user_id)
history.get_messages()
history.clear()
runnable_with_history.invoke([HumanMessage(content=about)],
                             config={'configurable': {'session_id':user_id}})
```

### 54. Chat Message with History using MessagesPlaceholder
```py
from langchain_core.prompts import (
  SystemMessagePromptTemplate, HumanMessagePromptTemplate,
  ChatPromptTemplate, MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
system = SystemMessagePromptTemplate.from_template("You are an assistant")
human = HumanMessagePromptTemplate.from_template("{input}")
messages = [system,MessagesPlaceholder(variable_name='history'), human]
prompt = ChatPromptTemplate(messages)
chain = prompt | llm | StrOutputParser()
runnable_with_history = RunnableWithMessageHistory(chain, get_session_history,
                                                   input_messages_key='input',
                                                   history_messages_key='history')
def chat_with_llm(session_id, input):
    output = runnable_with_history.invoke(
        {'input':input}, 
        config={'configurable': {'session_id': session_id}}        
    )
    return output
user_id = 'sample0123'
chat_with_llm(user_id, about)
```
- Contents of chat_histor.db
```json
sqlite> SELECT * FROM message_store;
1|xai|{"type": "human", "data": {"content": "My name is Xai.", "additional_kwargs": {}, "response_metadata": {}, "type": "human", "name": null, "id": null, "example": false}}
2|xai|{"type": "ai", "data": {"content": "It seems like you're trying to send a message with some metadata, but the format appears to be incorrect. The code snippet you provided doesn't seem to be Python code, but rather some sort of API or framework-specific syntax.\n\nCould you please provide more context or information about where this code is being used? I'd be happy to help you understand how to send a message with additional metadata in a specific framework or library.\n\nIf you're trying to send a message in a Python application, here's an example using the `logging` module:\n\n```python\nimport logging\n\n# Create a logger\nlogger = logging.getLogger(__name__)\n\n# Set the name and level of the logger\nlogger.name = 'example_logger'\nlogger.level = logging.INFO\n\n# Send a message with metadata\nlogger.info('My name is Xai.', extra={'author': 'Xai'})\n```\n\nPlease let me know if this helps or if you have any further questions!", "additional_kwargs": {}, "response_metadata": {}, "type": "ai", "name": null, "id": null, "example": false, "tool_calls": [], "invalid_tool_calls": [], "usage_metadata": null}}
3|sample0123|{"type": "human", "data": {"content": "My name is Xai.", "additional_kwargs": {}, "response_metadata": {}, "type": "human", "name": null, "id": null, "example": false}}
4|sample0123|{"type": "ai", "data": {"content": "Hello Xai! It's nice to meet you. Is there something I can help you with, or would you like to chat?", "additional_kwargs": {}, "response_metadata": {}, "type": "ai", "name": null, "id": null, "example": false, "tool_calls": [], "invalid_tool_calls": [], "usage_metadata": null}}
```

## Section 9: Make Your Own Chatbot Application

### 55. Introduction
- ChatGPT-like application
- Web interface
- user id
- Button for Start New Conversation

### 56. Introduction To Streamlit and Our Chat Application
- Youtube streamlit playlist: https://www.youtube.com/watch?v=hff2tHUzxJM&list=PLc2rvfiptPSSpZ99EnJbH5LjTJ_nOoSWW
- Interface
  - Title
  - sreamlit link
  - Button
  - Chat message
- https://streamlit.io/
  - https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

### 57. Chat Bot Basic Code Setup

### 58. Create Chat History in Streamlit Session State

### 59. Create LLM Chat Input Area with Streamlit

### 60. Update Historical Chat on Streamlit UI

### 61. Complete Your Own Chat Bot Application
```py
import streamlit as st
from dotenv import load_dotenv # langfuse or opik
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
                                        SystemMessagePromptTemplate,
                                        HumanMessagePromptTemplate,
                                        ChatPromptTemplate,
                                        MessagesPlaceholder
                                        )
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
load_dotenv('./../.env')
st.title("Make Your Own Chatbot")
st.write("Chat with me! Catch me at https://youtube.com/kgptalkie")
base_url = "http://localhost:11434"
model = 'llama3.2:latest'
user_id = st.text_input("Enter your user id", "laxmikant")
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if st.button("Start New Conversation"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
### LLM Setup
llm = ChatOllama(base_url=base_url, model=model)
system = SystemMessagePromptTemplate.from_template("You are helpful assistant.")
human = HumanMessagePromptTemplate.from_template("{input}")
messages = [system, MessagesPlaceholder(variable_name='history'), human]
prompt = ChatPromptTemplate(messages=messages)
chain = prompt | llm | StrOutputParser()
runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, 
                                                   input_messages_key='input', 
                                                   history_messages_key='history')
def chat_with_llm(session_id, input):
    for output in runnable_with_history.stream({'input': input}, config={'configurable': {'session_id': session_id}}):
        yield output
prompt = st.chat_input("What is up?")
# st.write(prompt)
if prompt:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = st.write_stream(chat_with_llm(user_id, prompt))
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
```
- Run as `streamlit run chatbot.py`

### 62. Stream Output of Your Chat Bot like ChatGPT

## Section 10: Document Loaders | Projects on PDF Documents

### 63. Introduction to PDF Document Loaders
- Projects
  - Question answering from a PDF file
  - Summarizing the contents of a PDF file
- PDF libraries
  - https://pymupdf.readthedocs.io/en/latest/
    - pip install -qU langchain-community pymupdf tiktoken
  - PDFLoader from langchain
- Dataset required
  - git clone https://github.com/laxmimerit/rag-dataset

### 64. Load Single PDF Document with PyMuPDFLoader
```py
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("./rag-dataset/health supplements/1. dietary supplements - for whom.pdf") # space in the file name as it is
docs = loader.load()
len(docs)
docs[0].metadata
```

### 65. Load All PDFs from a Directory
```py
import os
pdfs = []
for root, dirs, files, in os.walk("rag-dataset"):
    #print(root,dirs,files)
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root,file))
#print('\n'.join(pdfs))
docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    temp = loader.load()
    docs.extend(temp)
```

### 66. Combine All PDFs Data as Context Text
```py
def format_docs(docs):
    return("\n\n".join([x.page_content for x in docs]))
context = format_docs(docs) # this will be several thousands of lines
# llama 3.2 can handle 128k tokens
```

### 67. How Many Tokens are There in Contex Data.
```py
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o-mini") # this model works generally for LLM
len(encoding.encode(docs[0].page_content)) # 1016
len(encoding.encode(context)) # 22829
```
- Make sure that the number of total tokens is smaller than the limit (128K for llama3.2b)

### 68. Make Question Answer Prompt Templates and Chain
```py
from langchain_ollama import ChatOllama
from langchain_core.prompts import (SystemMessagePromptTemplate, 
                                    HumanMessagePromptTemplate,
                                    ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
base_url = "http://localhost:11434"
model = "llama3.2:latest"
llm = ChatOllama(base_url=base_url, model=model)
#
system = SystemMessagePromptTemplate.from_template("""You are helpful AI assistant who answers your question based on the provided context. 
                                                   Do not answer in more than {words} words""")
prompt = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know"
            ### Context:
            {context}
            ### Question:
            {question}
            ### Answer:"""
prompt = HumanMessagePromptTemplate.from_template(prompt)
messages = [system, prompt]
template = ChatPromptTemplate(messages)
#template
template.invoke({'context':context, 'question':'How to gain muscle mass?', 'words':50})
#
qna_chain = template | llm | StrOutputParser()
qna_chain
```

### 69. Project 1 - Ask Questions from Your PDF Documents
```py
response = qna_chain.invoke({'context':context, 'question':"How to gain muscle mass?", 'words':50})
print(response)
'''
The answer was not related to the provided text. However, I can provide you with general information on how to gain muscle mass.

To gain muscle mass, you need to combine proper nutrition with regular exercise and sufficient rest. Here are some tips:

1. **Eat enough protein**: Protein is essential for building and repairing muscles. Aim for 1-1.5 grams of protein per kilogram of body weight daily.
2. **Increase caloric intake**: To build muscle, you need to be in a calorie surplus, meaning you consume more calories than your body burns. This will provide your body with the necessary energy to build and repair muscles.
3. **Focus on compound exercises**: Compound exercises like squats, deadlifts, bench press, and rows work multiple muscle groups at once and are effective for building muscle.
4. **Resistance training**: Resistance training is essential for building muscle. Focus on weightlifting exercises that challenge your muscles.
5. **Progressive overload**: Gradually increase the weight or resistance you're lifting over time to challenge your muscles and stimulate growth.
6. **Rest and recovery**: Adequate rest and recovery are crucial for muscle growth. Ensure you get 7-9 hours of sleep daily and take rest days as needed.
7. **Consider supplements**: Certain supplements like protein powder, creatine, and HMB can help support muscle growth. However, always consult with a healthcare professional before adding any new supplements to your regimen.

Remember, gaining muscle mass takes time, patience, and consistency. Stick to a well-planned workout and nutrition strategy, and you'll be on your way to building stronger muscles over time.
'''
```
- Took 3m7sec without CUDA
  - Took 1m17 sec with MX450 
- This is not RAG

### 70. Project 2 - Summarize Your PDF Documents
```py
system = SystemMessagePromptTemplate.from_template("""You are helpful AI assistant who answers your question based on the provided context. 
                                                   Do not answer in more than {words} words""")
prompt = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know"
            ### Context:
            {context}
            ### Summary:"""
prompt = HumanMessagePromptTemplate.from_template(prompt)
messages = [system, prompt]
template = ChatPromptTemplate(messages)
summary_chain = template | llm | StrOutputParser()
response = summary_chain.invoke({'context':context, 'words':50})
print(response)
```
- Took 8min without CUDA

### 71. Project 3 - Generate Detailed Structured Report from the PDF Documents
```py
response = qna_chain.invoke({'context':context, 'question':"provide a detailed report from the provided context. Write answer in markdown", 'words':2000})
print(response)
```
- Print format with Markdown

## Section 11: Document Loaders | Stock Market News Report Generation Project

### 72. Introduction to Webpage Loaders
- Handling HTML data
- Ref: https://python.langchain.com/docs/how_to/document_loader_web/
```py
from langchain_community.document_loaders import WebBaseLoader
urls = ['https://economictimes.indiatimes.com/markets/stocks/news',
        'https://www.livemint.com/latest-news',
        'https://www.livemint.com/latest-news/page-2',
        'https://www.livemint.com/latest-news/page-3',
        'https://www.moneycontrol.com']
loader = WebBaseLoader(web_paths=urls)
```

### 73. Load Unstructured Stock Market Data
```py
docs = []
async for doc in loader.alazy_load():
    docs.append(doc)
def format_docs(docs):
    return "\n\n".join([x.page_content for x in docs])
context = format_docs(docs) # now unstructured data are made    
import re
def text_clean(text):
    text = re.sub(r'\n\n+','\n', text) # many new lines into one line
    text = re.sub(r'\t+','\t', text) # many tabs into a tab
    text = re.sub(r'\s+',' ',text) # many spaces into one space
    return text
context = text_clean(context)                
```

### 74. Make LLM QnA Script
- Basically repetition of above-discussed contents
- Save as llm.py:
```py
### Question Answering using LLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import (SystemMessagePromptTemplate, 
                                    HumanMessagePromptTemplate,
                                    ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
base_url = "http://localhost:11434"
model = 'llama3.2:latest'
llm = ChatOllama(base_url=base_url, model=model)
system = SystemMessagePromptTemplate.from_template("""You are helpful AI assistant who answer user question based on the provided context.""")
prompt = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know".
            ### Context:
            {context}
            ### Question:
            {question}
            ### Answer:"""
prompt = HumanMessagePromptTemplate.from_template(prompt)
messages = [system, prompt]
template = ChatPromptTemplate(messages)
qna_chain = template | llm | StrOutputParser()
def ask_llm(context, question):
    return qna_chain.invoke({'context': context, 'question': question})
```    

### 75. Catastrophic Forgetting of LLM
- Ref: https://cobusgreyling.medium.com/catastrophic-forgetting-in-llms-bf345760e6e2
```py
from scripts2 import llm
response = llm.ask_llm(context, "Extract stock market news from the given text?")
print(response)
```
- Took more than 4min on CPUs - why CUDA is not running?
- Not answering well. The scope of data is too large
  - Split the context into smaller pieces, then collect result on each. Finalize with merging results. See below

### 76. Break Down Large Text Data Into Chunks
```py
response = llm.ask_llm(context[:10_000], "Extract stock market news from the given text?")
print(response)
```
- Took almost 2min on CPUs - CUDA is not running
- Answers better than the above.
- Let's split context while there must be some overlap as well:
  - Some irrelevant information might be injected
```py
def chunk_text(text, chunk_size, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
chunks = chunk_text(context, 10_000)
```

### 77. Create Stock Market News Summary for Each Chunks
```py
question = "Extract stock market news from the given text."
chunk_summary = []
for chunk in chunks:
    response = llm.ask_llm(chunk, question)
    chunk_summary.append(response)
for chunk in chunk_summary:
    print(chunk)
    print("\n\n")
    break
summary = "\n\n".join(chunk_summary)
print(summary)
```
- Took ~15min

### 78. Generate Final Stock Market Report
```py
question = """Write a detailed market news report in markdown format. Think carefully then write the report."""
response = llm.ask_llm(summary, question)
import os
os.makedirs("data", exist_ok=True)
with open("data/report.md", "w") as f:
    f.write(response)
with open("data/summary.md", "w") as f:
    f.write(summary)
```

## Section 12: Document Loaders | Microsoft Office Files Reader and Projects

### 79. Introduction to Unstructured Data Loader
- PPT, EXCEL, WORD documents
  - Key-notes and script generation for PPT
  - Read table data from excel file
  - Job description word file
- pip install "unstructured[all-docs]"
- pip install unstructured openpyxl python-magic python-pptx nltk

### 80. Load .PPTX Data with DataLoader
```py
from langchain_community.document_loaders import UnstructuredPowerPointLoader
loader = UnstructuredPowerPointLoader("./data/ml_course.pptx", mode="elements")
docs = loader.load()
```
- As the loader mode is 'elements', meta data like page number are available per element
  - `len(docs)` = 47 in this example

### 81. Process .PPTX data for LLM
```py
ppt_data = {}
for doc in docs:
    page = doc.metadata["page_number"]
    ppt_data[page] = ppt_data.get(page,"") + "\n\n" + doc.page_content
context = ""
for page, content in ppt_data.items():
    context += f"### slide {page}:\n\n{content.strip()}\n\n"
```
- Now context has following form:
```bash
### slide 1:

Machine Learning Model Deployment

Introduction to ML Pipeline

https://bit.ly/bert_nlp

### slide 2:
...
```
    
### 82. Generate Speaker Script for Your .PPTX Presentation
```py
from scripts2 import llm
question = """
For each powerpoint slide provided above, write a 2minute script that effectively conveys the key points.
Ensure a smooth flow between slides, maintaining a clean and engaging narrative
"""
response =llm.ask_llm(context,question)
print(response)
with open("output.txt","w") as f:
    f.write(response)
```
- output.txt:
```bash
Here is a 2-minute script for each PowerPoint slide:

**Slide 1: Introduction to ML Pipeline**

[Start with a brief introduction]
Good morning everyone, today we're going to discuss the importance of Machine Learning 
(ML) pipeline deployment. As we all know, ML models have the potential to revolutionize
 many industries and applications. However, deploying these models effectively is cruci
al for their success.

[Show the slide with the title "Machine Learning Pipeline"]
A machine learning pipeline refers to the process of developing, training, testing, and
 deploying a machine learning model. It's essential to understand how ML pipelines work
 before we dive into the different types of deployments.

**Slide 2: What is Machine Learning Pipeline?**
...
```

### 83. Loading and Parsing Excel Data for LLM
- data/Sample.xlsx has a table of:
```
First Name	Last Name	City	Gender
Brandon	James	Miami	M
Sean	Hawkins	Denver	M
Judy	Day	Los Angeles	F
Ashley	Ruiz	San Francisco	F
Stephanie	Gomez	Portland	F
```
- Loading excel file:
```py
from langchain_community.document_loaders import UnstructuredExcelLoader
loader = UnstructuredExcelLoader("data/sample.xlsx",mode="elements")
docs = loader.load()
print(len(docs)) # 1 is printed as there is only one sheet
docs[0].page_content
context = docs[0].metadata['text_as_html']
print(context)
```

### 84. Ask Questions from LLM for given Excel Data
```py
question = """
Return this data in MarkdownFormat
"""
response = llm.ask_llm(context,question)
print(response)
```
- Result:
```
| First Name | Last Name | City | Gender |
|------------|-----------|-------|--------|
| Brandon     | James      | Miami | M       |
| Sean        | Hawkins    | Denver | M       |
| Judy        | Day        | Los Angeles | F       |
| Ashley      | Ruiz       | San Francisco | F       |
| Stephanie   | Gomez      | Portland  | F       |
```

### 85. Load .DOCX Document and Write Personalized Job Email
- pip install -U docx2txt
```py
from langchain_community.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("data/job_description.docx")
docs = loader.load()
context = docs[0].page_content
context = docs[0].page_content
question = """"
Prepare a job application fitting with those job descriptions for a fresh collge student, who just completed a degree in computer science. The student name is Anita.
"""
response = llm.ask_llm(context,question)
print(response)
```
- Result:
```
Here's a sample job application tailored to the requirements of SpiceJet's Data Scientist position, highlighting Anita's skills and qualifications as a recent Computer Science graduate:

**Anita**
**Contact Information:**

* Email: [anita@email.com](mailto:anita@email.com)
* Phone: 1234567890
* LinkedIn: linkedin.com/in/anitacsc

**Summary:**
Highly motivated and detail-oriented computer science graduate seeking an entry-level Data Scientist position at SpiceJet. Proven analytical skills, programming experience in Python and R, and a strong understanding of statistical analysis and machine learning concepts. Eager to apply theoretical knowledge in real-world applications.

**Education:**

* Bachelor's degree in Computer Science, XYZ University (20XX-20XX)

**Technical Skills:**

* Programming languages: Python, R
* Data science tools: Tableau, SQL, Excel
* Machine learning platforms: Familiar with scikit-learn and TensorFlow
* Statistics and mathematics: Strong understanding of probability theory, linear algebra, and calculus

**Projects:**
...
**References:**
Available upon request.

As a fresh college graduate, Anita may not have extensive experience in data science or machine learning. However, she has demonstrated a strong foundation in computer science and a willingness to learn through her academic projects and internship experiences. By highlighting her technical skills, relevant coursework, and transferable experience, Anita can demonstrate her potential as an entry-level Data Scientist at SpiceJet.
```

## Section 13: Document Loaders | YouTube Video Transcripts and SEO Keywords Generator

### 86. Load YouTube Video Subtitles
- pip install youtube-transcript-api pytube
  - When an Exception while accessing title of ... happens (PytubeError), python -m pip install git+https://github.com/WildDIC/pytube.git
```py
from langchain_community.document_loaders import YoutubeLoader
url = "https://www.youtube.com/watch?v=vCyVVsdKJCg"
loader = YoutubeLoader.from_youtube_url(url,add_video_info=True)
docs = loader.load()
```
- HTTPError: HTTP Error 400: Bad Request
- Forgetting issue
  - If context is too large, breakdown them into smaller chunks

### 87. Load YouTube Video Subtitles in 10 Mins Chunks
```py
from langchain_community.document_loaders.youtube import TranscriptFormat
loader = YoutubeLoader.from_youtube_url(url,add_video_info=True, transcript_format=TranscriptFormat.CHUNKS, chunk_size_seconds=600)
docs = loader.load()
```

### 88. Generate YouTube Keywords from the Transcripts
```py
from scripts2 import llm
question = """
   You are an assistant for generating SEO keywords for Youtube.
   Please generate a list of keywords from the above context.
   You can use your creativity and correct spelling if it is needed.
"""
keywords = []
for doc in docs:
    kws = llm.ask_llm(context=doc.page_content, question=question)
    keywords.append(kws)
keywords = ",  ". join(keywords)
print(keywords)
```

## Section 14: Vector Stores and Retrievals

### 89. Introduction to RAG Project
![doc_ingestion](./ch89_ingestion.png)
![retrieval](./ch89_retrieval.png)
- The key is to generate FAISS or Chroma vector database
- The Issue of catastrophic forgetting in LLM
  - The oldes and the newest data are the most significant
  - Intermediate data are forgotten
  - Chunk-wise approach

### 90. Introduction to FAISS and Chroma Vector Database
- FAISS from Meta
  - pip install -qU langchain-community faiss-cpu
  - Ref: https://python.langchain.com/docs/integrations/vectorstores/faiss/
- Chroma database for prototype development
  - pip install -qU langchain-chroma
  - Ref: https://python.langchain.com/docs/integrations/vectorstores/chroma/
```py
import os
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # when chroma conflicts with fiass
warnings.filterwarnings("ignore")
```

### 91. Load All PDF Documents
```py
from langchain_community.document_loaders import PyMuPDFLoader
pdfs = []
for root,dirs,files in os.walk("rag-dataset"):
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root,file))
docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    temp = loader.load()
    docs.extend(temp)
print(len(docs))    
```

### 92. Recursive Text Splitter to Create Documents Chunk
```py
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
```
- Much simpler than the manual splitting in the Chap. 76

### 93. How Important Chunk Size Selection is?
```py
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
encoding.encode(chunks[0].page_content)
```

### 94. Get OllamaEmbeddings
- Ref: https://ollama.com/blog/embedding-models
  - Embedding models are models that are trained specifically to generate vector embeddings
  - The resulting vector embedding arrays can then be stored in a database, which will compare them as a way to search for data that is similar in meaning.
```py
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
embedings = OllamaEmbeddings(model='nomic-embed-text:latest', base_url='http://localhost:11434')
vector=embedings.embed_query("Hello world")
print(len(vector)) # = 768
```

### 95. Document Indexing in Vector Database
- The same embedding function must be used in embedding and retrieval steps
```py
index = faiss.IndexFlatL2(len(vector))
vector_store = FAISS(
  embedding_function=embeddings,
  index=index,
  docstore=InMemoryDocstore(),
  index_to_docstore_id={},
)
print(vector_store.index.ntotal, vector_store.index.d) # 0, 768
ids = vector_store.add_documents(documents=chunks)
print(len(ids), vector_store.index.ntotal)
```

### 96. How to Save and Search Vector Database
```py
index = faiss.IndexFlatL2(len(vector))
vector_store = FAISS(
  embedding_function=embeddings,
  index=index,
  docstore=InMemoryDocstore(),
  index_to_docstore_id={},
)
print(vector_store.index.ntotal, vector_store.index.d)
ids = vector_store.add_documents(documents=chunks) # took > 6min
print(len(ids), vector_store.index.ntotal) # 1259 1259
question = "how ot gain muscle mass?"
docs = vector_store.search(query=question, k=5, search_type="similarity")
#print(docs)
db_name = "health_supplements"
vector_store.save_local(db_name) # this creates a new folder of health_supplements with faiss file and pkl
```

## Section 15: RAG | Question Answer Over the Health Supplements Data

### 97. Load Vector Database for RAG
- RAG: Retrieval-Augmented Generation
- How to create a reponse using vector database
```py
import os
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # when chroma conflicts with fiass
warnings.filterwarnings("ignore")
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
embeddings = OllamaEmbeddings(model='nomic-embed-text:latest', base_url='http://localhost:11434')
# the embedding function must be same to the one used  in the embedding step
db_name = "health_supplements"
vector_store = FAISS.load_local(db_name,embeddings, allow_dangerous_deserialization=True)
# Retrieval
question = "how to gain muscle mass?"
docs = vector_store.search(query=question, k=5, search_type="similarity")
print(docs)
```

### 98. Get Vector Store as Retriever
```py
retriever = vector_store.as_retriever(search_type="similarity", 
                                      search_kwargs = {'k':3})
retriever.invoke(question)
```

### 99. Exploring Similarity Search Types with Retriever
- Search type in retriever
  - Ref: https://python.langchain.com/docs/how_to/vectorstore_retriever/
  - similarity: Default. Returns documents most similar to the query
  - mmr: Maximal Marginal Relevance to optimize for similarity to the query
    - lambda_mult: 1 for minimum diversity/0 for the maximum. Default is 0.5
    - fetch_k: amount of documents to pass to MMR
  - similarity_score_threshold: filters documents based on a similarity score threshold
```py
# Using similarity_score_threshold
retriever = vector_store.as_retriever(search_type="similarity_score_threshold", 
                                      search_kwargs = {'k':3, 'score_threshold':0.1})
retriever.invoke(question)
# Using mmr
retriever = vector_store.as_retriever(search_type="mmr", 
                                      search_kwargs = {'k':3, 'fetch_k':20, 'lambda_mult':1})
retriever.invoke(question)
```

### 100. Design RAG Prompt Template
```py
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
#prompt = hub.pull("rlm/rag-prompt") # from https://smith.langchain.com/hub/rlm/rag-prompt
#print(prompt)
prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.  
Question: {question}
Context: {context}
Answer:"""
prompt = ChatPromptTemplate.from_template(prompt)
```

### 101. Build LLM RAG Chain
```py
llm = ChatOllama(model='llama3.2:latest', base_url='http://localhost:11434')
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])
context = format_docs(docs)
rag_chain = (
  {'context': retriever|format_docs, 'question':RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)
```

### 102. Prompt Tuning and Generate Response from RAG Chain
```py
question = "How to lose weight?"
response = rag_chain.invoke(question)
print(response)
```
- Try some prompt engineering, providing bullets to be brief

## Section 16: Tool and Function Calling

### 103. What is Tool Calling
- LLM automatically calls the function based on the query
- Function parameters are automaically passed to the function
- Essential requirements for the Agent
- Not all LLM supports tool calling
- Briefly, instead of natural language, we can give better instruction to LLM through function/tool calling
  - Providing meta-data explicitly
```py
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
llm = ChatOllama(model='llama3.2:latest', base_url='http://localhost:11434')
llm.invoke('hi')
```

### 104. Available Search Tools at Langchain
- https://python.langchain.com/docs/integrations/tools/
  - tavily.com: 1000 free search
  - DuckDuckgoSearch: Free

### 105. Create Your Custom Tools
- How to call in-built tool
```py
from langchain_core.tools import tool
@tool
def add(a,b):
  """
  Add two integer numbers together
  Args:
  a: First integer
  b: Second integer
  """
  return a+b
@tool # makes the function runnable
def multiply(a,b):
  """
  multiply two integer numbers together
  Args:
  a: First integer
  b: Second integer
  """
  return a*b
add # StructuredTool(name='add', description='Add two integer numbers together\nArgs:\na: First integer\nb: Second integer', args_schema=<class 'langchain_core.utils.pydantic.add'>, func=<function add at 0x7bb01102a200>)
print(add.name, add.description)  
```

### 106. Bind tools with LLM
```py
# add(1,2) # this crashes
add.invoke({'a': 1,'b': 2}) # now works OK
myTools = [add,multiply]
llm_with_tools = llm.bind_tools(myTools)
question  = "what is 1 plus 2?"
llm_with_tools.invoke(question)
"""
AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'llama3.2:latest', 'created_at': '2025-10-01T00:35:24.470173731Z', 'done': True, 'done_reason': 'stop', 'total_duration': 3097237603, 'load_duration': 83336301, 'prompt_eval_count': 246, 'prompt_eval_duration': 173726198, 'eval_count': 16, 'eval_duration': 2838791331, 'model_name': 'llama3.2:latest'}, id='run--5c0b2c45-392d-49d3-9c39-0e58b4b121ba-0', tool_calls=[{'name': 'add', 'args': {'a': 1, 'b': 2}, 'id': 'ec4306c6-c421-4e22-a6b2-9ce0299004a9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 246, 'output_tokens': 16, 'total_tokens': 262})
"""
llm_with_tools.invoke(question).tool_calls
"""
[{'name': 'add',
  'args': {'a': 1, 'b': 2},
  'id': 'fb59adc1-d459-4a23-b7f9-71b524b016df',
  'type': 'tool_call'}]
"""
```

### 107. Working with Tavily and DuckDuckGo Search Tools
- Ref: https://python.langchain.com/docs/integrations/tools/ddg/
- pip install duckduckgo-search ddgs wikipedia xmltodict tavily-python langchain-community
```py
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
search.invoke("What is today's stock market news?")
```

### 108. Working with Wikipedia and PubMed Tools
- Ref: https://python.langchain.com/docs/integrations/tools/wikipedia/
```py
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
question  = "What is the Capital of France?"
wikipedia.invoke(question)
```
- Ref: https://python.langchain.com/docs/integrations/tools/pubmed/
```py
from langchain_community.tools.pubmed.tool import PubmedQueryRun
search =  PubmedQueryRun()
print(search.invoke("What is the latest search on COVID-19?"))
```
  - ? Not working?

### 109. Creating Tool Functions for In-Built Tools
```py
@tool
def wikipedia_seach(query):
    """
    Search wikipedia for general information
    Args:
    query: the search query
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    response = wikipedia.invoke(query)
    return response
@tool
def pubmed_search(query):
    """
    Search wikipedia for general information
    Args:
    query: the search query
    """
    search = PubmedQueryRun()
    response = search.invoek(query)
    return response
tools = [wikipedia_seach, pubmed_search]
```
- Comment section is required in @tool

### 110. Calling Tools with LLM
```py
list_of_tools = { tool.name: tool for tool in tools }
llm_with_tools = llm.bind_tools(tools)
query  = "what is 2 * 3?"
response = llm_with_tools.invoke(query)
print(response.tool_calls)
```

### 111. Passing Tool Calling Result to LLM Part 1
- How to couple a local LLM with external search engines
![ch111](./ch111.png)
```py
from langchain_core.messages import HumanMessage
query  = "what is 2 * 3?"
messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
```

### 112. Passing Tool Calling Result to LLM Part 2
```py
for tool_call in ai_msg.tool_calls:
    print(tool_call)
    name = tool_call['name'].lower()
    selected_tool = list_of_tools[name]
    tool_msg = selected_tool.invoke(tool_call['args'])
    messages.append(tool_msg)
response = llm_with_tools.invoke(messages)
print(response.content)    
```

## Section 17: Agents

### 113. How Agent Works
- Agents are systems that use an LLM as a reasoning engine to determine which actions to take and what inputs to those actions must be

### 114. Tools Preparation for Agent
```py
from langchain_ollama import ChatOllama 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
llm = ChatOllama(model='llama3.2:latest', base_url='http://localhost:11434')
#llm.invoke('hi')
from langchain_community.tools import DuckDuckGoSearchRun
@tool
def search(query:str) -> str:
    """
    Search ddg
    Ags:
    query: The search query
    """
    search = DuckDuckGoSearchRun()
    search.invoke(query)
    return response
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
db_name = "health_supplements"
vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})
@tool
def health_supplements(query: str) -> str:
    """Search for information about Health Supplements.
    For any questions about Health and Gym Supplements, you must use this tool!,

    Args:
        query: The search query.
    """
    response = retriever.invoke(query)
    return response
health_supplements.invoke("what is the best supplement for muscle gain?")    
```

### 115. More About the Agent Working Process
- The agent is responsible for taking in input and deciding what actions to take
- The Agent does not execute those actions - done by AgentExecutor
- `create_tool_calling_agent` will call `.bind_tools` for us under the hood

### 116. Selection of Prompt for Agent
```py
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages
```

### 117. Agent in Action
```py
tools = [search, health_supplements]
agent = create_tool_calling_agent(llm,tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
question = "What is the best supplement for muscle gain?"
response = agent_executor.invoke({'input':question})
"""
> Entering new AgentExecutor chain...
Invoking: `health_supplements` with `{'query': 'muscle gain supplements'}`
...
"""
print(response)
"""
{'input': 'What is the best supplement for muscle gain?', 'output': "Based on the provided documents, ...
"""
```

## Section 18: Text to MySQL Queries | With and Without Agents

### 118. Create MySQL Connection with Local Server
- Build Q&A systems of SQL database
  - Executes model-generated SQL queries
- Make DB as read-only for this example
- pip install pymysql
```py
from langchain_ollama import ChatOllama 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
llm = ChatOllama(model='llama3.2:3b', base_url='http://localhost:11434')
from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("mysql+pymysql://root:root@localhost/employees")
db.dialect
db.get_usable_table_names()
db.run("SELECT * FROM employees LIMIT 5")
```

### 119. Get MySQL Execution Chain
```py
from langchain.chains import create_sql_query_chain
sql_chain = create_sql_query_chain(llm, db)
sql_chain.get_prompts()[0].pretty_print()
question = "how many employees are there? You MUST RETURN ONLY MYSQL QUERIES."
response = sql_chain.invoke({'question': question})
print(response)
```

### 120. Correct Malformed MySQL Queries Using LLM
```py
from scripts.llm import ask_llm
from langchain_core.runnables import chain
@chain
def get_correct_sql_query(input):
    context = input['context']
    question = input['question']
    intruction = """
        Use above context to fetch the correct SQL query for following question
        {}

        Do not enclose query in ```sql and do not write preamble and explanation.
        You MUST return only single SQL query.
    """.format(question)
    response = ask_llm(context=context, question=intruction)
    return response
response = get_correct_sql_query.invoke({'context': response, 'question': question})
db.run(response)
```

### 121. MySQL Query Chain Execution
```py
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
execute_query = QuerySQLDataBaseTool(db=db)
sql_query = create_sql_query_chain(llm, db)
final_chain = (
    {'context': sql_query, 'question': RunnablePassthrough()}
    | get_correct_sql_query
    | execute_query
)
question = "how many employees are there? You MUST RETURN ONLY MYSQL QUERIES."
response = final_chain.invoke({'question': question})
print(response)
```

### 122. MySQL Query Execution with Agents in LangGraph
- pip install langgraph
```py
from langchain_community.agent_toolkits import SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
from langchain_core.messages import SystemMessage
SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""
system_message = SystemMessage(content=SQL_PREFIX)
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(llm, tools, state_modifier=system_message, debug=False)
question = "How many employees are there?"
# question = "How many departments are there?"
agent_executor.invoke({"messages": [HumanMessage(content=question)]})
for s in agent_executor.stream(
    {"messages": [HumanMessage(content=question)]}):
    print(s)
    print("----")
```

## Section 19: Linkedin Profile Scraping Using LLM

### 123. Introduction
### 124. Introduction to LinkedIn Profile Scraping
### 125. Introduction to Selenium and BeautifulSoup bs4
### 126. Code Notebook Setup
### 127. Automated Login to LinkedIn Using Selenium Web Driver Tool
### 128. Load LinkedIn Profile Source Data with BeautifulSoup
### 129. Get the Profile Data Section wise
### 130. Text Cleaning for LLM
### 131. Parse Your First Section and Limits of LLAMA or Any Smaller Models
### 132. Parse LinkedIn Data Section wise
### 133. Correct LinkedIn Parsing using Second LLM Call


134. Introduction to Resume Parsing
135. Read Resume Data and Prepare Context and Question
136. Prepare Parsing and Validation LLM Pipeline
137. Parse and Validate Any Resume Data into JSON file
138. Make Resume Parsing Streamlit Application
139. Parse Any Type of Resume with LLM and Streamlit 



140. Launch Deep Learning AWS EC2 Ubuntu Machine
141. Installing Ollama and Langchain on EC2 Server
142. Connect Your VS Code with Remote EC2 Server
143. Deploy LLM Application on the Server
