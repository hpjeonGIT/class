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
### 56. Introduction To Streamlit and Our Chat Application
### 57. Chat Bot Basic Code Setup
### 58. Create Chat History in Streamlit Session State
### 59. Create LLM Chat Input Area with Streamlit
### 60. Update Historical Chat on Streamlit UI
### 61. Complete Your Own Chat Bot Application
### 62. Stream Output of Your Chat Bot like ChatGPT



63. Introduction to PDF Document Loaders
64. Load Single PDF Document with PyMuPDFLoader
65. Load All PDFs from a Directory
66. Combine All PDFs Data as Context Text
67. How Many Tokens are There in Contex Data.
68. Make Question Answer Prompt Templates and Chain
69. Project 1 - Ask Questions from Your PDF Documents
70. Project 2 - Summarize Your PDF Documents
71. Project 3 - Generate Detailed Structured Report from the PDF Documents



72. Introduction to Webpage Loaders
73. Load Unstructured Stock Market Data
74. Make LLM QnA Script
75. Catastrophic Forgetting of LLM
76. Break Down Large Text Data Into Chunks
77. Create Stock Market News Summary for Each Chunks
78. Generate Final Stock Market Report




79. Introduction to Unstructured Data Loader
80. Load .PPTX Data with DataLoader
81. Process .PPTX data for LLM
82. Generate Speaker Script for Your .PPTX Presentation
83. Loading and Parsing Excel Data for LLM
84. Ask Questions from LLM for given Excel Data
85. Load .DOCX Document and Write Personalized Job Email



86. Load YouTube Video Subtitles
87. Load YouTube Video Subtitles in 10 Mins Chunks
88. Generate YouTube Keywords from the Transcripts



89. Introduction to RAG Project
90. Introduction to FAISS and Chroma Vector Database
91. Load All PDF Documents
92. Recursive Text Splitter to Create Documents Chunk
93. How Important Chunk Size Selection is?
94. Get OllamaEmbeddings
95. Document Indexing in Vector Database
96. How to Save and Search Vector Database



97. Load Vector Database for RAG
98. Get Vector Store as Retriever
99. Exploring Similarity Search Types with Retriever
100. Design RAG Prompt Template
101. Build LLM RAG Chain
102. Prompt Tuning and Generate Response from RAG Chain



103. What is Tool Calling
104. Available Search Tools at Langchain
105. Create Your Custom Tools
106. Bind tools with LLM
107. Working with Tavily and DuckDuckGo Search Tools
108. Working with Wikipedia and PubMed Tools
109. Creating Tool Functions for In-Built Tools
110. Calling Tools with LLM
111. Passing Tool Calling Result to LLM Part 1
112. Passing Tool Calling Result to LLM Part 2


113. How Agent Works
114. Tools Preparation for Agent
115. More About the Agent Working Process
116. Selection of Prompt for Agent
117. Agent in Action


118. Create MySQL Connection with Local Server
119. Get MySQL Execution Chain
120. Correct Malformed MySQL Queries Using LLM
121. MySQL Query Chain Execution
122. MySQL Query Execution with Agents in LangGraph


123. Introduction
124. Introduction to LinkedIn Profile Scraping
125. Introduction to Selenium and BeautifulSoup bs4
126. Code Notebook Setup
127. Automated Login to LinkedIn Using Selenium Web Driver Tool
128. Load LinkedIn Profile Source Data with BeautifulSoup
129. Get the Profile Data Section wise
130. Text Cleaning for LLM
131. Parse Your First Section and Limits of LLAMA or Any Smaller Models
132. Parse LinkedIn Data Section wise
133. Correct LinkedIn Parsing using Second LLM Call


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
