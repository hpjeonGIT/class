## AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents
- Instructor: Ed Donner

## Section 1: New Week 1 - Build your First LLM Product: Exploring Top Models

### 1. Day 1 - Running Your First LLM Locally with Ollama and Open Source Models
- https://edwarddonner.com/2024/11/13/llm-engineering-resources/
- https://github.com/ed-donner/llm_engineering
- Running ollama
  - ollama run gemma3:270m

### 2. Day 1 - Spanish Tutor Demo with Open-Source Models & Course Overview

### 3. Day 1 - Setting Up Your LLM Development Environment with Cursor and UV
- Step 1: clone the repo, install Cursor
- Step 2: Install UV
- Step 3: Create an OpenAI key
- Step 4: Create the .env file

### 4. Day 1 - Setting Up Your PC Development Environment with Git and Cursor

### 5. Day 1 - Mac Setup: Installing Git, Cloning the Repo, and Cursor IDE
- Cursor: spawned from VS code

### 6. Day 1 - Installing UV and Setting Up Your Cursor Development Environment
- UV: something like anacondas
- curl -LsSf https://astral.sh/uv/install.sh | sh

### 7. Day 1 - Setting Up Your OpenAI API Key and Environment Variables
- https://platform.openai.com/docs/overview
- $5 payment required
- Optional in this class
- Store the key at `.env` file
  - OPENAI_API_KEY=XXXX

### 8. Day 1 - Installing Cursor Extensions and Setting Up Your Jupyter Notebook

### 9. Day 1 - Running Your First OpenAI API Call and System vs User Prompts
- System prompt: Tone of the system
```
systemp_prompt = ...
```
- User prompt: conversation with users
```
user_prompt_prefix= ...
```

### 10. Day 1 - Building a Website Summarizer with OpenAI Chat Completions API

### 11. Day 1 - Hands-On Exercise: Building Your First OpenAI API Call from Scratch

### 12. Day 2 - LLM Engineering Building Blocks: Models, Tools & Techniques
- 3 dimensions of LLM engineering
  - Models: open source, multi-modal, architecture, ...
  - Tools: HuggingFace, LangChain, Gradio, 
  - Techniques: APIs, RAG, Fine-tuning, Agentization

### 13. Day 2 - Your 8-Week Journey: From Chat Completions API to LLM Engineer
- Week 1: Foundations and the chat completions API
- Week 2: Frontier models with APIs, UIs, and Multi-modality
- Week 3: Open source with HuggingFace
- Week 4: Selecting LLMs and Code Generation
- Week 5: RAG and Question Answering - Creating an expert
- Week 6: Fine-tuning a Froniter Model
- Week 7: Fine-tuning an open source model
- Week 8: The finale - Agentic AI

### Day 2 - Frontier Models: OpenAI GPT, Claude, Gemini & Grok Compared
- Closed-source frontier
  - GPT from OpenAI
  - Claude from Anthropic
  - Gemini from Google
  - Grok from x.ai

### 15. Day 2 - Open-Source LLMs: LLaMA, Mistral, DeepSeek, and Ollama
- OSS models
  - Llama from Meta
  - Mixtral from Mistral
  - Qwen from Alibaba cloud
  - Gemma from Google
  - Phi from Miscrosoft
  - DeepSeek from DeepSeek AI
  - GPT-OSS from OpenAI
- Three ways to use models
  - Chat interfaces
  - APIs
    - LangChain
    - Cloud APIs like Amazon Bedrock, Google Vertex, Azure ML
  - Direct inference
    - Local Ollama run
    - HuggingFace Transformers library

### 16. Day 2 - Chat Completions API: HTTP Endpoints vs OpenAI Python Client
- Chat completions API
  - The simplest way to call an LLM
  - Created by OpenAI
  - All LLM vendor mimic this
- API end point
  - https://api.openai.com/v1/chat/completions
- openai package
  - Python client library

### 17. Day 2 - Using the OpenAI Python Client with Multiple LLM Providers
```py
# Create OpenAI client
from openai import OpenAI
openai = OpenAI()
response = openai.chat.completions.create(model="gpt-5-nano", messages=[{"role": "user", "content": "Tell me a fun fact"}])
response.choices[0].message.content
```
- For gemini:
```py
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
load_dotenv(override=True)
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("No API key was found - please be sure to add your key to the .env file, and save the file! Or you can skip the next 2 cells if you don't want to use Gemini")
elif not google_api_key.startswith("AIz"):
    print("An API key was found, but it doesn't start AIz")
else:
    print("API key found and looks good so far!")
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
response = gemini.chat.completions.create(model="gemini-2.5-flash-lite", messages=[{"role": "user", "content": "Tell me a fun fact"}])
response.choices[0].message.content
```

### 18. Day 2 - Running Ollama Locally with OpenAI-Compatible Endpoints
- pip3 install OpenAI
- Still needs OpenAI_API_KEY for ollama 

### 19. Day 3 - Base, Chat, and Reasoning Models: Understanding LLM Types
- LLMs come in 3 flavors
  - Base: ex) GPT
    - Better for fine-tuning to learn a new skill
  - Chat/instruct: ex) Chat-GPT
    - Better for interactive use cases and creative content generation?
  - Reasoning/Thinking: chain of thought
    - Better for problem solving
    - Budget forcing: A technique to control the amount of thinking or computational budget an LLM uses 
      - Adding word "wait" frequently, yields better results for long thinking

### 20. Day 3 - Frontier Models: GPT, Claude, Gemini & Their Strengths and Pitfalls
- Good at:
  - Synthesizing information
  - Fleshing out a skeleton
  - Coding
- Limitations
  - Specialized domains
  - Recent events
  - Can confidently make mistakes

### 21. Day 3 - Testing ChatGPT-5 and Frontier LLMs Through the Web UI
- Q: how many times does the letter 'a' appear in this sentence
  - Several seconds on even GPT5
- Q: How many words are there in your answer to this question?
  - A: one
  - Several seconds on GPT5

### 22. Day 3 - Testing Claude, Gemini, Grok & DeepSeek with ChatGPT Deep Research
- Claude: good at edge cases

### 23. Day 3 - Agentic AI in Action: Deep Research, Claude Code, and Agent Mode
- Agent mode: Find me a restaurant in NYC today 4-7pm, serving a meat pie
  - Research through web like Yelp and Reddit to find restaurings serving a meat pie
  - Looks for scheduling features

### 24. Day 3 - Frontier Models Showdown: Building an LLM Competition Game
- https://github.com/ed-donner/outsmart

### 25. Day 4 - Understanding Transformers: The Architecture Behind GPT and LLMs
- The extraordinary rise of the transformers
  - 2017, "Attention is All you need" by Google
  - 2018, GPT-1 by OpenAI
  - 2019, GPT-2
  - 2020, GPT-3
  - 2022, RLHF and ChatGPT
    - Chat mode
  - 2023, GPT-4
  - 2024, GPT-4o
- Transformation is a way of optimization
  - Efficient for very large data
  - No proven supriority

### 26. Day 4 - From LSTMs to Transformers: Attention, Emergent Intelligence & Agentic A
- LSTM: sequential training
- Transformers: Simple. No sequences. Attention is all. Can be parallelized
- Reaction to transformers
  - Stochastic Parrot
  - Predictive text on steroids
- Prompt engineering -> Context engineering  
- Now Agentic AI

### 27. Day 4 - Parameters: From Millions to Trillions in GPT, LLaMA & DeepSeek
- GPT-1: 117M
- GPT-2: 1.5B
- Llama 3.2: 3B
- Llama 3.1: 8B
- Llama 3.3: 70B
- GPT-OSS: 120B
- GPT-3: 175B
- DeepSeek: 671B
- GPT-4: 1.76T
- Latest Frontier models: undisclosed
- Training time scaling
  - Time taken for training models
- Infererence time scaling
  - Reasoning trick
    - RAG
  - Speed up inference

### 28. Day 4 - What Are Tokens? From Characters to GPT's Tokenizer
- In the early days, neural network were trained at the character level
- Then neural networks were trained off words
  - Predicts the next word in this sequence
- The breakthrough was to work with chunks of words, called "tokens"
  - Useful information for the neural network. Elegantly handles word stems

### 29. Day 4 - Understanding Tokenization: How GPT Breaks Down Text into Tokens
- GPT tokenizer: https://platform.openai.com/tokenizer
- Common words: split by spaces
  - `Today's temperature is less than 20F at Boston`
  - `Today's`, `temperature`, `is`, `less`, `than`, `20F`, `at`, `Boston`
- Less common words (and invented words!): broken into multiple tokens
  - Sometimes, a word is broken into fragments
  - `An exquisitely handcrafted IPs for mastery of witchcraft`
  - `An`, `exquis`,`it`,`ely`, `handcrafted` `IP`,`s`, `for`, `mastery`, `of`, `witch`, `craft`
- Rule-of-themb in typical english writing:
  - 1 token is ~4 characters
  - 1 token is ~0.75 words
  - so 1,000 tokens is ~750 words
  - The collected workds of Shakespeare are ~900,000 words or 1.2M tokens

### 30. Day 4 - Tokenizing with tiktoken and Understanding the Illusion of Memory
- I change to ollama interface from OpenAI package as no OPENAI_API_KEY is available
```py
from openai import OpenAI
import os
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
messages = [
  {"role":"system", "content": "You are a helpful assistant"},
  {"role":"user", "content": "Hi I am Ed"}
]
response = client.chat.completions.create(model="llama3.2:1b", messages=messages)
response.choices[0].message.content
```
- Result: 
```bash
12194 = Hi
922 =  my
1308 =  name
382 =  is
1648 =  Ed
```
- Illusion of memory
  - Every call to an LLM is stateless
    - No history recorded
  - To overcome, pass in the entire conversation in the input prompt, everytime, giving the illusion that LLM has memory

### 31. Day 4 - Context Windows, API Costs, and Token Limits in LLMs
- Context window
  - Max. number of tokens that the model can consider when generating the next token
  - Includes the original input prompt, subsequent conversation, the latest input prompt and almost all the output prompt
  - It governs how well the model can remember references, content, and context
  - Particularly important for muli-shot prompting where the prompt includes examples, or for long conversations
- API costs
  - Based on the number of input tokens and output tokens
  - Tokens of old prompt for illusion of memory
  - Includes tokens for reasoning
- Common context windows and API costs
  - https://www.vellum.ai/llm-leaderboard
  - GPT5: 400,000 context window, $125 / 1M input tokens, $10 / 1M output tokens
  - Gemini2.5/Flash: 1,000,000 context window, $0.15 / 1M input tokens, $0.6 / 1M output tokens

### 32. Day 5 - Building a Sales Brochure Generator with OpenAI Chat Completions API
- Company sales brochure generator
  - Create a product that can generate marking brochure about a company
    - For prospective clients
    - For investors
    - For recruiment
  - Use OpenAI API
  - Use on shot prompting
  - Stream back results and show with formatting


### 33. Day 5 - Building JSON Prompts and Using OpenAI's Chat Completions API
- Singleshot prompting: a technique where you provide an AI model with a single example to guide its response. Instead of explicitly explaining a task, you show the model one input-output pair, and it uses that example to perform the same task on a new query.
- Multishot prompting: a technique where you provide multiple examples to an AI model within the prompt itself to guide its output. By giving the model several examples of the desired input and output, it can better understand complex tasks and generate responses that match a specific pattern, format, or style

### 34. Day 5 - Chaining GPT Calls: Building an AI Company Brochure Generator
```py
import os
import json
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from scraper import fetch_website_links, fetch_website_contents
from openai import OpenAI
# Initialize and constants
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
if api_key and api_key.startswith('sk-proj-') and len(api_key)>10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")
MODEL = 'gpt-5-nano'
openai = OpenAI()
links = fetch_website_links("https://edwarddonner.com")
links
# First step: Have GPT-5-nano figure out which links are relevant
# Use a call to gpt-5-nano to read the links on a webpage, and respond in structured JSON.
# It should decide which links are relevant, and replace relative links such as "/about" with "https://company.com/about".
# We will use "one shot prompting" in which we provide an example of how it should respond in the prompt.
# This is an excellent use case for an LLM, because it requires nuanced understanding. Imagine trying to code this without LLMs by parsing and analyzing the webpage - it would be very hard!
# Sidenote: there is a more advanced technique called "Structured Outputs" in which we require the model to respond according to a spec. We cover this technique in Week 8 during our autonomous Agentic AI project.
link_system_prompt = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""
def get_links_user_prompt(url):
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company, 
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.
Links (some might be relative links):
"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt
print(get_links_user_prompt("https://edwarddonner.com"))
def select_relevant_links(url):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    links = json.loads(result)
    return links
select_relevant_links("https://edwarddonner.com")
```

### 35. Day 5 - Building a Brochure Generator with GPT-4 and Streaming Results
```py
def select_relevant_links(url):
    print(f"Selecting relevant links for {url} by calling {MODEL}")
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    links = json.loads(result)
    print(f"Found {len(links['links'])} relevant links")
    return links
select_relevant_links("https://edwarddonner.com")
select_relevant_links("https://huggingface.co")
#Second step: make the brochure!
#Assemble all the details into another prompt to GPT-5-nano
def fetch_page_and_all_relevant_links(url):
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links['links']:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    return result
print(fetch_page_and_all_relevant_links("https://huggingface.co"))
brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""
# Or uncomment the lines below for a more humorous brochure - this demonstrates how easy it is to incorporate 'tone':
# brochure_system_prompt = """
# You are an assistant that analyzes the contents of several relevant pages from a company website
# and creates a short, humorous, entertaining, witty brochure about the company for prospective customers, investors and recruits.
# Respond in markdown without code blocks.
# Include details of company culture, customers and careers/jobs if you have the information.
# """
def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt
get_brochure_user_prompt("HuggingFace", "https://huggingface.co")
def create_brochure(company_name, url):
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
    )
    result = response.choices[0].message.content
    display(Markdown(result))
create_brochure("HuggingFace", "https://huggingface.co")
# Finally - a minor improvement
# With a small adjustment, we can change this so that the results stream back from OpenAI, with the familiar typewriter animation
def stream_brochure(company_name, url):
    stream = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ],
        stream=True
    )    
    response = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        update_display(Markdown(response), display_id=display_handle.display_id)
stream_brochure("HuggingFace", "https://huggingface.co")
# Try changing the system prompt to the humorous version when you make the Brochure for Hugging Face:
stream_brochure("HuggingFace", "https://huggingface.co")
```

### 36. Day 5 - Business Applications, Challenges & Building Your AI Tutor
- Retry the above application using ollama

## Section 2: NEW Week 2 - Build a Multi-Modal Chatbot: LLMs, Gradio UI, and Agents

### 37. Day 1 - Connecting to Multiple Frontier Models with APIs (OpenAI, Claude, Gemini)
- Setting up API keys
```py
from openai import OpenAI
import os
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
messages = [
  {"role":"user", "content": "Tell a joke for a student on the journey to becoming an expert in LLM"}
]
response = client.chat.completions.create(model="llama3.2:1b", messages=messages)
response.choices[0].message.content
```

### 38. Day 1 - Testing GPT-5 Models with Reasoning Effort and Scaling Puzzles
- llama3.2:8b doesn't support thinking type(reasoning_effort)
  - Most of ollama don't support
  - Download GPT-OSS models
- To solve complex problem
  - Increaes reasoning level for small models
  - Or use larger models

### 39. Day 1 - Testing Claude, GPT-5, Gemini & DeepSeek on Brain Teasers
- Comparison of GPT-5 vs Claude vs Grok for complex puzzles

### 40. Day 1 - Local Models with Ollama, Native APIs, and OpenRouter Integration
- When Ollama doesn't run on GPU: https://github.com/wgong/py4kids/blob/master/lesson-18-ai/ollama/gpu/fix-GPU-access-failure-after-suspend-resume-linux.md
- When installing/reinstalling ollama, make sure:
```bash
>>> Adding current user to ollama group...
>>> Creating ollama systemd service...
>>> Enabling and starting ollama service...
>>> NVIDIA GPU installed.
```
- Will use gpt-oss:20b. This may require at least 16GB RAM
- openrouter.ai
  - Abstraction layer b/w users and Frontier models

### 41. Day 1 - LangChain vs LiteLLM: Choosing the Right LLM Framework
- LangChain: heavy framework
```py
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini")
response = llm.invoke(tell_a_joke)
display(Markdown(response.content))
```
- LiteLLM: light framework
```py
from litellm import completion
response = completion(model="openai/gpt-4.1", messages=tell_a_joke)
reply = response.choices[0].message.content
display(Markdown(reply))
```
  - Has features to show total costs
- Prompt caching
  - Stores previous calls and can save costs
  - Cached input is 4x cheaper in OpenAI

### 42. Day 1 - LLM vs LLM: Building Multi-Model Conversations with OpenAI & Claude
- Let Frontier models discuss together
```py
gpt_model = "gpt-4.1-mini"
claude_model = "claude-3-5-haiku-latest"
gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."
claude_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."
gpt_messages = ["Hi there"]
claude_messages = ["Hi"]
def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude in zip(gpt_messages, claude_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": claude})
    response = openai.chat.completions.create(model=gpt_model, messages=messages)
    return response.choices[0].message.content
call_gpt()
def call_claude():
    messages = [{"role": "system", "content": claude_system}]
    for gpt, claude_message in zip(gpt_messages, claude_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": claude_message})
    messages.append({"role": "user", "content": gpt_messages[-1]})
    response = anthropic.chat.completions.create(model=claude_model, messages=messages)
    return response.choices[0].message.content
call_claude()
call_gpt()
gpt_messages = ["Hi there"]
claude_messages = ["Hi"]
display(Markdown(f"### GPT:\n{gpt_messages[0]}\n"))
display(Markdown(f"### Claude:\n{claude_messages[0]}\n"))
for i in range(5):
    gpt_next = call_gpt()
    display(Markdown(f"### GPT:\n{gpt_next}\n"))
    gpt_messages.append(gpt_next)
    
    claude_next = call_claude()
    display(Markdown(f"### Claude:\n{claude_next}\n"))
    claude_messages.append(claude_next)
```
- Ollama version: using gemma3 and Llama3.2
```py
from openai import OpenAI
import os
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
gemma3_model = "gemma3:270m"
llama_model = "llama3.2:1B"
gemma3_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."
llama_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."
gemma3_messages = ["Hi there"]
llama_messages = ["Hi"]
def call_gemma3():
    messages = [{"role": "system", "content": gemma3_system}]
    for gemma3, llama in zip(gemma3_messages, llama_messages):
        messages.append({"role": "assistant", "content": gemma3})
        messages.append({"role": "user", "content": llama})
    response = client.chat.completions.create(model=gemma3_model, messages=messages)
    return response.choices[0].message.content
call_gemma3()
def call_llama():
    messages = [{"role": "system", "content": llama_system}]
    for gemma3, llama_message in zip(gemma3_messages, llama_messages):
        messages.append({"role": "user", "content": gemma3})
        messages.append({"role": "assistant", "content": llama_message})
    messages.append({"role": "user", "content": gemma3_messages[-1]})
    response = client.chat.completions.create(model=llama_model, messages=messages)
    return response.choices[0].message.content
call_llama()
call_gemma3()
gpt_messages = ["Hi there"]
claude_messages = ["Hi"]
print(f"### Gemma3:\n{gpt_messages[0]}\n")
print(f"### LLama:\n{claude_messages[0]}\n")
for i in range(5):
    gemma3_next = call_gemma3()
    print(f"### Gemma3:\n{gemma3_next}\n")
    gemma3_messages.append(gemma3_next)    
    llama_next = call_llama()
    print(f"### Llama:\n{llama_next}\n")
    llama_messages.append(llama_next)
```

### 43. Day 2 - Building Data Science UIs with Gradio (No Front-End Skills Required)
- UIs for non-frontend people
  - Gradio
  - Opensource Python UI by HuggingFace
- pip3 install gradio


### 44. Day 2 - Building Your First Gradio Interface with Callbacks and Sharing
```py
import os
import gradio as gr
def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()
shout("hello")
gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch()
```
- Will be hosted at http://127.0.0.1:7861/
- `.launch(share=True)` for sharing links with anybody
  - For security-wise, add authentication or consult with IT department
  - `share=False` for running locally. Default

### 45. Day 2 - Building Gradio Interfaces with Authentication and GPT Integration
- Adding authentication
```py
gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch(inbrowser=True, auth=("ed", "bananas"))
```
- Coupling with openai call
```py
message_input = gr.Textbox(label="Your message:", info="Enter a message to be shouted", lines=7)
message_output = gr.Textbox(label="Response:", lines=8)
#
view = gr.Interface(
    fn=shout,
    title="Shout", 
    inputs=[message_input], 
    outputs=[message_output], 
    examples=["hello", "howdy"], 
    flagging_mode="never"
    )
view.launch()
# And now - changing the function from "shout" to "message_gpt"
message_input = gr.Textbox(label="Your message:", info="Enter a message for GPT-4.1-mini", lines=7)
message_output = gr.Textbox(label="Response:", lines=8)
view = gr.Interface(
    fn=message_gpt,
    title="GPT", 
    inputs=[message_input], 
    outputs=[message_output], 
    examples=["hello", "howdy"], 
    flagging_mode="never"
    )
view.launch()
```

### 46. Day 2 - Markdown Responses and Streaming with Gradio and OpenAI
- Gradio can handle the python generator results
  - Using `stream=True` in client.chat.completion.create()

### 47. Day 2 - Building Multi-Model Gradio UIs with GPT and Claude Streaming
- Rebuild the instructor's code using ollama models:
```py
from openai import OpenAI
import os
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
def message_gemma3(prompt):
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    response = client.chat.completions.create(model="gemma3:270m", messages=messages)
    return response.choices[0].message.content
def stream_gemma3(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = client.chat.completions.create(
        model='gemma3:270m',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result
def stream_llama(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = client.chat.completions.create(
        model='llama3.2:1b',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result
def stream_model(prompt, model):
    if model=="gemma3":
        result = stream_gemma3(prompt)
    elif model=="llama":
        result = stream_llama(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result
message_input = gr.Textbox(label="Your message:", info="Enter a message for the LLM", lines=7)
model_selector = gr.Dropdown(["gemma3", "llama"], label="Select model", value="gemma3")
message_output = gr.Markdown(label="Response:")
view = gr.Interface(
    fn=stream_model,
    title="LLMs", 
    inputs=[message_input, model_selector], 
    outputs=[message_output], 
    examples=[
            ["Explain the Transformer architecture to a layperson", "gemma3"],
            ["Explain the Transformer architecture to an aspiring AI engineer", "llama"]
        ], 
    flagging_mode="never"
    )
view.launch()
```
- Error:
```bash
/anaconda3/2023.07/lib/python3.11/site-packages/gradio/queueing.py", line 171, in _get_df
    .infer_objects(copy=False)  # type: ignore
     ^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: NDFrame.infer_objects() got an unexpected keyword argument 'copy'
```
- Upgrade pandas: `pip3 install --upgrade pandas`

### 48. Day 3 - Building Chat UIs with Gradio: Your First Conversational AI Assistant
- The use of prompts with our assistant
  - The system prompt: Establishes ground rule. Provides critical background context
  - Context: During the conversation, insert context to give more relevant background information pertaining to the topic
  - Multi-shot prompt: Provide example conversations for specific scenarios
- Write a new call back, including the message and history, to provide context
- Basic UI test:
```py
import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
def chat(message, history):
    return "bananas"
gr.ChatInterface(fn=chat, type="messages").launch()
```
- Error: argument type is not recognized
- Remove `type=...` then it works OK

### 49. Day 3 - Building a Streaming Chatbot with Gradio and OpenAI API
```py
from openai import OpenAI
import os
import gradio as gr
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
system_message = "You are a helpful assistant"
def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model="gemma3:270m", messages=messages)
    return response.choices[0].message.content
gr.ChatInterface(fn=chat).launch()
```
- ![UI](./ch049_UI.png)


### 50. Day 3 - System Prompts, Multi-Shot Prompting, and Your First Look at RAG
- Oneshot prompting example:
```py
system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get."
```
- Multi-shot prompting example:
```py
system_message += "\nIf the customer asks for shoes, you should respond that shoes are not on sale today, \
but remind the customer to look at hats!"
```
  - Provides additional info for different scenarios

### 51. Day 4 - How LLM Tool Calling Really Works (No Magic, Just Prompts)
- Tools
  - Allows Frontier models to connect with external functions
    - Richer response by extending knowledge
    - Ability to carry out actions within the application
    - Enhanced capabilities, like calculations
  - We give an LLM the ability to run code
- Tool calling in theory
  - An LLM decides to run code on my computer
  - LLM doesn't run tool or SQLITE or DB
  - Those codes run tool or SQLITE/DB for LLM

### 52. Day 4 - Common Use Cases for LLM Tools and Agentic AI Workflows
- Common use cases for tools
  - Fetch data or add knowledge or context
  - Take action, like booking a meeting
  - Perform calculations or run code
  - Modify the UI
- There are TWO other ways to use tools than form basis of Agentic AI:
  - A tool can be used to maek another call to an LLM
  - A tool can be used to track a ToDo list and track progress towards a goal

### 53. Day 4 - Building an Airline AI Assistant with Tool Calling in OpenAI and Gradio
- At client.chat.completions.create(), `tools=` keyword is to hook with json data of tools feature (or Python dictionary data)
- Returned message has member data of `tool_calls`
- We implement a function, `handle_tool_call()` to extract tool related results from LLM message
  - `tool_call.id` becomes available
```py
system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""
# Let's start by making a useful function
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}
def get_ticket_price(destination_city):
    print(f"Tool called for city {destination_city}")
    price = ticket_prices.get(destination_city.lower(), "Unknown ticket price")
    return f"The price of a ticket to {destination_city} is {price}"
# There's a particular dictionary structure that's required to describe our function:
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}    
# And this is included in a list of tools:
tools = [{"type": "function", "function": price_function}]
# We have to write that function handle_tool_call:
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    if tool_call.function.name == "get_ticket_price":
        arguments = json.loads(tool_call.function.arguments)
        city = arguments.get('destination_city')
        price_details = get_ticket_price(city)
        response = {
            "role": "tool",
            "content": price_details,
            "tool_call_id": tool_call.id
        }
    return response
def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

```

### 54. Day 4 - Handling Multiple Tool Calls with OpenAI and Gradio
- Cases of multiple calls
  - Implement `handle_tool_calls()`
```py
def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses
def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    if response.choices[0].finish_reason=="tool_calls": # a criterion to finish the loop
        message = response.choices[0].message
        responses = handle_tool_calls(message)
        messages.append(message)
        messages.extend(responses)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content    
```

### 55. Day 4 - Building Tool Calling with SQLite Database Integration
```py
import sqlite3
DB = "prices.db"
with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)')
    conn.commit()
## setting up prices per city
def set_ticket_price(city, price):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO prices (city, price) VALUES (?, ?) ON CONFLICT(city) DO UPDATE SET price = ?', (city.lower(), price, price))
        conn.commit()
ticket_prices = {"london":799, "paris": 899, "tokyo": 1420, "sydney": 2999}
for city, price in ticket_prices.items():
    set_ticket_price(city, price)
##     
def get_ticket_price(city):
    print(f"DATABASE TOOL CALLED: Getting price for {city}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT price FROM prices WHERE city = ?', (city.lower(),))
        result = cursor.fetchone()
        return f"Ticket price to {city} is ${result[0]}" if result else "No price data available for this city"
```
- Storing message history into DB would be one alternative
- In production environment, we may not need JSON - mostly use framework

### 56. Day 5 - Introduction to Agentic AI and Building Multi-Tool Workflows
- Defining agents
  - LLM that controls the workflow
  - An LLM agent runs tools in a loop to achieve a goal
- Common features
  - Memory/persistence
  - Planning capabilities
  - Autonomy
  - LLM orchestration via tools
  - Functionality via tools
- We will
  - Image generation 
    - Use the OpenAI interface to generate images
  - Make separate LLM calls
    - Create agents to generate sound and images
  - Combine LLM calls into one solution
    - Teach our AI assistant to speak and draw

### 57. Day 5 - How Gradio Works: Building Web UIs from Python Code

### 58. Day 5 - Building Multi-Modal Apps with DALL-E 3, Text-to-Speech, and Gradio Bloc
```py
# Some imports for handling images
import base64
from io import BytesIO
from PIL import Image
def artist(city):
    image_response = openai.images.generate(
            model="dall-e-3",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))
image = artist("New York City")
display(image)
#
def talker(message):
    response = openai.audio.speech.create(
      model="gpt-4o-mini-tts",
      voice="onyx",    # Also, try replacing onyx with alloy or coral
      input=message
    )
    return response.content
def chat(history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    cities = []
    image = None
    while response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        responses, cities = handle_tool_calls_and_return_cities(message)
        messages.append(message)
        messages.extend(responses)
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]
    voice = talker(reply)
    if cities:
        image = artist(cities[0])
    return history, voice, image
def handle_tool_calls_and_return_cities(message):
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            cities.append(city)
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses, cities    
```
- 3 types of Gradio UI
  - gr.Interface for standard, simple UI
  - gr.ChatInterface for standard Chatbot UIs
  - gr.Blocks is for custom UIs

### 59. Day 5 - Running Your Multimodal AI Assistant with Gradio and Tools
```py
# Callbacks (along with the chat() function above)
def put_message_in_chatbot(message, history):
        return "", history + [{"role":"user", "content":message}]
# UI definition
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500, interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI Assistant:")
# Hooking up events to callbacks
    message.submit(put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output]
    )
ui.launch(inbrowser=True, auth=("ed", "bananas"))
```

## Section 3: NEW Week 3 - Open-Source Gen AI: Automated Solutions with HuggingFace

### 60. Day 1 - Introduction to Hugging Face Platform: Models, Datasets, and Spaces
- Hugging Face Platfrom
  - An ubiquitous platform for LLM engineers
  - Models: over 2M open source models of all shapes and sizes. 
  - Datasets: A treasure trove of 0.5M datasets
  - Spaces: apps, many built in Gradio, including Leaderboards
- Free Hugging Face account is required for this training
  - Not pro account

### 61. Day 1 - HuggingFace Libraries: Transformers, Datasets, and Hub Explained
- Hugging face libraries compared to using Ollama
  - hub
  - datasets
  - transformers
  - peft (parameter efficient fine training)
  - trl (transformers reinforcement library)
  - accelerate

### 62. Day 1 - Introduction to Google Colab and Cloud GPUs for AI Development
- Google colab
  - Run a jupyter notebook in the cloud with a powerful runtime
  - Collaborate with others
  - Integate with other Google services
- Runtimes
  - CPU based
  - Lower spec GPU for free or low-cost - T4(15GB GPU RAM)
  - HIghe spec GPU for resource intensive runs - A100 (40GB)

### 63. Day 1 - Getting Started with Google Colab: Setup, Runtime, and Free GPU Access
- Downside of colab
  - You might be off the box from Google anytime
  - Need pip install packages every single time
  - Some latency

### 64. Day 1 - Setting Up Google Colab with Hugging Face and Running Your First Model
- Jupyter note book is the default interface
- To use Linux CLI, use `!`
  - `!pip install gradio`
- User keys or tokens
  - From colab login menu -> Settings -> Access Tokens -> Create Tokens
  - In the jupyter notebook, from ions in the left pane -> keys-> Secrets
    - Can control each notebook to access keys/secrets
  - Accessing your secret keys in Python:
```py
from google.colab import userdata
userdata.get('OPENAI_API_KEY')
```

### 65. Day 1 - Running Stable Diffusion and FLUX on Google Colab GPUs
```py
from huggingface_hub import login
from google.colab import userdata
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
from IPython.display import display
from diffusers import AutoPipelineForText2Image
import torch
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")
prompt = "A class of students learning AI engineering in a vibrant pop-art style"
image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
display(image)
# Restart the kernel
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
from IPython.display import display
from diffusers import DiffusionPipeline
import torch
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
prompt = "A class of data scientists learning AI engineering in a vibrant high-energy pop-art style"
image = pipe(prompt=prompt, num_inference_steps=30).images[0]
display(image)
```

### 66. Day 2 - Introduction to Hugging Face Pipelines for Quick AI Inference
- Two API levels of Hugging Face
  - Pipelines: Higher level APIs to carry out standard tasks incredibly quickly
    - Sentimental analysis
    - Classifier
    - Named entity recognition
    - Question answering
    - Summarizing
    - Translation    
  - Tokenizers and Models: Lower level APIs to provide the most power and control
- Use pipelines to generate content
  - Text
  - Image
  - Audio

### 67. Day 2 - HuggingFace Pipelines API for Sentiment Analysis on Colab T4 GPU
```py
!pip install -q --upgrade datasets==3.6.0
# Imports
import torch
from google.colab import userdata
from huggingface_hub import login
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio
```
- Using pipeline
  - `my_pipeline = pipeline(task, model=xx, device=xx)`
  - `result = my_pipeline(input1)`
```py
better_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device="cuda")
result = better_sentiment("I should be more excited to be on the way to LLM mastery!!")
print(result)
```

### 68. Day 2 - Named Entity Recognition, Q&A, and Hugging Face Pipeline Tasks
```py
# Named Entity Recognition
ner = pipeline("ner", device="cuda")
result = ner("AI Engineers are learning about the amazing pipelines from HuggingFace in Google Colab from Ed Donner")
for entity in result:
  print(entity)
# Question Answering with Context
question="What are Hugging Face pipelines?"
context="Pipelines are a high level API for inference of LLMs with common tasks"
question_answerer = pipeline("question-answering", device="cuda")
result = question_answerer(question=question, context=context)
print(result)
# Text Summarization
summarizer = pipeline("summarization", device="cuda")
text = """
The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP).
It allows users to perform a wide range of tasks such as text classification, named entity recognition, and question answering, among others.
It's an extremely popular library that's widely used by the open-source data science community.
It lowers the barrier to entry into the field by providing Data Scientists with a productive, convenient way to work with transformer models.
"""
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])
# Translation
translator = pipeline("translation_en_to_fr", device="cuda") # no model is given but HF will use a default one
result = translator("The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API.")
print(result[0]['translation_text'])
```

### 69. Day 2 - Hugging Face Pipelines: Image, Audio & Diffusion Models in Colab
```py
# Image Generation - remember this?! Now you know what's going on
# Pipelines can be used for diffusion models as well as transformers
from IPython.display import display
from diffusers import AutoPipelineForText2Image
import torch
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")
prompt = "A class of students learning AI engineering in a vibrant pop-art style"
image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
display(image)
# Audio Generation
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
from IPython.display import Audio
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='cuda')
embeddings_dataset = load_dataset("matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
speech = synthesiser("Hi to an artificial intelligence engineer, on the way to mastery!", forward_params={"speaker_embeddings": speaker_embedding})
Audio(speech["audio"], rate=speech["sampling_rate"])
```

### 70. Day 3 - Tokenizers: How LLMs Convert Text to Numbers
- Introducing the Tokenizer
  - Maps b/w text and tokens for a particular model
  - Translates b/w text, tokens and token IDs with encode() and decode() methods
  - Contains a Vocab that can include special tokens to signal information to th elLM, like the start of prompt

### 71. Day 3 - Tokenizers in Action: Encoding and Decoding with Llama 3.1
```py
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)
text = "I am excited to show Tokenizers in action to my LLM engineers"
tokens = tokenizer.encode(text)
tokens
character_count = len(text)
word_count = len(text.split(' '))
token_count = len(tokens)
print(f"There are {character_count} characters, {word_count} words and {token_count} tokens")
# 61 characters, 12 words and 15 tokens
```
- In order to use Llama, Meta requires to sign up
  - To be granted, register email which was used to register HF 
  - For purpose, personal education/AI experimentalist will suffice

### 72. Day 3 - How Chat Templates Work: LLaMA Tokenizers and Special Tokens
- LLM models
  - Base model
  - Chat model
  - Reasoning model
- Instruct variants of models
  - HF trains models for chats
  - *-Instruct models
- LLM special tokens
  - `<|start_header_id|>`, `<|end_header_id>`, `<|eot_id|>`, ...

### 73. Day 3 - Comparing Tokenizers: Phi-4, DeepSeek, and QWENCoder in Action
```py
PHI4 = "microsoft/Phi-4-mini-instruct"
DEEPSEEK = "deepseek-ai/DeepSeek-V3.1"
QWEN_CODER = "Qwen/Qwen2.5-Coder-7B-Instruct"
phi4_tokenizer = AutoTokenizer.from_pretrained(PHI4)
text = "I am curiously excited to show Hugging Face Tokenizers in action to my LLM engineers"
print("Llama:")
tokens = tokenizer.encode(text)
print(tokens)
print(tokenizer.batch_decode(tokens))
print("\nPhi 4:")
tokens = phi4_tokenizer.encode(text)
print(tokens)
print(phi4_tokenizer.batch_decode(tokens))
print("Llama:")
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
print("\nPhi 4:")
print(phi4_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
deepseek_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK)
text = "I am curiously excited to show Hugging Face Tokenizers in action to my LLM engineers"
print(tokenizer.encode(text))
print()
print(phi4_tokenizer.encode(text))
print()
print(deepseek_tokenizer.encode(text))
print("Llama:")
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
print("\nPhi:")
print(phi4_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
print("\nDeepSeek:")
print(deepseek_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
```

### 74. Day 4 - Deep Dive into Transformers, Quantization, and Neural Networks
- Quantization
  - LLM handles lots of parameters, big matrix calculations, 16-32bit numbers
  - Store those numbers into smaller digit like 8bits or 4bits
  - Squeeze into smaller memory/less calculations
  - Affects neural network not much
- Model internals
- Streaming

### 75. Day 4 - Working with Hugging Face Transformers Low-Level API and Quantization
```py
!pip install -q --upgrade bitsandbytes accelerate
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
# instruct models and 1 reasoning model
# Llama 3.1 is larger and you should already be approved
# see here: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
# LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# Llama 3.2 is smaller but you might need to request access again
# see here: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
LLAMA = "meta-llama/Llama-3.2-1B-Instruct"
PHI = "microsoft/Phi-4-mini-instruct"
GEMMA = "google/gemma-3-270m-it"
QWEN = "Qwen/Qwen3-4B-Instruct-2507"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
messages = [
    {"role": "user", "content": "Tell a joke for a room of Data Scientists"}
  ]
# Quantization Config - this allows us to load the model into memory and use less memory
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)  # reduces into 4bit
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token # eos = end of sentence
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
# The model
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config) 
memory = model.get_memory_footprint() / 1e6
print(f"Memory footprint: {memory:,.1f} MB")
# yields 1,012.0 MB
```
- NF4 data type: a low-precision representation method used in machine learning, specifically designed to efficiently quantize the weights of large language models (LLMs). It uses only 4 bits to store floating-point values, significantly reducing memory usage and accelerating inference and finetuning. 

### 76. Day 4 - Inside LLaMA: PyTorch Model Architecture and Token Embeddings
- Looking under the hood at the Transformer model
  - This model object is a Neural Network, implemented with the Python framework PyTorch. The Neural Network uses the architecture invented by Google scientists in 2017: the Transformer architecture.
  - While we're not going to go deep into the theory, this is an opportunity to get some intuition for what the Transformer actually is.
  - Now take a look at the layers of the Neural Network that get printed in the next cell. Look out for this:
    - It consists of layers
    - There's something called "embedding" - this takes tokens and turns them into 4,096 dimensional vectors. We'll learn more about this in Week 5.
    - There are then 16 sets of groups of layers (32 for Llama 3.1) called "Decoder layers". Each Decoder layer contains three types of layer: (a) self-attention layers (b) multi-layer perceptron (MLP) layers (c) batch norm layers.
    - There is an LM Head layer at the end; this produces the output
      - This is opposite of Embedding layer
  - Notice the mention that the model has been quantized to 4 bits.

### 77. Day 4 - Inside LLaMA: Decoder Layers, Attention, and Why Non-Linearity Matters
- LlamaForCausalLM()
  - (model): LlamaModel
    - (embed_tokens): Embedding(128256,2048)
    - (layers):
      - (0-15): 16 x LlamaDecoderLayer
        - (self_attn): LlamaAtention
          - (q_proj): Linear4bit()
          - (k_proj): Linear4bit()
          - (v_proj): Linear4bit()
          - (o_proj): Linear4bit()
        - (mlp): LlamaMLP
        - (input_layernorm): LlamaRMSNorm
        - (post_attention_layernorm): LlamaRMSNorm
    - (norm): LlamaRMSNorm(2048,)
    - (rotary_emb): LlamaRotaryEmbedding()
  - (lm_head): Linear(in_features=2049, out_features=128256)

### 78. Day 4 - Running Open Source LLMs: Phi, Gemma, Qwen & DeepSeek with Hugging Face
```py
# Wrapping everything in a function - and adding Streaming and generation prompts
def generate(model, messages, quant=True, max_new_tokens=80):
  tokenizer = AutoTokenizer.from_pretrained(model)
  tokenizer.pad_token = tokenizer.eos_token
  input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
  attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
  streamer = TextStreamer(tokenizer)
  if quant:
    model = AutoModelForCausalLM.from_pretrained(model, quantization_config=quant_config).to("cuda")
  else:
    model = AutoModelForCausalLM.from_pretrained(model).to("cuda")
  outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, streamer=streamer)
messages = [
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]
generate(GEMMA, messages, quant=False)
generate(QWEN, messages)
generate(DEEPSEEK, messages, quant=False, max_new_tokens=500)
```

### 79. Day 5 - Visualizing Token-by-Token Inference in GPT Models
- Create a solution that makes a meeting minutes
  - Use a Frontier model to convert the audio to text
  - Use an open source model to generate minutes
  - Stream back results and show in Markdown
- How the next token is selected
  - By statistics

### 80. Day 5 - Building Meeting Minutes from Audio with Whisper and Google Colab
```py
!pip install -q --upgrade bitsandbytes accelerate
# imports
import os
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from google.colab import drive
from huggingface_hub import login
from google.colab import userdata
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
# New capability - connect this Colab to your Google Drive
# See immediately below this for instructions to obtain denver_extract.mp3
# Place the file on your drive in a folder called llms, and call it denver_extract.mp3
drive.mount("/content/drive")
audio_filename = "/content/drive/MyDrive/llms/denver_extract.mp3"
# Sign in to HuggingFace Hub
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
# Open the file
audio_file = open(audio_filename, "rb")
# Step 1: Transcribe Audio
## Opiton 1: using HF pipelines
from transformers import pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium.en",
    dtype=torch.float16,
    device='cuda',
    return_timestamps=True
)
result = pipe(audio_filename)
transcription = result["text"]
print(transcription)
```

### 81. Day 5 - Building Meeting Minutes with OpenAI Whisper and LLaMA 3.2
```py
# Option 2: using OpenAI
AUDIO_MODEL = "gpt-4o-mini-transcribe"
openai_api_key = userdata.get('OPENAI_API_KEY')
openai = OpenAI(api_key=openai_api_key)
transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio_file, response_format="text")
print(transcription)
#
open_source_transcription = transcription
display(Markdown(open_source_transcription))
print("\n\n")
display(Markdown(transcription))
# Step 2: Analyze and report
system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks.
"""
user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners
Transcription:
{transcription}
"""
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
response = tokenizer.decode(outputs[0])
display(Markdown(response))
```

### 82. Day 5 - Week 3 Wrap-Up: Build a Synthetic Data Generator with Open Source Models
- Generating Synthetic data
  - Write models that can generate datasets
  - Use a variety of models and prompts for diverse outputs
  - Create a Gradio UI for your product

## Section 4: New Week 4 - LLM Showdown: Evaluating Models for Code Gen & Business Tasks

### 83. Day 1 - Choosing the Right LLM: Model Selection Strategy and Basics
- What's the right LLM for the task at hand
  - Start with the basics
    - Parameters
    - Context length
    - Pricing
  - THen look at the results
    - Benchmark
    - Leaderboards
    - Arenas
- Comparing LLMs
  - The Basics (1)
    - Open source or closed
    - Chat/Reasoning/Hybrid
    - Release date and konwlege cut-off
    - Parameters
    - Training tokens
    - Context window
  - The Basics (2)
    - Inference cost: API charge or runtime compute
    - Training cost
    - Build cost
    - Time to market
    - Rate limits
    - Speed
    - Latency
    - License

### 84. Day 1 - The Chinchilla Scaling Law: Parameters, Training Data and Why It Matters
- Chinchilla scaling law
  - Number of parameters is proportional to the number of training tokens
  - If you upgrade a model with double number of weights, you need twice the data

### 85. Day 1 - Understanding AI Model Benchmarks: GPQA, MMLU-Pro, and HLE
- 6 hard, next-level benchmarks
  - GPQA from Google
    - PHD science expertise
    - 448 expert questions
  - MMLU-PRO
    - Language understanding
    - A more advanced and cleaned up version of Massive Multitask Language Undrstanding including choice of 10 instead of 4
  - AIME
    - Math
    - Hard competitive Math puzzles from the prestigious, invite-only Math competition for top high schoolers
  - LIveCode Bench
    - Coding
    - Holistic benchmark for evaluating Code LLMs based on problems from contests on LeetCode, AtCoder and Codeforces
  - MuSR
    - Reasoning
    - Logical deduction, such as analyzing 1,000 word murder mystery and answeroing: who has means, motive, and opportunity?
  - HLE (Humanity's Last Exam)
    - Super-human intelligence
    - 2,500 of the toughest, subject-diverse, multi-modal questions designed to be the last academic exame of its kind for AI

### 86. Day 1 - Limitations of AI Benchmarks: Data Contamination and Overfitting
- Limitations of Benchmarks
  - Training data contamination
  - Not consistentl applied
  - Too narrow in scope
  - Hard to measure nuanced reasoning
  - Saturation
  - Overfitting
- A new concern, not yet proven
  - Frontier LLMs may be aware that they are being evaluated

### 87. Day 1 - Build a Connect Four Leaderboard (Reasoning Benchmark)

### 88. Day 2 - Navigating AI Leaderboards: Artificial Analysis, HuggingFace & More
- Five leaderboards
  - Artificial analysis
    - THE leaderboard
  - Vellum
    - Includes API cost and context window
  - Scale.com
    - SEAL leaderboards
  - Hugging Face
    - Open source leaderboards
  - Live Bench
    - Contamination-free

### 89. Day 2 - Artificial Analysis Deep Dive: Model Intelligence vs Cost Comparison
- Artificial analysis
  - https://artificialanalysis.ai/
  - Artificial Analysis Intelligence Index v3.0 incorporates 10 evaluations: MMLU-Pro, GPQA Diamond, Humanity's Last Exam, LiveCodeBench, SciCode, AIME 2025, IFBench, AA-LCR, Terminal-Bench Hard, 𝜏²-Bench Telecom

### 90. Day 2 - Vellum, SEAL, and LiveBench: Essential AI Model Leaderboards
- Vellum
  - https://www.vellum.ai/llm-leaderboard?utm_source=google&utm_medium=organic#
  - context window size
  - Cached tokens
- Scale.com
  - https://scale.com/leaderboard
  - https://scale.com/leaderboard/humanitys_last_exam
- Hugging Face
  - https://huggingface.co/open-llm-leaderboard
- Live Bench
  - https://livebench.ai/#/

### 91. Day 2 - LM Arena: Blind Testing AI Models with Community Elo Ratings
- LM Arena (formerly LMSYS)
  - https://lmarena.ai/
  - Compares Frontier and opensource models directly
  - Blind human evals based on head-to-head comparison
  - LLMs are measured with an ELO-style score rating

### 92. Day 2 - Commercial Use Cases: Automation, Augmentation & Agentic AI
- Commercial use cases: Automate, then augment, then differentiate
  - ChatGTP wrapper
    - DuoLingo
    - Copilots
  - Specialized
    - Harvey
    - nebular.io
    - Khanmigo
    - Salesforce Health
    - Palantir
  - Agentic
    - Claude Code
    - OpenAI Codex
    - OpenAI Agent

### 93. Day 3 - Selecting LLMs for Code Generation: Python to C++ with Cursor
- Your duty as an AI engineer
  - What business problem are you solving?
- An AI engineer wears 2 hats
  - Data scienstist
    - What business problem?
    - How you measure the success?
    - What data do you have, and do you need?
  - Software engineer  
    - Architecture
    - Framework
    - DB
    - How to integrate?
- The 5 step strategy: to select, train, and apply LLM to a commercial problem
  - Understand
  - Prepare
  - Select
  - Customize
  - Productionize

### 94. Day 3 - Selecting Frontier Models: GPT-5, Claude, Grok & Gemini for C++ Code Gen
```py
import os
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
from IPython.display import Markdown, display
from system_info import retrieve_system_info
system_info = retrieve_system_info()
system_info
message = f"""
Here is a report of the system information for my computer.
I want to run a C++ compiler to compile a single C++ file called main.cpp and then execute it in the simplest way possible.
Please reply with whether I need to install any C++ compiler to do this. If so, please provide the simplest step by step instructions to do so.
If I'm already set up to compile C++ code, then I'd like to run something like this in Python to compile and execute the code:
------------
compile_command = # something here - to achieve the fastest possible runtime performance
compile_result = subprocess.run(compile_command, check=True, text=True, capture_output=True)
run_command = # something here
run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
return run_result.stdout
------------
Please tell me exactly what I should use for the compile_command and run_command.
System information:
{system_info}
"""
response = openai.chat.completions.create(model=OPENAI_MODEL, messages=[{"role": "user", "content": message}])
display(Markdown(response.choices[0].message.content))
```

### 95. Day 3 - Porting Python to C++ with GPT-5: 230x Performance Speedup
- Conversion using ollama model
```py
import os
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
from IPython.display import Markdown, display
message = f"""
Here is a report of the system information for my computer.
I want to run a C++ compiler to compile a single C++ file called main.cpp and then execute it in the simplest way possible.
Please reply with whether I need to install any C++ compiler to do this. If so, please provide the simplest step by step instructions to do so.
If I'm already set up to compile C++ code, then I'd like to run something like this in Python to compile and execute the code:
------------
compile_command = # something here - to achieve the fastest possible runtime performance
compile_result = subprocess.run(compile_command, check=True, text=True, capture_output=True)
run_command = # something here
run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
return run_result.stdout
------------
Please tell me exactly what I should use for the compile_command and run_command.
System information:

"""
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(model="gemma3:270m", messages=[{"role": "user", "content": message}])
display(Markdown(response.choices[0].message.content))
system_prompt = """
Your task is to convert Python code into high performance C++ code.
Respond only with C++ code. Do not provide any explanation other than occasional comments.
The C++ response needs to produce an identical output in the fastest possible time.
"""
def user_prompt_for(python):
    return f"""
Port this Python code to C++ with the fastest possible implementation that produces identical output in the least time.
The system information is:
Your response will be written to a file called main.cpp and then compiled and executed; the compilation command is:
Respond only with C++ code.
Python code to port:
------ 
{python}
------
"""
def messages_for(python):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(python)}
    ]
def write_output(cpp):
    with open("main.cpp", "w", encoding="utf-8") as f:
        f.write(cpp)
pi = """
import time
def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(200_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""
def run_python(code):
    globals = {"__builtins__": __builtins__}
    exec(code, globals)
run_python(pi)
def port(client, model, python):
    mess = messages_for(python)
    print(mess)
    response = client.chat.completions.create(model=model, messages=mess)
    reply = response.choices[0].message.content
    reply = reply.replace('```cpp','').replace('```','')
    write_output(reply)
    print(reply)
port(client, "gemma3:270m", pi)
```
- Generated cpp code:
```cpp
#include <iostream>
#include <algorithm>
using namespace std::endl;
int calculate(int iterations, int param1, int param2) {
  result = 1.0;
  for (int i = 0; i < iterations; ++i) {
    result -= (1/i) * param1;
    result += (1/i) * param2;
  }
  return result;
}
int main() {
  int iterations = 200;
  int param1 = 4;
  int param2 = 1;

  std::cout << "Result: " << calculate(iterations, param1, param2) << std::endl;
  std::cout << "Execution Time: {(end_time - start_time):.6f} seconds" << std::endl;
  return 0;
}
```

### 96. Day 3 - AI Coding Showdown: GPT-5 vs Claude vs Gemini vs Groq Performance
- With higher reasoning options, Grok/Gemini optimize code using multi-threading
  - Conversion cost is very high (several min) but those optimized codes outperform other LLM generated codes (184x -> 1440x than Python)

### 97. Day 4 - Open Source Models for Code Generation: Qwen, DeepSeek & Ollama
- Big Code Models Leaderboard: https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard


### 98. Day 4 - Building a Gradio UI to Test Python-to-C++ Code Conversion Models
```py
with gr.Blocks() as ui:
    with gr.Row():
        python = gr.Textbox(label="Python code:", lines=28, value=pi)
        cpp = gr.Textbox(label="C++ code:", lines=28)
    with gr.Row():
        model = gr.Dropdown(models, label="Select model", value=models[0])
        convert = gr.Button("Convert code")

    convert.click(port, inputs=[model, python], outputs=[cpp])
ui.launch(inbrowser=True)
```

### 99. Day 4 - Qwen 3 Coder vs GPT OSS: OpenRouter Model Performance Showdown

### 100. Day 5 - Model Evaluation: Technical Metrics vs Business Outcomes
- How to evaluate the performance of a Gen AI solution?
  - Model centric or Technical Metrics
    - Loss (eg cross-entropy loss)
    - Perplexity
    - Accuracy
    - Precision, Recall, F1
    - AUC-ROC
    - Easiest to optimize with
  - Business-centric or Outcome Metrics
    - KPIs tied to business objectives
    - ROI
    - Improvements in time, cost or resources
    - Customer satisfaction
    - Benchmark comparisons
    - Most tangible impact

### 101. Day 5 - Python to Rust Code Translation: Testing Gemini 2.5 Pro with Cursor
- Business centric metrics in this lesson
  - Performance of C++ solution with identical results
  - So far Gemini2.5 Pro is the winner
  - Time for a harder test
  
### 102. Day 5 - Porting Python to Rust: Testing GPT, Claude, and Qwen Models

### 103. Day 5 - Open Source Model Wins? Rust Code Generation Speed Challenge

## Section 5: New Week 5 - Mastering RAG: Build Advanced Solutions with Vector Embeddings

### 104. Day 1 - Introduction to RAG: Retrieval Augmented Generation Fundamentals
### 105. Day 1 - Building a Simple RAG Knowledge Assistant with GPT-4-1 Nano
### 106. Day 1 - Building a Simple RAG System: Dictionary Lookup and Context Retrieval
### 107. Day 1 - Vector Embeddings and Encoder LLMs: The Foundation of RAG
### 108. Day 1 - How Vector Embeddings Represent Meaning: From word2vec to Encoders
### 109. Day 1 - Understanding the Big Idea Behind RAG and Vector Data Stores
### 110. Day 2 - Vectors for RAG: Introduction to LangChain and Vector Databases
### 111. Day 2 - Breaking Documents into Chunks with LangChain Text Splitters
### 112. Day 2 - Encoder Models vs Vector Databases: OpenAI, BERT, Chroma & FAISS
### 113. Day 2 - Creating Vector Stores with Chroma and Visualizing Embeddings with t-SNE
### 114. Day 2 - 3D Vector Visualizations and Comparing Embedding Models
### 115. Day 3 - Building a Complete RAG Pipeline with LangChain and Chroma
### 116. Day 3 - Building a RAG Pipeline with LangChain: LLM & Retriever Setup
### 117. Day 3 - Building RAG with LangChain: Retriever and LLM Integration
### 118. Day 3 - Building Production RAG with Python Modules and Gradio UI
### 119. Day 3 - RAG with Conversation History: Building a Gradio UI and Debugging Chunki
### 120. Day 4 - RAG Evaluations: Measuring Performance and Iterating on Your Pipeline
### 121. Day 4 - Evaluating RAG Systems: Retrieval Metrics, LLM as Judge, and Golden Data
### 122. Day 4 - Evaluating RAG Systems: MRR, NDCG, and Test Data with Pydantic
### 123. Day 4 - LLM as a Judge: Evaluating RAG Answers with Structured Outputs
### 124. Day 4 - Running RAG Evaluations with Gradio: MRR, nDCG, and Test Results
### 125. Day 4 - Experimenting with Chunking Strategies and Embedding Models in RAG
### 126. Day 4 - Testing OpenAI Embeddings and Evaluating RAG Performance Gains
### 127. Day 5 - Advanced RAG Techniques: Pre-processing, Re-ranking & Evals
### 128. Day 5 - Advanced RAG Techniques: Chunking, Encoders, and Query Rewriting
### 129. Day 5 - Advanced RAG Techniques: Query Expansion, Re-ranking & GraphRAG
### 130. Day 5 - Building Advanced RAG Without LangChain: Semantic Chunking with LLMs
### 131. Day 5 - Creating Embeddings with Chroma, Visualizing with t-SNE, and Re-ranking
### 132. Day 5 - Building RAG Without LangChain: Re-ranking and Query Rewriting
### 133. Day 5 - Building Production RAG with Query Expansion and Multiprocessing
### 134. Day 5 - Advanced RAG Evaluation: From 0.73 to 0.91 MRR with GPT-4o
### 135. Day 5 - RAG Challenge: Beat My Results & Build Your Knowledge Worker

## 
### 136. Day 1 - Training, Datasets, and Generalization: Your Capstone Begins
### 137. Day 1 - Finetuning LLMs & The Price is Right Capstone Project Intro
### 138. Day 1 - Curating Datasets: Finding Data Sources and Building Training Sets
### 139. Day 1 - Curating Amazon Data with Hugging Face for Price Prediction
### 140. Day 1 - Exploring Amazon Dataset Distribution and Removing Duplicates
### 141. Day 1 - Weighted Sampling with NumPy and Uploading Datasets to Hugging Face
### 142. Day 2 - Five-Step Strategy for Selecting and Applying LLMs to Business Problems
### 143. Day 2 - The Five-Step AI Process & Productionizing with MLOps
### 144. Day 2 - Data Pre-processing with LLMs and Groq Batch Mode
### 145. Day 2 - Batch Processing with Groq API and JSONL Files for LLM Workflows
### 146. Day 2 - Batch Processing with Groq: Running 22K LLM Requests for Under $1
### 147. Day 3 - Building Baseline Models with Traditional ML and XGBoost
### 148. Day 3 - Building Your First Baseline with Random Pricer and Scikit-learn
### 149. Day 3 - Baseline Models and Linear Regression with Scikit-Learn
### 150. Day 3 - Bag of Words and CountVectorizer for Linear Regression NLP
### 151. Day 3 - Random Forest and XGBoost: Ensemble Models in Scikit-Learn
### 152. Day 4 - Training Your First Neural Network and Testing Frontier Models
### 153. Day 4 - Human Baseline Performance vs Machine Learning Models in PyTorch
### 154. Day 4 - Building Your First Neural Network with PyTorch
### 155. Day 4 - Testing GPT-4o-mini and Claude Opus Against Neural Networks
### 156. Day 4 - Testing Gemini 3, GPT-5.1, Claude 4.5 & Grok on Price Prediction
### 157. Day 5 - Fine-Tuning OpenAI Frontier Models with Supervised Fine-Tuning
### 158. Day 5 - Fine-Tuning GPT-4o Nano with OpenAI's API for Custom Models
### 159. Day 5 - Fine-Tuning GPT-4o-mini-nano: Running Jobs and Monitoring Training
### 160. Day 5 - Fine-Tuning Results: When GPT-4o-mini Gets Worse, Not Better
### 161. Day 5 - When Fine-Tuning Frontier Models Fails & Building Deep Neural Networks
### 162. Day 5 - Deep Neural Network Redemption: 289M Parameters vs Frontier Models

##

### 163. Day 1 - Mastering Parameter-Efficient Fine-Tuning: LoRa, QLoRA & Hyperparameters
### 164. Day 1 - Introduction to LoRA Adaptors: Low-Rank Adaptation Explained
### 165. Day 1 - QLoRA: Quantization for Efficient Fine-Tuning of Large Language Models
### 166. Day 1 - Optimizing LLMs: R, Alpha, and Target Modules in QLoRA Fine-Tuning
### 167. Day 1 - Parameter-Efficient Fine-Tuning: PEFT for LLMs with Hugging Face
### 168. Day 1 - How to Quantize LLMs: Reducing Model Size with 8-bit Precision
### 169. Day 1: Double Quantization & NF4: Advanced Techniques for 4-Bit LLM Optimization
### 170. Day 1 - Exploring PEFT Models: The Role of LoRA Adapters in LLM Fine-Tuning
### 171. Day 1 - Model Size Summary: Comparing Quantized and Fine-Tuned Models
### 172. Day 2 - How to Choose the Best Base Model for Fine-Tuning Large Language Models
### 173. Day 2 - Selecting the Best Base Model: Analyzing HuggingFace's LLM Leaderboard
### 174. Day 2 - Exploring Tokenizers: Comparing LLAMA, QWEN, and Other LLM Models
### 175. Day 2 - Optimizing LLM Performance: Loading and Tokenizing Llama 3.1 Base Model
### 176. Day 2 - Quantization Impact on LLMs: Analyzing Performance Metrics and Errors
### 177. Day 2 - Comparing LLMs: GPT-4 vs LLAMA 3.1 in Parameter-Efficient Tuning
### 178. Day 3 - QLoRA Hyperparameters: Mastering Fine-Tuning for Large Language Models
### 179. Day 3 - Understanding Epochs and Batch Sizes in Model Training
### 180. Day 3 - Learning Rate, Gradient Accumulation, and Optimizers Explained
### 181. Day 3 - Setting Up the Training Process for Fine-Tuning
### 182. Day 3 - Configuring SFTTrainer for 4-Bit Quantized LoRA Fine-Tuning of LLMs
### 183. Day 3 - Fine-Tuning LLMs: Launching the Training Process with QLoRA
### 184. Day 3 - Monitoring and Managing Training with Weights & Biases
### 185. Day 4 - Keeping Training Costs Low: Efficient Fine-Tuning Strategies
### 186. Day 4 - Efficient Fine-Tuning: Using Smaller Datasets for QLoRA Training
### 187. Day 4 - Visualizing LLM Fine-Tuning Progress with Weights and Biases Charts
### 188. Day 4 - Advanced Weights & Biases Tools and Model Saving on Hugging Face
### 189. Day 4 - End-to-End LLM Fine-Tuning: From Problem Definition to Trained Model
### 190. Day 5 - The Four Steps in LLM Training: From Forward Pass to Optimization
### 191. Day 5 - QLoRA Training Process: Forward Pass, Backward Pass and Loss Calculation
### 192. Day 5 - Understanding Softmax and Cross-Entropy Loss in Model Training
### 193. Day 5 - Monitoring Fine-Tuning: Weights & Biases for LLM Training Analysis
### 194. Day 5 - Revisiting the Podium: Comparing Model Performance Metrics
### 195. Day 5 - Evaluation of our Proprietary, Fine-Tuned LLM against Business Metrics
### 196. Day 5 - Visualization of Results: Did We Beat GPT-4?
### 197. Day 5 - Hyperparameter Tuning for LLMs: Improving Model Accuracy with PEFT

##

### 198. Day 1 - From Fine-Tuning to Multi-Agent Systems: Next-Level LLM Engineering
### 199. Day 1: Building a Multi-Agent AI Architecture for Automated Deal Finding Systems
### 200. Day 1 - Unveiling Modal: Deploying Serverless Models to the Cloud
### 201. Day 1 - LLAMA on the Cloud: Running Large Models Efficiently
### 202. Day 1 - Building a Serverless AI Pricing API: Step-by-Step Guide with Modal
### 203. Day 1 - Multiple Production Models Ahead: Preparing for Advanced RAG Solutions
### 204. Day 2 - Implementing Agentic Workflows: Frontier Models and Vector Stores in RAG
### 205. Day 2 - Building a Massive Chroma Vector Datastore for Advanced RAG Pipelines
### 206. Day 2 - Visualizing Vector Spaces: Advanced RAG Techniques for Data Exploration
### 207. Day 2 - 3D Visualization Techniques for RAG: Exploring Vector Embeddings
### 208. Day 2 - Finding Similar Products: Building a RAG Pipeline without LangChain
### 209. Day 2 - RAG Pipeline Implementation: Enhancing LLMs with Retrieval Techniques
### 210. Day 2 - Random Forest Regression: Using Transformers & ML for Price Prediction
### 211. Day 2 - Building an Ensemble Model: Combining LLM, RAG, and Random Forest
### 212. Day 2 - Wrap-Up: Finalizing Multi-Agent Systems and RAG Integration
### 213. Day 3 - Enhancing AI Agents with Structured Outputs: Pydantic & BaseModel Guide
### 214. Day 3 - Scraping RSS Feeds: Building an AI-Powered Deal Selection System
### 215. Day 3 - Structured Outputs in AI: Implementing GPT-4 for Detailed Deal Selection
### 216. Day 3 - Optimizing AI Workflows: Refining Prompts for Accurate Price Recognition
### 217. Day 3 - Mastering Autonomous Agents: Designing Multi-Agent AI Workflows
### 218. Day 4 - The 5 Hallmarks of Agentic AI: Autonomy, Planning, and Memory
### 219. Day 4 - Building an Agentic AI System: Integrating Pushover for Notifications
### 220. Day 4 Implementing Agentic AI: Creating a Planning Agent for Automated Workflows
### 221. Day 4 - Building an Agent Framework: Connecting LLMs and Python Code
### 222. Day 4 - Completing Agentic Workflows: Scaling for Business Applications
### 223. Day 5 - Autonomous AI Agents: Building Intelligent Systems Without Human Input
### 224. Day 5 - AI Agents with Gradio: Advanced UI Techniques for Autonomous Systems
### 225. Day 5 - Finalizing the Gradio UI for Our Agentic AI Solution
### 226. Day 5 Enhancing AI Agent UI: Gradio Integration for Real-Time Log Visualization
### 227. Day 5 - Analyzing Results: Monitoring Agent Framework Performance
### 228. Day 5 - AI Project Retrospective: 8-Week Journey to Becoming an LLM Engineer


