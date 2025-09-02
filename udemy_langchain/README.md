## 2025 Master Langchain and Ollama - Chatbot, RAG and Agents
- Instructor: Laxmi Kant | KGP Talkie

### Section 1: Introduction

#### 1. Introduction

#### 2. Code Files and Install Requirements.txt
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

### Section 2: Latest LLM Updates

#### 3. Run Deep Seek R1 Models Locally with Ollama

### Section 3: Ollama Setup

#### 4. Install Ollama

#### 5. Touch Base with Ollama

#### 6. Inspecting LLAMA 3.2 Model

#### 7. LLAMA 3.2 Benchmarking Overview
- Phi3.x doesn't support tool calling

#### 8. What Type of Models are Available on Ollama
- For embedding, we use nomic-embed-text

#### 9. Ollama Commands - ollama server, ollama show
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

#### 10. Ollama Commands - ollama pull, ollama list, ollama rm

#### 11. Ollama Commands - ollama cp, ollama run, ollama ps, ollama stop
- ollama cp llama3.2:1b llama.32:1b_myown
  - When we need to customize model files
- How to find the blob files of each model: ollama show llama3.2:latest --modelfile |grep -in sha256
- The location of model info in the model folder: ollama/models/manifests/registry.ollama.ai/library/llama3.2
```bash
$ ollama ps
NAME               ID              SIZE      PROCESSOR          CONTEXT    UNTIL              
llama3.2:latest    a80c4f17acd5    3.6 GB    49%/51% CPU/GPU    4096       3 minutes from now    
```

#### 12. Create and Run Ollama Model with Predefined Settings
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

#### 13. Ollama Model Commands - /show
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

#### 14. Ollama Model Commands - /set, /clear, /save_model and /load_model
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

#### 15. Ollama Raw API Requests
- Ref: https://github.com/ollama/ollama/blob/main/docs/api.md
- Use REST API with curl command
- Ex:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?"
}'
```

#### 16. Load Uncesored Models for Banned Content Generation [Only Educational Purpose]
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

### Section 4: Getting Started with Langchain

#### 17. Langchain Introduction | LangChain (Lang Chain) Intro
- https://www.langchain.com/
  - Supports Python and Javascript

#### 18. Lanchain Installation | LangChain (Lang Chain) Intro

#### 19. Langsmith Setup of LLM Observability | LangChain (Lang Chain) Intro

#### 20. Calling Your First Langchain Ollama API | LangChain (Lang Chain) Intro

#### 21. Generating Uncensored Content in Langchain [Educational Purpose]

#### 22. Trace LLM Input Output at Langsmith | LangChain (Lang Chain) Intro

#### 23. Going a lot Deeper in the Langchain | LangChain (Lang Chain) Intro


24. Why We Need Prompt Template
25. Type of Messages Needed for LLM
26. Circle Back to ChatOllama
27. Use Langchain Message Types with ChatOllama
28. Langchain Prompt Templates
29. Prompt Templates with ChatOllama

30. Introduction to LCEL
31. Create Your First LCEL Chain
32. Adding StrOutputParser with Your Chain
33. Chaining Runnables (Chain Multiple Runnables)
34. Run Chains in Parallel Part 1
35. Run Chains in Parallel Part 2
36. How Chain Router Works
37. Creating Independent Chains for Positive and Negative Reviews
38. Route Your Answer Generation to Correct Chain
39. What is RunnableLambda and RunnablePassthrough
40. Make Your Custom Runnable Chain
41. Create Custom Chain with chain Decorator


42. What is Output Parsing
43. What is Pydantic Parser
44. Get Pydantic Parser Instruction
45. Parse LLM Output Using Pydantic Parser
46. Parsing with `.with_structured_output()` method
47. JSON Output Parser
48. CSV Output Parsing - CommaSeparatedListOutputParser
49. Datetime Output Parsing



50. How to Save and Load Chat Message History (Concept)
51. Simple Chain Setup
52. Chat Message with History Part 1
53. Chat Message with History Part 2
54. Chat Message with History using MessagesPlaceholder



55. Introduction
56. Introduction To Streamlit and Our Chat Application
57. Chat Bot Basic Code Setup
58. Create Chat History in Streamlit Session State
59. Create LLM Chat Input Area with Streamlit
60. Update Historical Chat on Streamlit UI
61. Complete Your Own Chat Bot Application
62. Stream Output of Your Chat Bot like ChatGPT



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
