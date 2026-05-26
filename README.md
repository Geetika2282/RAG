# AI Retrieval Systems Repository

This repository contains implementations of different Retrieval-Augmented Generation (RAG) architectures, starting from basic RAG pipelines to advanced Agentic RAG systems.

The goal of this repo is to understand how modern AI systems retrieve information, reason over it, and generate accurate responses using Large Language Models (LLMs).

---

# Repository Structure

```txt
├── RAG/
├── RAG_Agent/
├── Agentic_RAG/
└── README.md
```

---

# 1. RAG

Basic Retrieval-Augmented Generation pipeline.

### What it is

RAG combines:

* Retrieval system
* Vector database
* Large Language Model

The system first retrieves relevant chunks from documents and then sends them to the LLM to generate an answer.

### Workflow

```txt
User Query
   ↓
Retrieve Relevant Chunks
   ↓
Pass Context to LLM
   ↓
Generate Answer
```

### Features

* Text chunking
* Embeddings generation
* FAISS vector store
* Similarity search
* Context-based answering

### Purpose

Used to reduce hallucination by grounding LLM responses in external data.

---

# 2. RAG Agent

RAG pipeline enhanced with agent-like behavior.

### What it is

Instead of following a fixed pipeline, the system can make simple decisions during execution.

The agent can:

* retrieve context
* summarize long context
* decide next action
* generate refined responses

### Workflow

```txt
User Query
   ↓
Retrieve Context
   ↓
Agent Decision Making
   ↓
Optional Summarization
   ↓
Generate Final Answer
```

### Features

* Modular tool-based architecture
* Retrieval tool
* Summarization tool
* Answer generation tool
* Dynamic workflow execution

### Purpose

Introduces reasoning and adaptability into traditional RAG systems.

---

# 3. Agentic RAG

Advanced autonomous retrieval and reasoning system.

### What it is

Agentic RAG uses AI agents that can plan, reason, and use multiple tools dynamically to solve complex tasks.

Unlike traditional RAG, the workflow is not fixed.

The agent can:

* decide what to retrieve
* rewrite queries
* call multiple tools
* validate answers
* retry retrieval
* reason step-by-step

### Workflow

```txt
User Query
   ↓
Planning & Reasoning
   ↓
Tool Selection
   ↓
Retrieval / Web Search / Memory
   ↓
Answer Validation
   ↓
Final Response
```

### Features

* Multi-step reasoning
* Autonomous decision making
* Query rewriting
* Multi-tool usage
* Context validation
* Memory integration
* Adaptive retrieval strategies

### Purpose

Designed for more intelligent AI systems capable of handling complex and dynamic tasks.

---

# Tech Stack

* Python
* LangChain
* FAISS
* Ollama
* LLMs
* Vector Embeddings

---

# Learning Goal

This repository demonstrates the evolution from:

```txt
Traditional RAG
      ↓
Agent-based RAG
      ↓
Fully Agentic AI Systems
```

It is intended for learning:

* Retrieval systems
* Vector databases
* LLM orchestration
* AI agents
* Production AI workflows
