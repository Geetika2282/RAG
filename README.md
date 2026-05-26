# AI Retrieval Systems Repository

This repository contains implementations of different Retrieval-Augmented Generation (RAG) architectures using locally hosted Llama models through Ollama.

---

# Repository Structure

```txt id="1vk7c2"
├── Vanilla RAG/
├── RAG Agent/
├── Agentic RAG/
└── README.md
```

---

## 1. Vanilla RAG

Basic Retrieval-Augmented Generation pipeline.

### Includes

* Text chunking
* Embeddings
* FAISS vector database
* Context retrieval
* LLM-based answering

### Workflow

```txt id="s7glwl"
Query → Retrieve Context → Generate Answer
```

---

## 2. RAG Agent

Enhanced RAG system with decision-making capabilities.

### Includes

* Retrieval tools
* Context summarization
* Dynamic response generation
* Tool-based workflow

### Workflow

```txt id="9d97yw"
Query → Retrieve → Agent Decision → Answer
```

---

## 3. Agentic RAG

Advanced AI agent system with autonomous reasoning and multi-step execution.

### Includes

* Query rewriting
* Multi-tool usage
* Context validation
* Adaptive retrieval
* Step-by-step reasoning

### Workflow

```txt id="5igzkr"
Query → Reasoning → Tool Usage → Validation → Final Answer
```

---

# Tech Stack

* Python
* LangChain
* FAISS
* Local Llama Models
* Ollama
