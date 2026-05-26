import ollama
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------------
# LOAD TEXT
# -----------------------------------

with open("intro.txt", "r", encoding="utf-8") as file:
    content = file.read()

print(content)

# -----------------------------------
# SIMPLE CHUNKING
# -----------------------------------

chunk_size = 200
chunk_overlap = 20

chunks = []

start = 0

while start < len(content):
    end = start + chunk_size

    chunk = content[start:end]

    chunks.append(chunk)

    start += chunk_size - chunk_overlap

print("\nTotal Chunks:", len(chunks))

# -----------------------------------
# EMBEDDING MODEL
# -----------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------------
# CREATE EMBEDDINGS
# -----------------------------------

chunk_embeddings = embedding_model.encode(chunks)

# Convert to float32 for FAISS
chunk_embeddings = np.array(
    chunk_embeddings,
    dtype=np.float32
)

# -----------------------------------
# CREATE FAISS INDEX
# -----------------------------------

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(chunk_embeddings)

# -----------------------------------
# USER QUERY
# -----------------------------------

query = input("\nAsk something about me: ")

# -----------------------------------
# QUERY EMBEDDING
# -----------------------------------

query_embedding = embedding_model.encode([query])

query_embedding = np.array(
    query_embedding,
    dtype=np.float32
)

# -----------------------------------
# SIMILARITY SEARCH
# -----------------------------------

k = 2

distances, indices = index.search(
    query_embedding,
    k
)

# -----------------------------------
# RETRIEVE RELEVANT CHUNKS
# -----------------------------------

retrieved_chunks = []

for idx in indices[0]:
    retrieved_chunks.append(chunks[idx])

context = "\n".join(retrieved_chunks)

print("\nRetrieved Context:")
print(context)

# -----------------------------------
# PROMPT
# -----------------------------------

prompt = f"""
Answer the question only using the context below.

Context:
{context}

Question:
{query}
"""

# -----------------------------------
# OLLAMA CLIENT
# -----------------------------------

client = ollama.Client(
    host="http://127.0.0.1:11434"
)

# -----------------------------------
# GENERATE RESPONSE
# -----------------------------------

response = client.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# -----------------------------------
# FINAL OUTPUT
# -----------------------------------

print("\nAnswer:")
print(response["message"]["content"])