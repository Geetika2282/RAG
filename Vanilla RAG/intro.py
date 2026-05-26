from langchain_text_splitters import RecursiveCharacterTextSplitter # For Chunking
from langchain_ollama import OllamaEmbeddings # For Embedding model
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import ollama

with open('intro.txt', 'r') as file:
    content = file.read()
    print(content)

# -----------------------------------
# CHUNKING
# -----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, # Each chunk will have a maximum of 200 characters.
    chunk_overlap=20 # Before moving to Chunk 2, the splitter grabs the last 20 characters from the end of Chunk 1
)

chunks = splitter.create_documents([content])
# Below code is for debugging, to see the content of the first two chunks.

# print(chunks[0].page_content)
# print("-----------------------------")
# print(chunks[1].page_content)


# -----------------------------------
# EMBEDDING MODEL (Ollama local)
# -----------------------------------
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe:latest",
    base_url="http://127.0.0.1:11434") # This is the default URL for Ollama running locally. If you have a different setup, adjust accordingly.[Wind + R -> "%LOCALAPPDATA%\Ollama" -> Server.log -> Ollama_host -> 127.0.0.1:11434]


# -----------------------------------
# VECTOR DATABASE
# -----------------------------------
vector_db = FAISS.from_documents(
    chunks, 
    embedding_model)

# -----------------------------------
# USER QUESTION
# -----------------------------------
query = input("Ask something about me: ")

# -----------------------------------
# RETRIEVE RELEVANT CHUNKS
# -----------------------------------
results = vector_db.similarity_search(query, k=2) # FAISS identifies the top 2 closest vectors (k=2)

context = "\n".join([doc.page_content for doc in results]) # extracts the content of the retrieved chunks and combines them into a single string, separated by newlines.

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
# LLAMA RESPONSE
# -----------------------------------
client = ollama.Client(
    host="http://127.0.0.1:11434"
)

response = client.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "user",
            "content": prompt       
        }
    ]
)

print("\nAnswer: ")
print(response["message"]["content"])
