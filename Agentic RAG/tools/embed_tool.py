from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings # For Embedding model
from langchain_community.embeddings import SentenceTransformerEmbeddings
import ollama 

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe:latest",
    base_url="http://127.0.0.1:11434") # This is the default URL for Ollama running locally. If you have a different setup, adjust accordingly.[Wind + R -> "%LOCALAPPDATA%\Ollama" -> Server.log -> Ollama_host -> 127.0.0.1:11434]

def create_vector_store(chunks):
    vectorstore = FAISS.from_texts(
        chunks,
        embedding_model
    )

    vectorstore.save_local("vectorstore/db")

    return "Vector store created successfully."
