from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe:latest",
    base_url="http://127.0.0.1:11434") # This is the default URL for Ollama running locally. If you have a different setup, adjust accordingly.[Wind + R -> "%LOCALAPPDATA%\Ollama" -> Server.log -> Ollama_host -> 127.0.0.1:11434]

def retrieve_context(query, k=2):

    vectorstore = FAISS.load_local(
    "vectorstore/db",
    embedding_model,
    allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(query, k=k)
    results = []
    for doc in docs:
        results.append(doc.page_content)
    return results