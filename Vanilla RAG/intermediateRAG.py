# pip install llama-index llama-index-embeddings-huggingface llama-index-postprocessor-cohere qdrant-client
# pip install llama-index-postprocessor-sbert-rerank sentence-transformers




import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# CORRECT IMPORT: Points to the updated sbert_rerank module registry
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

def initialize_rag_system():
    print("[1/4] Checking environment configurations...")
    
    # Verify GROQ Key is present for the final generation step
    groq_key = os.environ.get("GROQ_API")
    if not groq_key:
        print("\n❌ Error: Missing GROQ_API in your environment variables.")
        print("Please set it in your terminal before running:")
        print('  $env:GROQ_API="your_key_here"\n')
        sys.exit(1)

    # Check if the text data directory exists and has files
    data_dir = "./data"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"\n❌ Error: Data directory '{data_dir}' is empty or missing.")
        print("Please create a folder named 'data' and add your text/PDF files inside it.\n")
        sys.exit(1)

    print("[2/4] Initializing local embedding engine (BAAI/bge-small-en-v1.5)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("[3/4] Parsing and indexing documents into local vector store...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

def main():
    index = initialize_rag_system()
    
    print("[4/4] Loading local Cross-Encoder Reranker (BAAI/bge-reranker-base)...")
    # This automatically instantiates a free cross-encoder on your CPU/GPU hardware
    local_reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base", 
        top_n=3  # Compresses the payload down to the top 3 best matching blocks
    )

    # Build the query engine using the corrected local rerank handler
    query_engine = index.as_query_engine(
        similarity_top_k=20, 
        node_postprocessors=[local_reranker]
    )

    print("\n🚀 Local Reranker RAG Pipeline Ready! Enter a query or type 'exit' to quit.\n")
    
    while True:
        user_query = input("Ask a question about your data: ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        if not user_query.strip():
            continue

        print("\nRetrieving candidates and locally reranking...")
        response = query_engine.query(user_query)

        print("\n=== AI RESPONSE ===")
        print(response)
        print("===================\n")

        print("--- Local Reranker Analytics (Top Sources Routed to LLM) ---")
        for i, node in enumerate(response.source_nodes, start=1):
            print(f"[{i}] Relevance Score: {node.score:.4f}")
            snippet = node.node.get_content().strip().replace('\n', ' ')
            print(f"    Snippet: {snippet[:120]}...\n")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
