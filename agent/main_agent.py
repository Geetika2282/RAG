from torch import chunk

from tools.chunk_tool import chunk_text
from tools.embed_tool import create_vector_store        
from tools.retreive_tool import retrieve_context
from tools.summarize_tool import summarize_text
from tools.answer_tool import generate_answer

class RagAgent:
    def process_document(self, text):
        # 1. Chunk the text
        chunks = chunk_text(text)

        # 2. Embeddings + FAISS index
        create_vector_store(chunks)
        print(chunks)
        print(f"Number of chunks: {len(chunks)}")
        print("Document processed Successfully.")

    def ask_question (self, query):
        # 1. Retreive relevant chunks
        retreived_chunks = retrieve_context(query)

        # 2. Retreived chunks
        # for chunk in retreived_chunks:
            # print("\nRETRIEVED CHUNK:")
            # print(chunk, '-----------')

        context = "\n".join(retreived_chunks)

        # 4. Generate answer
        final_answer = generate_answer(query, context)
        return final_answer
