from tools.chunk_tool import chunk_text
from tools.embed_tool import create_vector_store
from tools.retreive_tool import retrieve_context
from tools.summarize_tool import summarize_text
from tools.answer_tool import generate_answer

class RagAgent:

    def process_document(self, text):
        chunks = chunk_text(text)
        create_vector_store(chunks)
        print("Document processed successfully.")

    def ask(self, query):
        print("\n[Agent] Retrieving context...")
        context = retrieve_context(query)
        print("\n[Retrieved Context]")
        print(context)
        # Agent decision
        if len(context) > 1200:
            print("\n[Agent] Context too large.")
            print("[Agent] Summarizing context...")
            context = summarize_text(context)
            print("\n[Summarized Context]")
            print(context)

        print("\n[Agent] Generating final answer...")
        answer = generate_answer(query, context)

        return answer
