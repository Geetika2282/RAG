from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2:7b",
    base_url="http://localhost:11434")

def generate_answer(query, context):

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY from the provided context.

Question:
{query}

Context:
{context}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content
