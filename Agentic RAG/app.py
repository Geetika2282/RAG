from agent.main_agent import RagAgent

# Load text file
with open("data/demo.txt", "r", encoding="utf-8") as f:
    text = f.read()

agent = RagAgent()

# One time processing
agent.process_document(text)

while True:
    query = input("\nAsk Question: ")
    if query.lower() == "exit":
        break
    answer = agent.ask(query)
    print("\nFinal Answer:")
    print(answer)