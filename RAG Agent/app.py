from agent.main_agent import RagAgent

def load_document(path):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    
    return text

agent = RagAgent()

# 1. Load dcoument
document_text = load_document("data/demo.txt")    

# 2. Process document
agent.process_document(document_text)

# 3. Chat loop
while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        print("--Goodbye!--")
        break
    response = agent.ask_question(query)

    print("---------\nLLMAnswer:\n", response)

    
