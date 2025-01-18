from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below.

Here is the conversation history: {context}

Question : {question}

Answer:

"""




model = OllamaLLM(model = "gemma2:2b")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

result = chain.invoke({"context": "context","question":"user_input"})
#print(result)

def handle_conversation():
    contect = ""
    print("Welcome")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        result = model.invoke(input= user_input)
        print("Bot: " ,result)
      #  context += f"\nUser: {user_input}\n AI: {result}"  


if __name__ == "__main__":
    handle_conversation()    