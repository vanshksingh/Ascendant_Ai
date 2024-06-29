import json
import os
from datetime import datetime

from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the file path for saving chat history
CHAT_HISTORY_FILE = "chat_history.json"

llm = Ollama(model="mistral:instruct")

chat_history = []

# Load chat history from file if exists
if os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "r") as f:
        try:
            chat_history_data = json.load(f)
            for item in chat_history_data:
                if item['type'] == 'human':
                    chat_history.append(HumanMessage(content=item['content'], timestamp=datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S")))
                elif item['type'] == 'ai':
                    chat_history.append(AIMessage(content=item['content'], timestamp=datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S")))
        except json.JSONDecodeError:
            pass

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI named Mike, you answer questions with simple answers and no funny stuff.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt_template | llm

def save_chat_history():
    chat_history_data = []
    for message in chat_history:
        if isinstance(message, HumanMessage) or isinstance(message, AIMessage):
            chat_history_data.append({"type": "human" if isinstance(message, HumanMessage) else "ai",
                                      "content": message.content,
                                      "timestamp": message.timestamp.strftime("%Y-%m-%d %H:%M:%S")})

    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(chat_history_data, f, default=str)  # Use default=str to serialize datetime if needed

def clear_chat_history():
    global chat_history
    chat_history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)

def start_app():
    while True:
        question = input("You: ")
        if question == "/clear":
            clear_chat_history()
            print("Chat history cleared.")
            continue
        elif question == "done":
            save_chat_history()
            return

        response = chain.invoke({"input": question, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=question, timestamp=datetime.now()))
        chat_history.append(AIMessage(content=response, timestamp=datetime.now()))

        print("AI:", response)

if __name__ == "__main__":
    start_app()
