
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter

from langchain_core.messages import HumanMessage, AIMessage




st.toast("Using Mistral")
model = Ollama(model='qwen2.5:1.5b-instruct')

chat_history = [] # Store the chat history



@tool
def converse(input: str) -> str:
    """Provide a natural language response using the user input."""
    bar.progress(40)
    return model.invoke(input)

#tools = [repl, converse ,recognize_speech_from_microphone , ]
tools = [
    converse
]


# Configure the system prompts
rendered_tools = render_text_description(tools)

system_prompt = f"""You answer questions with simple answers and no funny stuff , You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys. The value associated with the 'arguments' key should be a dictionary of parameters."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
     MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ]
)

# Define a function which returns the chosen tools as a runnable, based on user input.
def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

# The main chain: an LLM with tools.
chain = prompt | model | JsonOutputParser() | tool_chain





# Set up message history.
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("From calculations to image generation, data analysis to task prioritization, I'm here to assist. Always on, always learning. How can I help you today?")

# Set the page title.
st.title("Ascendant Ai")

# Render the chat history.
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# React to user input
if input := st.chat_input("What is up?"):

    if input == "/clear":
        #print("Chat history cleared.")
        st.chat_message("assistant").write("Chat history cleared.")
        st.toast("Data Cleared")

    else:
        # Display user input and save to message history.
        st.chat_message("user").write(input)
        msgs.add_user_message(input)

        # Invoke chain to get response.
        bar = st.progress(0)
        response = chain.invoke({"input": input, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=input))
        chat_history.append(AIMessage(content=response))
        bar.progress(90)

        # Display AI assistant response and save to message history.
        st.chat_message("assistant").write(str(response))
        msgs.add_ai_message(response)

        bar.progress(100)

        # Ensure the model retains context
        #msgs.add_ai_message(model.invoke(input))
