"""

Problem:
Users will be able to ask questions in writing and receive real-time, natural responses.
    - Turkey
    - Places, food, history, culture, etc.

Model Introduction: LLAMA
    - LLAMA 3.2 3B
    - Open Source
    - Efficient, low-parameter, strong performance
    - This script allows users to interact with a local LLAMA model (1B, 3B, 8B, etc.) in English.
    - Can be run locally

plan:

install lib

import lib

download ollama and download model llama 3.2 3b
    - https://ollama.com/library/llama3.2:3b


"""


# import lib

from langchain.chat_models import ChatOllama # ollama llm
from langchain.schema import SystemMessage, HumanMessage # Chat messages
from langchain.memory import ConversationBufferMemory # Conversation history


# define model llama

llm = ChatOllama(model="llama3.2:3b")


# add memory, conversation history track

memory = ConversationBufferMemory(return_messages=True) # return_messages=True -> messages return best

# welcome message

print("Welcome to the Terminal Tourist Bot!")
print("Ask me anything about Turkey (places, food, history, culture, etc.). Type 'exit' to quit.")


# on terminal talk to llama

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # we will adduser input to memory?
    memory.chat_memory.add_user_message(user_input)

    # messages
    messages = [
        SystemMessage(content="You are a helpful assistant for users asking about food, cities, transportation, vacations, suggestions, and advice related to Turkey.""You are a idiot bot, you will answer in a funny way."),
        *memory.load_memory_variables({})["history"],
        HumanMessage(content=user_input)
    ]

    response = llm(messages)

    memory.chat_memory.add_ai_message(response.content)

    print(f"Guide: {response.content}")