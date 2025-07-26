"""
problem statement: users questions about their health and the assistant provides answers. GPT chatbot
        - age and name
        - history of the user
        - Langchain ve OPENAI GPT
        - ilk terminalde ve sonra FastAPI ile yapılacak

dataset: none

model explanation: Generative Pre-trained Transformer GPT-3.5 Turbo
        - real time health support with GPT-3.5 Turbo API

Langchain:
        - prompt management
        - memory
        - tools
        - chain of thought


plan:
        
install libraries:
    - fastapi: for web api framework (asenkron)
    - uvicorn: for running the fastapi app host
    - langchain: for LLM and memory
    - openai: for LLM
    - python-dotenv: for loading environment variables from .env file
    - rich: for terminal styling
    - colorama: for cross-platform colored terminal output
    - termcolor: for terminal color output
    - requests: for making HTTP requests
    - json: for parsing JSON data


import libraries:


"""

# 1. import libraries
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

import warnings
warnings.filterwarnings("ignore")

# 2. set up the environment (OPENAI API KEY)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")



# 3. LLM + Memory
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7, # 0.0: deterministic, 1.0: random
    api_key=api_key) 

memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True)

# 4. Collect user input
name = input("Enter your name: ")
age = input("Enter your age: ")

intro = (
    f"You are a doctor's assistant. You are talking to {name}, who is {age} years old. "
    f"{name} wants to talk about health problems."
    f" Please address {name} by name, and provide careful and gentle advice suitable for their age. "
)

memory.chat_memory.add_user_message(intro)
print(f"Hello {name}! I'm your doctor's assistant. How can I help you today?")






# 5. Chatbot loop
while True:
    # user asked something
    user_msg = input(f"{name}: ")
    if user_msg.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
    # llm answer
    reply = conversation.predict(input=user_msg)
    print(f"Assistant: {reply}")

    # save the conversation to the memory
    print("\nMemory")
    for idx, m in enumerate(memory.chat_memory.messages, start=1):
        print(f"{idx:02d}. {m.type.upper()}: {m.content}")
    print("---------------------------\n")