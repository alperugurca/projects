"""
Problem: Smart Customer Support Chatbot
    - A chatbot that can answer questions and help with tasks
        - FAQ questions answers
        - I lost my password
        - I want to change my password
        - Where is my receipt
        - How do I return an item
Solution:
    - RAG (Retrieval-Augmented Generation)
        - 1. Retrieve relevant documents(FAQ) from a vector database
        - 2. Users ask questions and the chatbot answers based on the retrieved documents
        
Technologies:
    - Langchain: rag 
    - faiss: vector database(embedding)
    - openai: llm
    - streamlit: ui, interaction

Dataset:
    - FAQ questions answers
        -Example
        Question: Do you sell internationally?
        Answer: No.

Plan:
    - 1. Client FAQ questions answers
    - 2. User upload documents from ui
    - 3. Document is converted to embeddings and stored in the vector database
    - 4. User ask questions and the chatbot answers based on the retrieved documents
    - 5. Chat history is stored in the memory

install lib : freeze

"""

from langchain.chat_models import ChatOpenAI # langchain openai chat model
from langchain.chains import ConversationalRetrievalChain # rag + chat chain
from langchain.vectorstores import FAISS # langchain faiss vector database
from langchain.embeddings import OpenAIEmbeddings # langchain openai embedding
from langchain.memory import ConversationBufferMemory # langchain memory

from dotenv import load_dotenv # load environment variables
import os

load_dotenv() # load environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

os.environ["OPENAI_API_KEY"] = api_key # set environment variable

# start embedding model text to vector
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# load vector database
vectordb = FAISS.load_local(
    "faq_vectorstore", # vector database path
    embeddings,
    allow_dangerous_deserialization=True # allow dangerous deserialization
    )


# create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
    )

# create llm
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0 # deterministic output
    )

# rag + memory chain
# llm
# faiss k3
# memory

qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = vectordb.as_retriever(search_kwargs = {"k": 3}),
    memory = memory,
    verbose = True
    )

print("Welcome the customer support chatbot")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Thank you for using the customer support chatbot")
        break

    response = qa_chain.run(user_input)
    print("Customer Support Chatbot: ", response)