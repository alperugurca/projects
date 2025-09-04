"""
What is the problem? I want to talk with documents.
    - Extract content from the contract file uploaded by the user
    - Represent the extracted content as vectors using embeddings
    - Use faiss fast search
    - Take the user's question and find the most relevant documents and answer with llm

Used tech:

- Embeddings ( Make vectors from text)
- FAISS ( Vector Store for fast search)
- LLM: cheapest gpt

- RAG: Retrieval Augmented Generation: Serving info for LLM
    - 1. Take the user's question
    - 2. Find the most relevant documents
    - 3. Answer the question with the most relevant documents (LLM)

    - users question -> Embeddings -> FAISS -> Chunking -> LLM

    - retrieval: find the most relevant documents
    - augmentation: add the most relevant documents to the user's question
    - generation: answer the question with the most relevant documents

Plan:

1. Contract file upload
2. Text extraction and chunking
3. Make vector store
4. Question and answer system


install libraries: freeze

import libraries

"""

# import libraries
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI


# load environment variables
load_dotenv() # load .env file for api key
api_key = os.getenv("OPENAI_API_KEY") # get api key

client = OpenAI(api_key=api_key) # openai 

model = SentenceTransformer("all-MiniLM-L6-v2") # model for embeddings

# load faiss index
index = faiss.read_index("data\contract_index.faiss")

# load chunks
with open("data\contract_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# take user input
while True:
    question = input("Enter your question: ") 

    # if user wants to exit
    if question.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    #user question -> vector
    question_embedding = model.encode([question])

    # faiss search 3 most relevant chunks
    k = 3 # number of chunks to return
    distances, indices = index.search(question_embedding, k)

    # get context

    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n ---\n".join(retrieved_chunks)

    # system prompt for llm
    prompt = f"""
                    Your are a contract lawyer AI assistant. Based on the contract context below, answer the user's question clearly and concisely.

                    Context: {context}

                    Question: {question}

                    Answer:
                    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25
        )
    
    print("AI Assistant: \n", response.choices[0].message.content.strip())
