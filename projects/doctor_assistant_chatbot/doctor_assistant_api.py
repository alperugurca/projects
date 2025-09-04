"""
Fast api GPT Doctor Assistant
Each user has a conversation history
"""

import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2. Define the FastAPI app

app = FastAPI(title="GPT Doctor Assistant", description="A chatbot that can answer questions about health and medicine")

# 3. Define the model
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7,
    openai_api_key=api_key
)

# 4. Define the conversation history
user_memories: Dict[str, ConversationBufferMemory] = {}

# 5. Define the conversation chain
class ChatRequest(BaseModel):
    name: str
    age: int
    message: str

class ChatResponse(BaseModel):
    response: str

# 6. Define the API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_doctor(request: ChatRequest):
    try:
        if request.name not in user_memories:
            user_memories[request.name] = ConversationBufferMemory(return_messages=True)
        memory = user_memories[request.name]

        if len(memory.chat_memory.messages) == 0:
            intro = (
                f"You are a doctor's assistant. You are talking to {request.name}, who is {request.age} years old. "
                f"You are helping them with their health and medicine questions. "
                f"Please provide careful, gentle, and age-appropriate advice to help them with their health and medical questions."
            )
            memory.chat_memory.add_user_message(intro)

        conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
        reply = conversation.predict(input=request.message)

        print(f"\n Memory: ")
        for idx, m in enumerate(memory.chat_memory.messages, start=1):
            print(f"{idx:02d}. {m.type.upper()}: {m.content}")
        print("--------------------------------")

        return ChatResponse(response=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# swagger ui: http://127.0.0.1:8000/docs