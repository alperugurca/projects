"""
# chatbot ui with streamlit
"""

import streamlit as st
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory


st.set_page_config(page_title="Tourist Bot", page_icon=":earth_asia:", layout="wide")
st.title("Tourist Bot")
st.markdown("You can ask questions to get information about tourist attractions all over Turkey.")

#session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

#ollama
llm = ChatOllama(model="llama3.2:3b")

# message box
user_input = st.text_input("Ask me anything about Turkey")

if user_input:
    st.session_state.memory.chat_memory.add_user_message(user_input)

    messages = [
        SystemMessage(content="You are a helpful assistant for users asking about food, cities, transportation, vacations, suggestions, and advice related to Turkey." "You are a helpful assistant who provides users with beautiful and informative answers about cities, historical places, and transportation suggestions in Turkey.")
    ] + st.session_state.memory.load_memory_variables({})["history"] + [HumanMessage(content=user_input)]

    response = llm(messages)


    st.session_state.memory.chat_memory.add_ai_message(response.content)

for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)
