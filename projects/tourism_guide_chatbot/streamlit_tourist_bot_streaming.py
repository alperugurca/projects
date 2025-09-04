import streamlit as st

from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory


# streaming call back

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # Write terminal
from langchain.callbacks import BaseCallbackHandler # Write to session state
from typing import Any

# streamlit special callback

class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.final_text = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.final_text += token
        self.placeholder.markdown(self.final_text + " ")


st.set_page_config(page_title="Tourist Bot (Live)", page_icon=":earth_asia:", layout="wide")
st.title("Tourist Bot(Streaming Mode)")
st.markdown("You can ask questions to get information about tourist attractions all over Turkey.")
 

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)


# message box
user_input = st.chat_input("Ask me anything about Turkey")

for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)


if user_input:
    st.session_state.memory.chat_memory.add_user_message(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty() # empty placeholder temporary
        stream_handler = StreamHandler(response_placeholder)

        llm = ChatOllama(model="llama3.2:3b", streaming=True, callbacks=[stream_handler])
        # messages
        messages = [
            SystemMessage(content="You are a helpful assistant for users asking about food, cities, transportation, vacations, suggestions, and advice related to Turkey." "You are a helpful assistant who provides users with beautiful and informative answers about cities, historical places, and transportation suggestions in Turkey.")
        ] + st.session_state.memory.load_memory_variables({})["history"] + [HumanMessage(content=user_input)]

        response = llm(messages)


        st.session_state.memory.chat_memory.add_ai_message(response.content)