import streamlit as st # web app

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os
import tempfile # temporary file

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# streamlit app
st.set_page_config(page_title="Customer Support Chatbot", page_icon=":paperclip:")
st.title("Customer Support Chatbot (RAG + MEMORY)")
st.write("Upload a PDF file and talk with it!")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")

if uploaded_file is not None:
    if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
        with st.spinner("Processing the PDF file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name # temporary file path

                # load the pdf file with PyPDFLoader
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.split_documents(documents)

                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

                # create a FAISS index
                vectordb = FAISS.from_documents(docs, embeddings)

                # memory and llm
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

                # rag + memory chain
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
                    memory=memory,
                )

                st.session_state.qa_chain = qa_chain
                st.session_state.chat_history = []
                st.session_state.last_uploaded_name = uploaded_file.name
            
            st.success("PDF file processed successfully!")


if "qa_chain" in st.session_state:
    user_question = st.text_input("Ask me anything about the PDF file:")
    if user_question:
        response = st.session_state.qa_chain.invoke({"question": user_question}) # send the user question to the qa_chain
        st.session_state.chat_history.append(("user", user_question)) # add the user question to the chat history
        st.session_state.chat_history.append(("assistant", response["answer"])) # add the assistant response to the chat history

    if st.session_state.chat_history:
        st.subheader("Chat History")
        for sender, msg in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {msg}")
