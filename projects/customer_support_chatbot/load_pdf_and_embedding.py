from langchain.embeddings import OpenAIEmbeddings # langchain openai embedding
from langchain.vectorstores import FAISS # langchain faiss vector database
from langchain.document_loaders import PyPDFLoader # langchain pdf loader
from langchain.text_splitter import RecursiveCharacterTextSplitter # langchain text splitter (chunking)

from dotenv import load_dotenv # load environment variables
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

os.environ["OPENAI_API_KEY"] = api_key # set environment variable

# load faq
loader = PyPDFLoader("faq.pdf")

# langchain documents
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # chunk size max 500 characters
    chunk_overlap=200 # overlap 200 characters behind and 200 characters ahead
    )

# chunks
docs = text_splitter.split_documents(documents)

# openai embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # Turkish support is good.

# The vector database is split into chunks. The text is converted into vectors using embeddings and an index is created.
vectordb = FAISS.from_documents(docs, embeddings)

# save vector database local disk
vectordb.save_local("faq_vectorstore")

print("Embeddings created and saved to local disk")

