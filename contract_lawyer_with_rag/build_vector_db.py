"""
Vector Database Builder
Faiss is library for efficient similarity search and clustering of dense vectors.
"""

# import libraries

import os
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer # Embeddings
import faiss # Vector Store
import numpy as np
import pickle # for saving and loading

# read pdf and extract text
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, max_length=500):
    """
    Split text into chunks of a maximum length
    """
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) < max_length:
            current += line + line.strip()
        else:
            chunks.append(current.strip())
            current = line.strip()
    if current:
        chunks.append(current.strip())
    return chunks


model = SentenceTransformer("all-MiniLM-L6-v2")
pdf_file_path = "data\contract_sample.pdf"

# Extract text from pdf
text = extract_text_from_pdf(pdf_file_path)

# Chunk text
chunks = chunk_text(text, max_length=500)

# Each chunk is embedded
embeddings = model.encode(chunks)


print(f"Embeddings shape: {embeddings.shape}")

# Create a FAISS index
dimension = embeddings.shape[1] # dimension of the embeddings


index = faiss.IndexFlatL2(dimension) # L2 distance(euclidean distance) similarity search

# Add embeddings to the index
index.add(np.array(embeddings)) # add embeddings to the index

# faiss index and chunks are saved
faiss.write_index(index, "data\contract_index.faiss")
with open("data\contract_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("faiss index and chunks are saved")







