import pickle
import faiss

# Load the FAISS index
index = faiss.read_index("faq_vectorstore/index.faiss")
print("\nFAISS Index Information:")
print(f"Total vectors in index: {index.ntotal}")
print(f"Dimension of vectors: {index.d}")

# Load the pickle file
with open("faq_vectorstore/index.pkl", "rb") as f:
    pkl_data = pickle.load(f)
print("\nPickle File Contents:")
print(f"Type of stored data: {type(pkl_data)}")
print("Contents:")
print(pkl_data)