"""
Automated ingestion script for CV analysis knowledge base
"""

import os
import json
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify OpenAI API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
        "in your .env file or export it in your shell."
    )

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseIngestion:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.vectorstore = None
        
    def ingest_documents(self, documents_path: str):
        """
        Ingest documents into both vector store and BM25 index
        """
        try:
            # Load documents
            documents = self._load_documents(documents_path)
            logger.info(f"Loaded {len(documents)} documents")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info("Created vector store")
            
            # Create BM25 index
            self._create_bm25_index(documents)
            logger.info("Created BM25 index")
            
            # Save metadata
            self._save_metadata(len(documents))
            logger.info("Saved ingestion metadata")
            
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}")
            raise
            
    def _load_documents(self, documents_path: str) -> List[Document]:
        """
        Load documents from the specified path and convert to LangChain Documents
        """
        raw_documents = []
        path = Path(documents_path)
        
        # Load raw documents
        if path.is_file():
            with open(path, 'r') as f:
                raw_documents = json.load(f)
        elif path.is_dir():
            for file in path.glob('*.json'):
                with open(file, 'r') as f:
                    raw_documents.extend(json.load(f))
        
        # Convert to LangChain Documents
        documents = []
        for doc in raw_documents:
            documents.append(
                Document(
                    page_content=doc['text'],
                    metadata={'type': doc.get('type', 'unknown')}
                )
            )
                    
        return documents
        
    def _create_bm25_index(self, documents: List[Document]):
        """
        Create and save BM25 index
        """
        # Tokenize documents
        tokenized_docs = [doc.page_content.split() for doc in documents]
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_docs)
        
        # Save index
        index_path = Path(self.persist_directory) / 'bm25_index.json'
        with open(index_path, 'w') as f:
            json.dump({
                'corpus': [doc.page_content for doc in documents],
                'doc_freqs': bm25.doc_freqs,
                'doc_len': bm25.doc_len,
                'avgdl': bm25.avgdl
            }, f)
            
    def _save_metadata(self, num_documents: int):
        """
        Save ingestion metadata
        """
        metadata = {
            'num_documents': num_documents,
            'timestamp': str(datetime.now()),
            'embeddings_model': self.embeddings.__class__.__name__
        }
        
        metadata_path = Path(self.persist_directory) / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Ingest documents into knowledge base')
    parser.add_argument('--documents', required=True, help='Path to documents directory or file')
    parser.add_argument('--persist-dir', default='data/chroma', help='Directory to persist vector store')
    args = parser.parse_args()
    
    # Create ingestion instance
    ingestion = KnowledgeBaseIngestion(args.persist_dir)
    
    # Run ingestion
    ingestion.ingest_documents(args.documents)
    
if __name__ == "__main__":
    main()