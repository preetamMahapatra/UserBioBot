# vector_db_creator.py

import os
import shutil
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

### Set variables from config or directly ###
DATA_PATH = "data_sources"
CHROMA_PATH = "chroma"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2.gguf2.f16.gguf"

# Moved embedding model initialization here so this script is self-contained
try:
    from langchain_community.embeddings import GPT4AllEmbeddings
    gpt4all_embeddings = GPT4AllEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        gpt4all_kwargs={'allow_download': 'True'}
    )
    logger.info("GPT4AllEmbeddings initialized for vector DB creation.")
except Exception as e:
    logger.error(f"Failed to load GPT4AllEmbeddings for vector DB creation: {e}", exc_info=True)
    exit("Exiting: Embedding model failed to load.")


def create_vector_db():
    "Create vector DB from personal PDF files."
    logger.info("Starting vector database creation process...")
    documents = load_documents()
    if not documents:
        logger.warning(f"No documents found in {DATA_PATH}. Vector DB will be empty.")
        # Ensure Chroma path exists even if no documents, to prevent issues later
        if not os.path.exists(CHROMA_PATH):
            os.makedirs(CHROMA_PATH)
        return

    doc_chunks = split_text(documents)
    save_to_chroma(doc_chunks)
    logger.info("Vector database creation process completed.")

def load_documents():
    "Load PDF documents from a folder."
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data source directory '{DATA_PATH}' does not exist.")
        return []

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", show_progress=True)
    try:
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {DATA_PATH}.")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents from {DATA_PATH}: {e}", exc_info=True)
        return []

def split_text(documents: list[Document]):
    "Split documents into chunks."
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    "Clear previous db, and save the new db."
    if os.path.exists(CHROMA_PATH):
        logger.info(f"Clearing existing Chroma DB at {CHROMA_PATH}.")
        shutil.rmtree(CHROMA_PATH)

    # Create db
    # Ensure CHROMA_PATH directory exists before creating the DB
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, gpt4all_embeddings, persist_directory=CHROMA_PATH
    )
    
    logger.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    
if __name__ == "__main__":    
    create_vector_db()