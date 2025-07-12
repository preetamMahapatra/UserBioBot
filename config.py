# config.py

import os
import logging
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import GoogleGenerativeAI

# Load environment variables at the very beginning
load_dotenv()

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHROMA_PATH = "chroma"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2.gguf2.f16.gguf"
GEMINI_MODEL_NAME = "gemini-1.5-flash"
DB_CONNECTION_STRING = os.getenv("DATABASE_URL", "sqlite:///langchain_chat_history.db")

# --- Initializations (without Streamlit decorators here) ---

def get_embedding_model_instance():
    """Initializes and returns the GPT4All Embeddings model."""
    try:
        embedding_model = GPT4AllEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            gpt4all_kwargs={'allow_download': 'True'}
        )
        logger.info("GPT4AllEmbeddings initialized.")
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to load GPT4AllEmbeddings: {e}", exc_info=True)
        raise SystemExit("Embedding model initialization failed. Exiting.")

def get_vector_db_and_retriever_instance(_embeddings_instance):
    """Initializes and returns the Chroma DB and retriever."""
    try:
        if not os.path.exists(CHROMA_PATH):
            logger.error(f"Chroma path '{CHROMA_PATH}' not found. Please run vector_db_creator.py first.")
            # This is a critical dependency, so we exit if not found
            raise SystemExit(f"Vector database not found at '{CHROMA_PATH}'. Please run `python vector_db_creator.py` first.")

        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=_embeddings_instance)
        retriever = db.as_retriever(search_type="similarity")
        logger.info("Chroma DB and retriever initialized.")
        return db, retriever
    except Exception as e:
        logger.error(f"Failed to initialize Chroma DB: {e}", exc_info=True)
        raise SystemExit("Chroma DB initialization failed. Exiting.")

def get_llm_instance():
    """Initializes and returns the Google Generative AI LLM."""
    try:
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please add it to your .env file.")
        
        llm_model = GoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=gemini_api_key)
        # Simple invoke to check connection
        llm_model.invoke("Hello, check connection.")
        logger.info(f"LLM '{GEMINI_MODEL_NAME}' initialized successfully.")
        return llm_model
    except Exception as e:
        logger.error(f"Failed to initialize GoogleGenerativeAI or connect: {e}", exc_info=True)
        raise SystemExit("LLM initialization or connection failed. Exiting.")

# Initialize global resources directly. Caching for LangGraph app happens in app.py
GPT4ALL_EMBEDDINGS = get_embedding_model_instance()
CHROMA_DB, RETRIEVER = get_vector_db_and_retriever_instance(GPT4ALL_EMBEDDINGS)
LLM = get_llm_instance()