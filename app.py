# app.py

import streamlit as st
import asyncio
import logging
import uuid
import os # Required for os.makedirs if DB path doesn't exist (handled in rag_core now too)

# Import the query executor function and the workflow definition from rag_core
# We import 'workflow' here so it can be compiled within the Streamlit cache context.
from rag_core import execute_user_query_rag, workflow

# Configure logging for the Streamlit app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Preetam's BioBot – Your Personal AI Assistant", page_icon="🤖")
st.title("About Preetam 🙋‍♂️")

# --- Streamlit-specific caching for the LangGraph App (without checkpointer) ---
@st.cache_resource(ttl=3600) # Cache for 1 hour, or until app restart
def get_cached_langgraph_app_no_checkpointer():
    """
    Compiles and returns the LangGraph workflow without a checkpointer.
    The checkpointer will be provided during each invocation by execute_user_query_rag.
    """
    logger.info("Compiling LangGraph workflow without a checkpointer for caching.")
    try:
        compiled_app = workflow.compile() # Compile WITHOUT checkpointer here
        logger.info("LangGraph workflow compiled (no checkpointer).")
        return compiled_app
    except Exception as e:
        logger.error(f"Failed to compile LangGraph workflow in app.py (no checkpointer): {e}", exc_info=True)
        st.error("Application startup failed: Could not compile conversational engine.")
        st.stop()

# Get the cached compiled app (without checkpointer)
LANGGRAPH_APP_COMPILED_NO_CHECKPOINT = get_cached_langgraph_app_no_checkpointer()


# --- Session State Initialization and Management ---

# Initialize chat history in Streamlit's session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize a unique thread ID for the current user session
# This ensures each user interacting with the Streamlit app has their own LangGraph chat history.
if "session_thread_id" not in st.session_state:
    st.session_state.session_thread_id = str(uuid.uuid4())
    logger.info(f"New session started with thread_id: {st.session_state.session_thread_id}")

# Display a welcome message if the chat is empty
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("Hello, you can ask me anything about Preetam 👋")
        # st.session_state.messages.append({"role": "assistant", "content": ""})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Response Generation ---

if query_text := st.chat_input("Ask away!"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query_text)
        
    # Add user message to chat history for immediate display
    st.session_state.messages.append({"role": "user", "content": query_text})

    # Execute the RAG query asynchronously
    with st.spinner("Thinking..."): # Add a spinner for better UX
        try:
            # Pass the pre-compiled app (without checkpointer) to the executor.
            # The executor function will now handle the AsyncSqliteSaver creation
            # and connection within its own async context.
            response = asyncio.run(execute_user_query_rag(LANGGRAPH_APP_COMPILED_NO_CHECKPOINT, query_text, st.session_state.session_thread_id))
        except Exception as e:
            logger.error(f"Error executing RAG query in Streamlit app: {e}", exc_info=True)
            response = "I apologize, but I encountered an unexpected error. Please try asking again."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})