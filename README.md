# BioBot

# Preetam's BioBot: A Conversational AI with Persistent Memory

**Preetam's BioBot** is an interactive conversational AI built with Streamlit, LangGraph, and LangChain. This application allows users to ask questions about a specific knowledge base (e.g., a person's biography or professional profile) and maintains conversational memory across interactions, making the experience seamless and context-aware.

-----

## Features

  * **Conversational AI:** Engage in natural language conversations.
  * **Retrieval-Augmented Generation (RAG):** Answers questions by retrieving information from a dedicated knowledge base, ensuring factual accuracy and reducing hallucinations.
  * **Persistent Chat History:** Remembers past interactions within a session, allowing for follow-up questions and maintaining context, even if the Streamlit app reruns.
  * **Streamlit UI:** A simple and intuitive web interface for easy interaction.
  * **Modular Architecture:** Clear separation of concerns with `config.py`, `rag_core.py`, and `app.py`.
  * **Local Data Storage:** Uses SQLite for chat history persistence and ChromaDB for the vector store, making it easy to run locally.

-----

## Core Technologies

  * **Streamlit:** For building the interactive web application UI.
  * **LangGraph:** Orchestrates the complex conversational flow and state management.
  * **LangChain:** Powers the RAG capabilities, including LLM integration, prompt engineering, and retriever setup.
  * **ChromaDB:** A lightweight vector database used to store and retrieve contextual documents.
  * **GPT4AllEmbeddings:** Used for converting text into numerical embeddings for vector search.
  * **Google Gemini Flash:** The Large Language Model (LLM) for generating responses.
  * **`AsyncSqliteSaver` (from LangGraph):** Manages asynchronous persistence of chat history in a SQLite database.
  * **`aiosqlite`:** Asynchronous SQLite database driver.
  * **`pyenv`:** Recommended for managing Python versions and virtual environments.

-----

## Project Structure

```
.
├── .env                  # Environment variables (API keys, DB paths)
├── app.py                # Main Streamlit application
├── config.py             # Global configurations and component initializations
├── rag_core.py           # LangChain RAG chains and LangGraph workflow definition
└── vector_db_creator.py  # Script to create and populate the ChromaDB vector store
```

-----

## Setup and Installation

Follow these steps to get your Preetam BioBot running locally.

### 1\. Create Your `.env` file

In the root of your project directory, create a file named `.env` and add your Google Gemini API key and the desired path for your SQLite database. Replace `YOUR_GEMINI_API_KEY_HERE` with your actual API key.

```dotenv
# .env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
DATABASE_URL=sqlite:///langchain_chat_history.db # Path for your SQLite DB
```

### 2\. Prepare Your Python Environment with `pyenv`

Using `pyenv` is highly recommended for managing Python versions and creating isolated virtual environments. This helps prevent dependency conflicts.

  * **Install `pyenv` (if you haven't already):**
    Follow the official `pyenv` installation guide for your operating system. For macOS/Linux, a common method is:

    ```bash
    curl https://pyenv.run | bash
    # Follow the instructions to add pyenv to your shell's configuration (e.g., ~/.bashrc, ~/.zshrc)
    # Then, restart your shell: exec "$SHELL"
    ```

  * **Install a specific Python version (e.g., Python 3.11.9):**

    ```bash
    pyenv install 3.11.9
    ```

  * **Create a virtual environment for your project:**
    Navigate to your project's root directory and create a virtual environment using the installed Python version. This will create a `.venv` directory inside your project.

    ```bash
    cd /path/to/your/user_rag_genai_poc # Navigate to your project folder
    pyenv virtualenv 3.11.9 preetam-biobot-env
    ```

    *Replace `preetamm-biobot-env` with a name you prefer for your environment.*

  * **Set the local Python version for your project:**
    This command creates a `.python-version` file in your project directory, ensuring `pyenv` automatically activates this virtual environment whenever you are in this directory.

    ```bash
    pyenv local preetam-biobot-env
    ```

  * **Activate the virtual environment (if not automatically activated):**
    You should see `(preetamm-biobot-env)` or similar in your terminal prompt. If not, you might need to manually activate it:

    ```bash
    pyenv activate preetam-biobot-env
    # Or, if you're already in the project directory where you ran `pyenv local`:
    # exec "$SHELL" # To re-read your shell config and activate the environment
    ```

### 3\. Install Dependencies

With your `pyenv` virtual environment activated, install all the required Python packages:

```bash
pip install streamlit langchain_chroma langchain_community langchain-google-genai langgraph python-dotenv tenacity aiosqlite
```

### 4\. Prepare Your Knowledge Base

You need to create a text file that contains the information for your RAG system. For this "Preetam's BioBot" example, create a file named `preetam_bio.txt` in the root of your project and fill it with biographical details, skills, experiences, etc., about "Preetam."

Example `preetam_bio.txt` content:

```
Preetam Mahapatra is a highly skilled software engineer with 10+ years of experience in developing scalable and robust applications. He specializes in cloud architecture, machine learning operations (MLOps), and full-stack development. Preetam is known for his proactive approach, problem-solving abilities, and strong technical leadership.

His key skills include:
- Programming Languages: Python, Java, JavaScript
- Cloud Platforms: AWS, Azure, Google Cloud Platform (GCP)
- MLOps Tools: MLflow, Kubeflow, Docker, Kubernetes
- Databases: PostgreSQL, MongoDB, SQLite
- Frameworks: Django, Flask, React, Angular
- AI/ML: Natural Language Processing (NLP), Computer Vision, Deep Learning frameworks (TensorFlow, PyTorch)

Preetam has led several successful projects, delivering high-performance solutions that exceed client expectations. He is passionate about leveraging cutting-edge technology to solve real-world problems.
```

### 5\. Run the Vector DB Creator (Once)

This script will load the content from `preetam_bio.txt`, chunk it, embed it using GPT4AllEmbeddings, and store it in a ChromaDB instance at the `chroma` directory.

```bash
python vector_db_creator.py
```

*You only need to run this command once, or whenever you update your `preetam_bio.txt` data.*

### 6\. Run the Streamlit Application

Finally, launch your Streamlit application:

```bash
streamlit run app.py
```

Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`). You can now start chatting with Preetam's BioBot\!

-----

## 💡 Troubleshooting and Lessons Learned

A common challenge when integrating `asyncio` libraries like `AsyncSqliteSaver` with Streamlit is the "RuntimeError: ... is bound to a different event loop." This error arises because:

  * **Streamlit's Nature:** Streamlit re-executes the entire script on every user interaction, and it typically runs in a synchronous fashion or on its own event loop that isn't directly exposed.
  * **`asyncio.run()`:** Each call to `asyncio.run()` creates a *new*, isolated event loop for the duration of its execution.
  * **`AsyncSqliteSaver`:** Internally, `AsyncSqliteSaver` (and other `asyncio` primitives) create objects that are inherently "bound" to the specific event loop active at their creation time.

**Our Solution:**

The core fix implemented in `rag_core.py` ensures that the `AsyncSqliteSaver` and its underlying `aiosqlite` database connection are always created *within the same asynchronous context* (i.e., the same event loop) that will be used to execute the LangGraph workflow's `ainvoke` call for a given user query.

  * The main `app.py` caches the **compiled LangGraph workflow structure** (which is stateless).
  * For each user query, `rag_core.py`'s `execute_user_query_rag` function dynamically creates a **new `AsyncSqliteSaver` instance** and its `aiosqlite` connection. This ensures the checkpointer and its internal `asyncio` locks are always bound to the *current* event loop created by `asyncio.run()` for that specific query.
  * The newly created checkpointer is then passed directly into the `ainvoke()` method's `config` parameter (`config={"configurable": {"thread_id": thread_id, "checkpoint": checkpointer_for_this_run}}`).
  * Crucially, the `aiosqlite` connection is explicitly closed after each invocation, preventing resource leaks.

This pattern circumvents the event loop binding issue, providing a stable and persistent chat experience within Streamlit.

-----

## 🌟 Future Enhancements

  * **Multi-user Support:** Implement user authentication and separate chat histories per logged-in user.
  * **More Data Sources:** Extend `vector_db_creator.py` to ingest data from various file types (PDFs, webpages, etc.).
  * **Tool Usage:** Integrate LangChain tools into the LangGraph workflow to give the AI agent external capabilities (e.g., searching the web for current events, performing calculations).
  * **Advanced UI:** Utilize more advanced Streamlit components for a richer user experience (e.g., clear chat history button, different display options for retrieved context).
  * **Deployment:** Containerize the application using Docker for easier deployment to cloud platforms.

-----