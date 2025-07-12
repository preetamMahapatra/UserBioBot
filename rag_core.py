# rag_core.py

import logging
from typing import Sequence
from typing_extensions import Annotated, TypedDict
import os # Added for directory creation in execute_user_query_rag

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from tenacity import retry, wait_exponential, stop_after_attempt, Retrying

# Import necessary modules for async checkpointer here
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Import initialized components from config
from config import LLM, RETRIEVER, DB_CONNECTION_STRING

logger = logging.getLogger(__name__)

# --- LangChain RAG Chain Definitions ---

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3), reraise=True)
def get_history_aware_retriever():
    """
    Creates a history-aware retriever to reformulate user queries into standalone questions.
    Includes retry logic for robustness.
    """
    question_reformulation_prompt = """
    You are an intelligent assistant. Your role is to understand the user's intent from the current question and the chat history.
    Based on the provided chat history and the latest user query, which might implicitly refer to previous turns,
    reformulate the latest question into a clear, standalone question. This standalone question must be
    understandable without any reference to the chat history.

    Instructions:
    1. If the latest question is already fully explicit and doesn't rely on history, return it as is.
    2. If the question depends on previous turns, combine the necessary context from the chat history
       with the latest question to form a complete, unambiguous question.
    3. Do NOT answer the question. Your sole task is reformulation.
    4. Ensure the reformulated question is concise and directly asks for the information needed.

    Example:
    Chat History:
    User: What is the capital of France?
    AI: Paris.
    User: And its population?
    Reformulated Question: What is the population of Paris?
    """

    question_reformulation_template = ChatPromptTemplate.from_messages(
        [
            ("system", question_reformulation_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        LLM, RETRIEVER, question_reformulation_template
    )
    logger.debug("History-aware retriever created.")
    return history_aware_retriever

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3), reraise=True)
def get_rag_chain():
    """
    Creates a Retrieval-Augmented Generation (RAG) chain to answer user questions
    by leveraging a history-aware retriever and a question-answering chain.
    Includes retry logic for robustness.
    """
    answer_question_prompt = """
    You are a highly knowledgeable and concise AI assistant, specialized in providing information about Preetam.
    Your answers must be based *strictly* on the following retrieved context.

    Instructions for Answering:
    1. Use ONLY the provided "Retrieved Context" to formulate your answer.
    2. Do NOT introduce any outside or general knowledge.
    3. If the "Retrieved Context" does not contain sufficient information to answer the question,
       state clearly and politely: "I'm sorry, but based on the information I have, I cannot provide a complete answer to that question."
    4. Your answer should be between three and seven sentences in length, providing sufficient detail while remaining concise.
    5. Start your response directly with the answer, without conversational filler like "Based on the context..." or "The answer is...".

    Retrieved Context:
    ---
    {context}
    ---
    """

    answer_question_template = ChatPromptTemplate.from_messages(
        [
            ("system", answer_question_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    answer_question_chain = create_stuff_documents_chain(LLM, answer_question_template)

    history_aware_retriever = get_history_aware_retriever()

    rag_chain = create_retrieval_chain(history_aware_retriever, answer_question_chain)
    logger.debug("RAG chain created.")
    return rag_chain

# --- LangGraph Components ---

class State(TypedDict):
    """
    Represents the application state for a conversational workflow.
    """
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str # This will store the retrieved documents as a string for logging/debugging
    answer: str

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3), reraise=True)
async def call_model(state: State):
    """
    Asynchronously executes a Retrieval-Augmented Generation (RAG) chain and updates the application state.
    Includes retry logic.
    """
    logger.info(f"LangGraph node 'call_model' triggered with input: {state['input']}")
    rag_chain = get_rag_chain() # Retrieve the RAG chain

    try:
        response = await rag_chain.ainvoke({"input": state["input"], "chat_history": state["chat_history"]})
        logger.info(f"RAG chain response received. Answer length: {len(response.get('answer', ''))}")

        retrieved_context_docs = response.get("context", [])
        context_str = "\n\n".join([doc.page_content for doc in retrieved_context_docs]) if retrieved_context_docs else "No specific context retrieved."

        updated_chat_history = state["chat_history"] + [
            HumanMessage(content=state["input"]),
            AIMessage(content=response.get("answer", "I couldn't generate an answer.")),
        ]

        return {
            "chat_history": updated_chat_history,
            "context": context_str,
            "answer": response.get("answer", "I encountered an issue generating a response."),
        }
    except Exception as e:
        logger.error(f"Error during model invocation in LangGraph: {e}", exc_info=True)
        error_message = "I apologize, but I encountered an error while processing your request. Please try again later."
        updated_chat_history = state["chat_history"] + [
            HumanMessage(content=state["input"]),
            AIMessage(content=error_message),
        ]
        return {
            "chat_history": updated_chat_history,
            "context": "Error during retrieval or generation.",
            "answer": error_message,
        }

# Defines the StateGraph workflow (NOT compiled with checkpointer here)
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

async def execute_user_query_rag(compiled_app_instance, query_text: str, thread_id: str):
    """
    Asynchronously executes the question-answering (QA) workflow with the provided query.
    This function now handles the creation of AsyncSqliteSaver and its connection
    within its scope to ensure correct asyncio loop binding.

    Parameters
    ----------
    compiled_app_instance : langgraph.graph.graph.CompiledGraph
        The compiled LangGraph application instance (without a default checkpointer).
    query_text : str
        The input query or question to be processed by the QA system.
    thread_id : str
        A unique identifier for the conversation thread, used by LangGraph for history management.

    Returns
    -------
    str
        The generated answer to the input query.

    Raises
    ------
    Exception
        If the workflow execution encounters a critical error.
    """
    logger.info(f"Executing query for thread_id '{thread_id}': '{query_text}'")
    temp_conn = None # Initialize to None

    try:
        # Create the checkpointer for THIS invocation's event loop
        db_file_path = DB_CONNECTION_STRING.replace("sqlite:///", "")
        
        # Ensure directory exists for the SQLite DB file
        db_directory = os.path.dirname(db_file_path)
        if db_directory and not os.path.exists(db_directory):
            os.makedirs(db_directory)
            logger.info(f"Created directory for SQLite DB: {db_directory} in rag_core for checkpoint.")

        # Establish the aiosqlite connection asynchronously in this context
        temp_conn = await aiosqlite.connect(db_file_path)
        checkpointer_for_this_run = AsyncSqliteSaver(conn=temp_conn)
        logger.info("AsyncSqliteSaver created for current query invocation.")
        
        # Pass the checkpointer via the config to ainvoke
        config = {"configurable": {"thread_id": thread_id, "checkpoint": checkpointer_for_this_run}}
        
        result = await compiled_app_instance.ainvoke(
            {"input": query_text},
            config=config, # Pass the dynamically created checkpointer in config
        )
        
        logger.info(f"Query executed. Answer: {result.get('answer', '')[:100]}...")
        return result.get("answer", "No answer was generated.")
    except Exception as e:
        logger.error(f"Failed to execute query for thread_id '{thread_id}': {e}", exc_info=True)
        return "I'm sorry, but I couldn't process your request due to an internal error."
    finally:
        # IMPORTANT: Close the connection if it was opened in this scope
        if temp_conn:
            await temp_conn.close()
            logger.info("AsyncSqliteSaver connection closed for current query invocation.")