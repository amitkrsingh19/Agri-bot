from functools import lru_cache
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .chain import build_refine_chain
from .ingest import load_and_split_document, get_embedding, create_vector_store
from .logger import logger


@lru_cache(maxsize=1)
def get_retrieval_chain() -> Any:
    """Build or load the retrieval chain used for RAG responses."""
    logger.info("Setting up RAG system (retrieval chain)...")

    # Load or create document chunks
    chunks = load_and_split_document()
    embedding = get_embedding()

    if not chunks:
        logger.warning("No document chunks found. Attempting to load persisted vector store.")
        try:
            vector_store = create_vector_store([], embedding)
            retriever = vector_store.as_retriever()
            logger.info("Successfully loaded persisted vector store from disk.")
            return build_refine_chain(retriever)
        except Exception as e:
            logger.error(f"Failed to load persisted vector store: {e}")
            class _FallbackChain:
                def invoke(self, payload: dict):
                    return {"answer": "Knowledge base is empty. Please add documents first."}

            return _FallbackChain()

    vector_store = create_vector_store(chunks, embedding)
    retriever = vector_store.as_retriever()
    logger.info("Retriever instance created from scraped chunks.")
    return build_refine_chain(retriever)


def reload_retrieval_chain() -> Any:
    global _graph_instance
    get_retrieval_chain.cache_clear()
    _graph_instance = None 
    logger.info("Cleared retrieval chain and graph cache after new ingestion.")
    return get_retrieval_chain()


def build_message_history(
    user_message: str, history: Optional[List[Dict[str, str]]] = None
) -> List[BaseMessage]:
    """Convert a simple history payload into LangChain message objects."""
    messages: List[BaseMessage] = []

    if history:
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if not role or not content or not isinstance(content, str):
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))
    return messages




