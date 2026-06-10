from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from langchain_core.messages import AIMessage

from .ingest import ingest_pdf
from .rag_service import reload_retrieval_chain
from .logger import logger
from .ingest import ingest_pdf
from .rag_service import reload_retrieval_chain, build_message_history
from.graph import create_graph

class QueryRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None


class FinalResponse(BaseModel):
    answer: str
    sources: List[Dict]
    season_context: str

app = FastAPI(title="Kerela based farming AI Assistant (RAG) API")


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=FinalResponse)
async def ask(query: QueryRequest):
    result = ask_rag(query.message, query.history)
    return FinalResponse(
        answer=result["reply"],
        sources=result.get("sources", []),
        season_context=result.get("season_context", "")
    )


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a PDF to enrich the knowledge base."""
    try:
        response = ingest_pdf(file.file,file.filename)
        # rebuild the retrieval chain so new content is searchable immediately
        reload_retrieval_chain()
        return {"status": "ok", "message": response}
    except Exception as e:
        logger.error(f"Failed to process uploaded PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest PDF.")


@app.post("/reload")
async def reload() -> Dict[str, str]:
    """Force reloading the underlying RAG index and LangGraph."""
    try:
        reload_retrieval_chain()
        return {"status": "ok", "message": "Reloaded RAG index."}
    except Exception as e:
        logger.error(f"Failed to reload RAG index: {e}")
        raise HTTPException(status_code=500, detail="Reload failed.")


_graph_instance = None

def get_graph():
    global _graph_instance
    if _graph_instance is None:
        logger.info("Compiling and caching LangGraph.")
        _graph_instance = create_graph()
    return _graph_instance

app_graph = get_graph()

def ask_rag(user_message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Send a message through the LangGraph/RAG pipeline and return the response."""
    messages = build_message_history(user_message, history)

    logger.info("Invoking LangGraph with user input.")
    response = app_graph.invoke({"messages": messages})

    # Expect response to include "messages" (list of BaseMessage)
    output_messages = response.get("messages", [])
    if output_messages:
        last = output_messages[-1]
        return {
            "reply": last.content,
            "sources": response.get("sources",[]),
            "season_context" : response.get("season_context",""),
            "messages": [
                {"role": "assistant" if isinstance(m, AIMessage) else "user", "content": m.content}
                for m in output_messages
            ],
        }

    logger.error(f"Graph returned empty messages. Full response: {response}")
    return {"reply": "Something went wrong processing your request. Please try again."}
