from typing import Annotated, Literal, Any, List, Dict
from pydantic import Field, BaseModel,Field
from langgraph.graph import START, END, StateGraph
from operator import add
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

from .logger import logger
from .config import GOOGLE_API_KEY
from .rag_service import get_retrieval_chain 
from .chain import get_season_hint

# --- State & Schemas ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    intent: str | None
    draft_answer: str | None
    sources: Annotated[List[Dict], ...]
    season_context : str | None

class MessageClassifier(BaseModel):
    intent: Literal["agricultural", "rag_chain", "logical"] = Field(
        ...,
        description="Classify query: 'agricultural' (farming/crop advice), 'rag_chain' (factual docs/schemes), 'logical' (general reasoning)."
    )


# --- LLM Setup ---
"""
classifier_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY, temperature=0.0)
generation_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY, temperature=0.7)
"""

from langchain_ollama import ChatOllama

# By default, ChatOllama looks for the local server running on http://localhost:11434
classifier_llm = ChatOllama(
    model="llama3.2:3b", 
    temperature=0.0
)

generation_llm = ChatOllama(
    model="llama3.2:3b", 
    temperature=0.7
)

# truncate_text function - for clear logs
def truncate_text(text: Any, max_length: int = 50) -> str:
    """Keeps logs clean by shortening long user inputs."""
    if not text:
        return "None"
    return text if len(text) <= max_length else text[:max_length] + "..."

# 1. Classifier Node
def classifier_node(state: State):
    user_query = state["messages"][-1].content
    logger.info(f"[NODE: Classifier] Started. Analyzing query: '{truncate_text(user_query)}'")
    
    structured_llm = classifier_llm.with_structured_output(MessageClassifier)
    try:
        result = structured_llm.invoke([
            SystemMessage("Classify the farmer's query into exactly one category:\n"
            "- 'agricultural': crop care, soil, pests, irrigation, farming techniques\n"
            "- 'rag_chain': government schemes, market prices, document lookup, subsidies\n"
            "- 'logical': general knowledge, calculations, anything unrelated to farming\n"
            "Reply only with the structured output."),
            HumanMessage(user_query)
        ])
        intent = result.intent
        logger.info(f"[NODE: Classifier] Success. Routed to -> '{intent}'")
    except Exception as e:
        logger.error(f"Classification failed: {e}. Falling back to -> 'logical'")
        intent = "logical" # Safe default
        
    return {"intent": intent}


# 2. Agricultural Agent Node (RAG + Kerala System Prompt)
def agricultural_agent(state: State):
    user_query = state["messages"][-1].content
    logger.info(f"[NODE: Agricultural] Started. Processing agricultural logic.")
    
    chain = get_retrieval_chain()
    result = chain.invoke({"input": user_query})
    context = result.get("answer", "")
    sources = result.get("sources", [])
    
    season_hint = get_season_hint()

    logger.debug(f"[NODE: Agricultural] Retrieved context length: {len(context)} chars")
    
    system_prompt = f"""You are an expert Kerala agricultural assistant.
    Current season context: {season_hint}
    Use this retrieved context to inform your advice: {context}
    Focus on: soil health, IPM, water conservation, Kerala crop cycles."""
    
    reply = generation_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_query)
    ])
    logger.info("[NODE: Agricultural] Success. Draft generated.")
    return {"draft_answer": reply.content, "sources": sources, "season_context":season_hint}


def rag_chain_node(state: State):
    user_query = state["messages"][-1].content
    logger.info(f"[NODE: RAG Chain] Started. Searching documents for factual data.")
    
    chain = get_retrieval_chain()
    result = chain.invoke({"input": user_query})
    sources = state.get("sources")

    logger.info(f"[NODE: RAG Chain] Success. Found {len(sources)} sources.")

    return {
        "draft_answer": result.get("answer", "No relevant documents found."),
        "sources": result.get("sources", [])
    }

# 4. Logical Agent Node (No RAG)
def logical_agent(state: State):
    user_query = state["messages"][-1].content
    logger.info(f"[NODE: Logical] Started. Applying general reasoning without RAG.")

    system_prompt = (
        "You are a general reasoning assistant. Answer step-by-step using only what you know. "
        "If the question is about Kerala farming, government schemes, or specific crop data, "
        "say you don't have that specific information and suggest the user ask more specifically."
    )
    reply = generation_llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_query)
    ])
    logger.info("[NODE: Logical] Success. Reasoning complete.")

    # No sources since it's pure reasoning
    return {"draft_answer": reply.content, "sources": []}

# 5. Response Builder Node
def response_builder(state: State):
    logger.info("[NODE: Response Builder] Started. Compiling final JSON payload.")
    

    draft = state.get("draft_answer", "I couldn't process that.")
    sources = state.get("sources", [])
    
    # season_hint = get_season_hint()
    season_hint = state.get("season_context")
    
    logger.info(f"[NODE: Response Builder] Success. Payload ready (Sources: {len(sources)}, Season attached). Ending Graph.")
    return {
        "messages": [AIMessage(content=draft)],
        "sources": sources,
        "season_context": season_hint
    }
    

# --- Graph Compilation ---
def create_graph():
    logger.info("Compiling StateGraph...")
    graph_builder = StateGraph(State)
    
    # Add Nodes
    graph_builder.add_node("classifier", classifier_node)
    graph_builder.add_node("agricultural", agricultural_agent)
    graph_builder.add_node("rag_chain", rag_chain_node)
    graph_builder.add_node("logical", logical_agent)
    graph_builder.add_node("builder", response_builder)
    
    # Flow: START -> Classifier
    graph_builder.add_edge(START, "classifier")
    
    # Flow: Classifier -> (Conditional Route) -> Agents
    graph_builder.add_conditional_edges(
        "classifier",
        lambda state: state["intent"], 
        {
            "agricultural": "agricultural",
            "rag_chain": "rag_chain",
            "logical": "logical",
        }
    )
    
    # Flow: Agents -> Response Builder
    graph_builder.add_edge("agricultural", "builder")
    graph_builder.add_edge("rag_chain", "builder")
    graph_builder.add_edge("logical", "builder")
    
    # Flow: Response Builder -> END
    graph_builder.add_edge("builder", END)
    
    return graph_builder.compile()