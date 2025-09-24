from langchain_ollama.chat_models import ChatOllama
from typing import Annotated,Literal,Any
from langgraph.graph import START,END,StateGraph
from pydantic import Field,BaseModel
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from logger import logger

llm = ChatOllama(model="llama3.2:3b")

# states to be saved at each nodes
class State(TypedDict):
    messages: Annotated[list,add_messages] # manages coversation history
    message_types: str | None
    question: str | None
    context: str | None
    rag_chain: Any
    
class MessageClassifier(BaseModel):
    message_types: Literal["agricultural","logical","emotional"] = Field(
        ...,
        description ="Classify if the message requires an agricultural or farming schemes information or logical response"
    )

def classify_message(state:State):
    """
    Classifies the user's message into one of three categories:
    'agricultural', 'informational', or 'logical'.
    """
    last_message = state["messages"][-1] # get the last message 
    logger.info("Starting Message Classifications...")
    system_prompt = """
    Classify the user's message based on its core intent:

    - 'agricultural': **Hands-on farming advice.** Questions about crop care, soil, pests, diseases, or agricultural techniques.
    - 'informational': **Factual data retrieval.** Queries about government schemes, market prices, or content from documents/websites.
    - 'logical': **General knowledge/reasoning.** Simple questions, calculations, or topics unrelated to farming.
    
    Provide only one of the three category names as your output.
    """
    classifier_llm= llm.with_structured_output(MessageClassifier) 
    user_query = last_message.content
    try:
        result = classifier_llm.invoke([
            {"role":"system",
            "content":system_prompt},
            {"role":"user","content": user_query}
        ]) 
        message_types = result.message_types #type: ignore
        logger.info(f"Message Classified as: {message_types}")
        return {"message_types": message_types ,"question":user_query} #type: ignore
    except Exception as e:
        logger.error(f"Classification failed: {e}. Defaulting to 'logical'.")
        return {"message_types": "logical","question":user_query} 

def call_rag_chain(state: State):
    # This function now expects the RAG chain to be in the state
    logger.info("Executing RAG chain for informational query.")
    rag_chain = state["rag_chain"]
    last_message = state["messages"][-1]
    response = rag_chain.invoke({"input": last_message.content})
    answer = response.get("answer", "I'm not sure. Please rephrase or upload a document.")
    logger.info("RAG chain execution complete.")
    return {"messages": [{"role": "assistant", "content": answer}]}


def router(state: State):
    logger.info("Routing based on message type...")
    message_type = state.get("message_types")
    
    if message_type == "informational":
        logger.info("Message type 'informational' detected. Routing to 'informational_rag_node'.")
        return {"next": "informational_rag_node"}
    
    elif message_type == "agricultural":
        logger.info("Message type 'agricultural' detected. Routing to 'agricultural'.")
        return {"next": "agricultural"}
    
    else: # This acts as the default or fallback
        logger.info("Message type is not 'informational' or 'agricultural'. Routing to 'logical'.")
        return {"next": "logical"}


def agricultural_agent(state: State):
    """
    Provides a sustainable farming response based on user input and available context.
    """
    logger.info("Executing agricultural agent...")
    
    # Get the last message from the state
    last_message = state["messages"][-1]

    # Check if the object has a .content attribute, which is common for LangChain messages
    if hasattr(last_message, 'content'):
        user_query = last_message.content
    else:
        # Fallback for unexpected message types
        user_query = str(last_message)
        logger.warning(f"Unexpected message type in state: {type(last_message)}. Using its string representation.")
    
    context = state.get("context", "")
    question = state.get("question", user_query)
    
    # Enhanced, concise system prompt
    system_prompt = f"""
    You are a sustainable farming expert and an ecological guide.
    
    **Instructions:**
    1.  Use the following `Context` to answer the `Question`.
    2.  If the context is insufficient, provide a solution based on your general knowledge.
    3.  Your advice must promote soil health, integrated pest management (IPM), and water conservation.
    4.  Encourage long-term sustainability and a holistic view of the farm ecosystem.
    
    **Context:** {context}
    **Question:** {question}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    try:
        reply = llm.invoke(messages)
        logger.info("Agricultural agent execution complete.")
        return {"messages": [{"role": "assistant", "content": reply.content}]}
    except Exception as e:
        logger.error(f"Error in agricultural_agent: {e}")
        return {"messages": [{"role": "assistant", "content": "I'm sorry, an error occurred while processing your request. Please try again."}]}



def logical_agent(state:State):
    logger.info("Executing logical agent...")
    last_message = state["messages"][-1]
    user_query = state.get("question","")
    messages =[
        {"role":"system",
        "content":f"""You are a Logical Replier. Your only function is to provide a structured, reasoned response to any query.
        question:{user_query}.
        Your process:
    Deconstruct: Identify the core parts of the user's query.
    Analyze: Use step-by-step logic to process the information.
    Conclude: Provide a direct, factual conclusion based on your analysis.
    Be concise, objective, and use a structured format (e.g., numbered lists). Avoid conversational fillers, opinions, or unnecessary details.
         """},
         {"role":"user",
          "content":last_message.content}
    ]
    reply = llm.invoke(messages)
    logger.info("Logical agent execution complete.")
    return {"messages":[{"role":"assistant",'content':reply.content}]}

def informational_agent(state:State):
    logger.info("Executing informational agent...")
    last_message = state["messages"][-1]
    user_query = state.get("question","")
    context = state.get("context","")
    messages =[
        {"role":"system","content":f"""
         question:{user_query}
         context:{context}
         you are a rag agent you have certain pdf and data arround the informaional farming schemes by government and crop knowledge with
         disease detection and cure recommendation with keeping the cure as envirometnal friendly and healthy for soil and seasonal crops.
         Your job will be to evaluate the answers with all the informational context. 
         """},
         {"role":"user",
          "content":last_message.content}
    ]
    reply = llm.invoke(messages)
    logger.info("Informational agent execution complete.")
    return {"messages":[{"role":"assistant",'content':reply.content}]}

def create_graph():
    logger.info("Starting graph compilation...")
    graph_builder = StateGraph(State)
# Nodes
    graph_builder.add_node("classifier",classify_message)
    graph_builder.add_node("router",router)
    graph_builder.add_node("informational_rag_node",call_rag_chain)
    graph_builder.add_node("agricultural",agricultural_agent)
    graph_builder.add_node("logical",logical_agent)

    graph_builder.add_edge(START,"classifier")
    graph_builder.add_edge("classifier","router")
    
    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {"agricultural": "agricultural","informational":"informational_rag_node","logical":"logical"}
    )
    
    graph_builder.add_edge("agricultural",END)
    graph_builder.add_edge("informational_rag_node",END)
    graph_builder.add_edge("logical",END)

    graph = graph_builder.compile()
    logger.info("Graph Compilation Complete...")
    return graph
    
def run_chatbot():
    state = {"messages":[],"message_types":None}
    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("bye")
            break
        # create a message to model
        state["messages"] = state.get("messages",[]) +[
            {"role":"user","content":user_input}
            ]
        graph = create_graph()
        state = graph.invoke(state) #type: ignore
        if state.get("messages") and len(state["messages"])> 0:
            last_message = state["messages"][-1]
            print(f"Assistant:{last_message.content}")

