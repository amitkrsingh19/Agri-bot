import streamlit as st
import time
from core import load_and_split_document, create_embedding, create_vector_store, pdf_reader, build_refine_chain
from logger import logger 
from graph import create_graph, State
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List, cast


@st.cache_resource
def setup_rag_system():
    logger.info("Attempting to set up RAG system...")
    # Use a relative path, which works on both local and cloud environments
    data_file = "data/scraped_chunks.json"
    db_path = "rag_database"
    chunks = load_and_split_document(data_file)
    embedding = create_embedding()

    if not chunks:
        logger.warning("No new document chunks found. Trying to load persisted vector store.")
        try:
            vector_store = create_vector_store([], embedding, db_path)
            retriever = vector_store.as_retriever()
            logger.info("Successfully loaded persisted vector store from disk.")
            retrieval_chain = build_refine_chain(retriever)
            logger.info("Built refined retrieval chain from persisted DB.")
            return retrieval_chain
        except Exception as e:
            logger.error(f"Failed to load persisted vector store: {e}")
            class _FallbackChain:
                def invoke(self, payload: dict):
                    return {"answer": "Knowledge base is empty. Please scrape a website or add documents first."}
            return _FallbackChain()
    
    vector_store = create_vector_store(chunks, embedding, db_path)
    retriever = vector_store.as_retriever()
    logger.info("Retriever instance created from scraped chunks.")
    retrieval_chain = build_refine_chain(retriever)
    logger.info("Built refined retrieval chain from scraped chunks.")
    return retrieval_chain

@st.cache_resource
def get_graph():
    logger.info("Compiling and caching LangGraph.")
    return create_graph()

def setup_ui():
    st.set_page_config(
        page_title="🌱 Terra AI: Your Agricultural Assistant",
        page_icon="🌾",
        layout="wide",
    )
    logger.info("UI page config and styles loaded.")

def render_messages(container):
    with container:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            # Fixed: Access the content property of the LangChain message object
            content = msg.content
            if role == "assistant":
                st.markdown(f"<div class='chat-line bot'><div class='avatar'>🌱</div>"
                            f"<div class='chat-bubble bot'>{content}</div></div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-line user'><div class='chat-bubble user'>{content}</div>"
                            f"<div class='avatar'>👩‍🌾</div></div>",
                            unsafe_allow_html=True)
        # Fixed: Removed the extra closing div markdown from inside the loop
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    logger.info("Application starting...")
    setup_ui()
    st.markdown("<h2 style='text-align:center;color:#2d6a4f;'>🌾 Terra AI: Your Agricultural Assistant</h2>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        # Fixed: Use a LangChain AIMessage object for initialization
        st.session_state.messages = cast(List[BaseMessage], [
            AIMessage(content="Hello! I'm Terra 🌱. I'm here to help with your farming questions.")
        ])
        logger.info("Session initialized with a LangChain message object.")
        
    if "retrieval_chain" not in st.session_state:
        st.session_state.retrieval_chain = setup_rag_system()
        logger.info("RAG system initialized and stored in session state.")

    with st.sidebar:
        st.header("⚙️ App Control")
        pdf = st.file_uploader(label="📄 Upload a PDF", type="pdf")
        if st.button("Upload"):
            if pdf:
                logger.info("PDF upload button clicked.")
                with st.spinner("Processing PDF..."):
                    try:
                        pdf_reader(pdf)
                        st.session_state.retrieval_chain = setup_rag_system()
                        logger.info("Knowledge base successfully updated from PDF.")
                        st.success("✅ Knowledge base updated!")
                    except Exception as e:
                        logger.error(f"❌ Failed to process PDF: {e}")
                        st.error(f"❌ Failed: {e}")
            else:
                st.warning("Please upload a PDF first.")
                logger.warning("Upload button clicked with no PDF file selected.")

    chat_container = st.container()
    render_messages(chat_container)

    if prompt := st.chat_input("Type your farming question here..."):
        logger.info(f"User submitted a message: '{prompt}'")
        with st.spinner("Thinking... 🌾"):
            time.sleep(1) 
        
        # Append the new message as a LangChain HumanMessage object
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        # Prepare the state dictionary for LangGraph
        initial_state = {
            "messages": st.session_state.messages,
            "rag_chain": st.session_state.retrieval_chain 
        }
        
        graph = get_graph()
        
        try:
            logger.info("Invoking LangGraph with the current state.")
            response = graph.invoke(initial_state) #type:ignore
            final_message_obj = response["messages"][-1]
            logger.info(f"final_message : {final_message_obj}")
            st.session_state.messages.append(final_message_obj)
            
            logger.info(f"LangGraph execution complete. Response received.")
        except Exception as e:
            logger.error(f"An error occurred during graph invocation: {e}")
            st.session_state.messages.append(
                AIMessage(content="I'm sorry, an error occurred while processing your request. Please try again.")
            )
    render_messages(chat_container)
if __name__ == "__main__":
    main()