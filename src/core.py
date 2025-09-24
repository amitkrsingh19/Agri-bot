# Imports
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
import json
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders.web_base import WebBaseLoader
from bs4.filter import SoupStrainer
from urllib.parse import urlparse
from pypdf import PdfReader
import os
from logger import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

llm = ChatGoogleGenerativeAI(model="gemini-pro")


# Ensure an identifying USER_AGENT is set for web requests (helps with polite scraping and avoids warnings)
# You can override this in your environment if you prefer a different identifier.
os.environ.setdefault("USER_AGENT", "TerraAI/1.0 (+https://example.com)")

file_path = "C:\\krishisahayi\\lang-chain-bot\\data\\scraped_chunks.json"
SCRAPED_FILE = "scraped.json"
# Data Ingestion and indexing
# Data loading & indexing function
def load_and_split_document(file_path:str):
    """
    load document from json file and splits then into chunks
    return: list of document chunks
    """
    # check for file type
    if not isinstance(file_path,str):
        raise TypeError("file_path must be a string.") and logger.error("file path not string")
    
    # load the data using json loader
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        # No data file yet — return empty list of chunks so caller can handle it
        return []
    except json.JSONDecodeError:
        # Bad JSON — treat as empty dataset and surface a warning via exception to caller if needed
        return []

    # Normalize/validate json_data is a list of items
    if not isinstance(json_data, list):
        return []

    docs = [
        Document(page_content=item.get('page_content', ''),
                 metadata=item.get('metadata', {}))
        for item in json_data
    ]
# Split the loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,chunk_overlap=200)
    # If no documents were found, return empty list
    if not docs:
        return []

    # split_documents expects a list of Document objects, not raw json dicts
    chunks = text_splitter.split_documents(docs)
    logger.info("text splitted using tiktoken")
    return chunks

#update the json file after scraping
def update_scraped_data(new_data):
    #make a list
    existing_data = []

    # ensure parent directory exists
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # load existing data safely
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing_data = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read existing scraped file: {e}. Starting fresh.")

    # extend and write atomically
    existing_data.extend(new_data)
    tmp_path = file_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        # atomic replace
        os.replace(tmp_path, file_path)
        logger.info(f"Saving {len(new_data)} new chunks. Total size now {len(existing_data)}")
    except Exception as e:
        logger.error(f"Failed to write scraped data to {file_path}: {e}")
        # cleanup tmp if exists
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass



def pdf_reader(pdf):
    # Handle UploadedFile or raw path
    pdf_name = pdf.name if hasattr(pdf, "name") else str(pdf)
    reader = PdfReader(pdf)
    logger.info(f"Read PDF: {pdf_name}")
    # Create text
    text=""
    for page in reader.pages:
        text+=page.extract_text() +"\n"
    # Split docs
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    all_splits = text_splitter.create_documents([text])
    logger.info(f"Split into {len(all_splits)} chunks")

    with open("C:\\krishisahayi\\lang-chain-bot\\data\\scraped_chunks.json", "a", encoding="utf-8") as f:
        json.dump(
        [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in all_splits],
        f,
        ensure_ascii=False,
        indent=2
    )
    logger.info(f"split and saved chunks sucessfully")

def web_scrapper(url:str):
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = "https://" + url
    # strainer
    bs4_strainer = SoupStrainer(class_=("content-column-content","content-area","site-content"))
    logger.info(f"scraping the url {url}")
    # load the document from web
    loader = WebBaseLoader(web_path=url,
                           bs_kwargs={"parse_only": bs4_strainer})
    # load and split
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    # write a simple serializable structure (page_content + metadata)
    with open("C:\krishisahayi\lang-chain-bot\data\scraped_chunks.json", "a", encoding="utf-8") as f:
        json.dump([doc.model_dump() for doc in all_splits], f, ensure_ascii=False, indent=2)
    logger.info(f"successfully scraped document: {len(docs)} and saved it")
    return "successfully scraped and saved it"


# function to store embeddings in vector db
def create_embedding():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    hf = HuggingFaceEmbeddings(model_name=model_name)
    logger.info("created embeddings")
    return hf

def create_vector_store(chunks:list[Document],embedding,
                                   db_path:str):
    """
    Creates a Chroma vector store from document chunks and saves it.
    
    Args:
        chunks: A list of Document objects (your text chunks).
        embeddings: The embeddings model object.
        db_path: The directory path to save the vector store.
    
    Returns:
        The Chroma vector store object.
    """
    # Ensure directory exists
    os.makedirs(db_path, exist_ok=True)
    # Try to load an existing persisted Chroma DB
    try:
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding)
        logger.info(f"vectorstore created to {db_path}")
        # If there are new chunks, add them and persist.
        if chunks:
            try:
                vectorstore.add_documents(chunks)
                logger.info(f"added new chunks of len: {len(chunks)}")
            except Exception as e:
                logger.error(f"Exception occured {e}")
        return vectorstore
    except Exception:
        # Fallback: create a new vectorstore and persist it
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=db_path
        )
        try:
            vectorstore.persist()
        except Exception as e:
            logger.error(f"error persisiting {e}")
        return vectorstore

def build_refine_chain(retriever):
    # Prompt to extract raw text from each document
    # The retrieval/refine helpers expect the document variable to be named 'context'
    document_prompt = PromptTemplate(
        input_variables=["context"],
        template="{context}"
    )
    logger.info("refined_chain_building")
    # Variable names must match what RefineDocumentsChain expects
    document_variable_name = "context"
    initial_response_name = "prev_response"

    logger.info("loaded the model llama3.2:3b")
    # Initial summarization prompt — produce concise, actionable, safety-aware farming guidance
    initial_prompt = PromptTemplate.from_template("""
    You are an expert agronomist and practical farming advisor. Speak directly to a farmer in clear, non-technical language.

    Output structure (always follow):
    1) Short summary (1-3 sentences) with the main recommendation.
    2) Numbered, step-by-step actionable instructions (include quantities, units, timing, and thresholds when applicable).
    3) Quick diagnostics: what to observe and how to verify the problem and solution.
    4) Safety & environmental notes (PPE, label/legal constraints, runoff, withdrawal periods, etc.) when relevant.
    5) Sources: cite any supporting context items using bracketed references like [1], [2]; if no sources in context, say "No supporting sources available in context.".

    Rules:
    - If required information is missing (crop, growth stage, location/climate, duration), ask ONE concise clarifying question before giving a prescriptive plan.
    - Prefer concrete recommendations over vague statements. Use measurable thresholds where possible (e.g., "apply 1.5 L/ha on day 0 and reapply after 14 days if symptoms persist").
    - Keep answers concise (prefer under 500 words) and practical.

    Context: {context}
    Question: {input}
    """)

    # Initial chain (first pass)
    initial_llm_chain = LLMChain(llm=llm, prompt=initial_prompt)
    logger.info("first initial chain")
    # Refine prompt (takes previous + new context)
    refine_prompt = PromptTemplate.from_template(
        "You previously produced this answer: {prev_response}.\n"
        "A new piece of context is available: {context}.\n\n"
        "Instructions:\n"
        "- Incorporate the new context only where it changes or improves the recommendation.\n"
        "- Preserve the output structure: concise summary, numbered actionable steps, diagnostics, safety notes, and sources.\n"
        "- If the new context contradicts earlier assumptions, explicitly state which assumptions changed and why, then provide corrected action steps.\n"
        "- Update or add source citations as needed. If the new context provides no useful information, return the original answer unchanged and note that no update was necessary.\n\n"
        "Return the updated final answer and a short bullet list titled 'Changes made' describing what was updated."
    )

    # Refine chain (subsequent passes)
    refine_llm_chain = LLMChain(llm=llm, prompt=refine_prompt)
    logger.info("refined llm chain")
    # Combine into a RefineDocumentsChain
    refine_chain = RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,
        refine_llm_chain=refine_llm_chain,
        document_variable_name=document_variable_name,   
        initial_response_name=initial_response_name      
    )
    # Create a "stuff" documents chain that returns a string and uses the same LLM;
    # the retrieval helper expects a combine_docs_chain that outputs a string, so use this.
    docs_chain = create_stuff_documents_chain(llm=llm,prompt=document_prompt)
    logger.info("docs chain created")
    # Optionally keep refine_chain for separate usage, but pass docs_chain to create_retrieval_chain
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain=docs_chain)
    return retrieval_chain
