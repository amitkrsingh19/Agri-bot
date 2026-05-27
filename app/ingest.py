from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
import json
from pypdf import PdfReader
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import GOOGLE_API_KEY
from .logger import logger
from .scrape import update_scraped_data

REPO_ROOT = Path(__file__).resolve().parents[1]
VECTORSTORE_DIR = REPO_ROOT / "rag_database"
DATA_DIR = REPO_ROOT / "data"
SCRAPED_JSON_PATH = DATA_DIR / "scraped_chunks.json"

_embedding_instance = None

def get_embedding():
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedding_instance


def create_vector_store(chunks: list[Document], embedding, db_path: str = str(VECTORSTORE_DIR)):
    os.makedirs(db_path, exist_ok=True)
    
    chroma_index = Path(db_path) / "chroma.sqlite3"
    
    if chroma_index.exists():
        # Existing store — just add new chunks
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding)
        if chunks:
            vectorstore.add_documents(chunks)
            logger.info(f"Added {len(chunks)} chunks to existing vectorstore")
    else:
        # Fresh store
        if not chunks:
            raise ValueError("Cannot create a new vectorstore with zero chunks")
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=db_path
        )
        logger.info(f"Created new vectorstore with {len(chunks)} chunks")
    
    return vectorstore


# Data Ingestion and indexing
# Data loading & indexing function
def load_and_split_document(file_path: str = str(SCRAPED_JSON_PATH)):
    """Load documents from a JSON file and split them into chunks."""

    # check for file type
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.") and logger.error("file path not string")
    
    # load the data using json loader
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
      logger.warning(f"Data file not found at {file_path} — returning empty dataset")
      return []
    except json.JSONDecodeError as e:
      logger.error(f"Corrupted JSON at {file_path}: {e} — returning empty dataset")
      return []

    # Normalize/validate json_data is a list of items
    if not isinstance(json_data, list):
        return []

    docs = [
        Document(page_content=item.get('page_content', ''),
                 metadata=item.get('metadata', {}))
        for item in json_data
    ]

    # If no documents were found, return empty list
    if not docs:
        return []
    
    # Split the loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    # split_documents expects a list of Document objects, not raw json dicts
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Loaded and split into {len(chunks)} chunks")
    return chunks

def pdf_reader(pdf):
    pdf_name = pdf.name if hasattr(pdf, "name") else str(pdf)
    reader = PdfReader(pdf)
    
    new_data = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""   # just this page, nothing else
        
        if text.strip():
            new_data.append({
                "page_content": text,
                "metadata": {"source": pdf_name, "page": i + 1}
            })
    
    if not new_data:
        logger.warning(f"No extractable text found in {pdf_name} — scanned PDF?")
        return
    
    update_scraped_data(new_data)
    logger.info(f"Saved {len(new_data)} raw pages from {pdf_name}")

def ingest_pdf(file) -> str:
    """Ingest a PDF, update persisted chunks, and rebuild retrieval chain."""
    pdf_reader(file)
    return "Knowledge base updated with uploaded PDF."