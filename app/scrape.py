from urllib.parse import urlparse
import json
import os
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from .logger import logger


# Paths are kept relative to the repository root for portability.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SCRAPED_JSON_PATH = DATA_DIR / "scraped_chunks.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR = REPO_ROOT / "rag_database"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_METADATA = VECTORSTORE_DIR / "vectorstore.pkl"



#update the json file after scraping
def update_scraped_data(new_data):
  """ Append new scraped chunks to the persisted JSON file.

    This function safely loads existing data (if any), appends the new
    chunks, and writes back using atomic replace to avoid corruption.
  """ 

  existing_data = []

    # Ensure directory exists
  SCRAPED_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data safely
  if SCRAPED_JSON_PATH.exists():
      try:
          with open(SCRAPED_JSON_PATH, "r", encoding="utf-8") as f:
              content = f.read().strip()
              if content:
                 existing_data = json.loads(content)
      except (FileNotFoundError, json.JSONDecodeError) as e:
          logger.warning(f"Could not read existing scraped file: {e}. Starting fresh.")

  existing_data.extend(new_data)
  tmp_path = SCRAPED_JSON_PATH.with_suffix(".json.tmp")
  try:
      with open(tmp_path, "w", encoding="utf-8") as f:
          json.dump(existing_data, f, ensure_ascii=False, indent=2)
      # atomic replace
      os.replace(tmp_path, SCRAPED_JSON_PATH)
      logger.info(f"Saving {len(new_data)} new chunks. Total size now {len(existing_data)}")
  except Exception as e:
      logger.error(f"Failed to write scraped data to {SCRAPED_JSON_PATH}: {e}")
      # cleanup tmp if exists
      try:
          if tmp_path.exists():
              tmp_path.unlink()
      except Exception:
          pass



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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    # write a simple serializable structure (page_content + metadata)
    new_data = [doc.model_dump() for doc in all_splits]
    update_scraped_data(new_data)
    logger.info(f"successfully scraped document: {len(docs)} and saved it")
    return "successfully scraped and saved it"



