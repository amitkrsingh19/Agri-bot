# Kerala AgriBot

An agentic farming assistant built for Kerala farmers. Ask about crop care, pest management, government schemes, or seasonal advice — it retrieves answers from real KAU and ICAR documents instead of making things up.

---

## Why this exists

Most agricultural chatbots give generic advice. Kerala has specific crop cycles, a distinct climate, and its own set of government schemes. A farmer asking about rice yellowing in June needs different guidance than one asking the same question in February. This project tries to get that right by grounding every response in actual documents from the Kerala Agricultural University and the Indian Council of Agricultural Research.

---

## Kerala AgriBot answering a crop advisory question
![Kerala AgriBot answering a crop advisory question](/chatbot-screenshot.png)

### Sources
![Sources](/chatbot-screenshot.png)



## How it works

When you send a message, a classifier routes it to one of three agents:

- **Agricultural agent** — crop care, soil, pests, irrigation. Retrieves from the KAU/ICAR vector store, then layers on a Kerala season context before generating a response.
- **RAG chain agent** — government schemes, subsidies, market information. Pure document retrieval, answer sourced directly from ingested PDFs.
- **Logical agent** — general questions and reasoning that don't need document lookup.

Every response from the first two agents includes the source document and page number it drew from.

```
User query
    │
    ▼
Classifier (Gemini, temp=0)
    │
    ├── agricultural ──► RAG retrieval + Kerala season context ──► Gemini generation
    ├── rag_chain    ──► RAG retrieval ──► direct answer + sources
    └── logical      ──► Gemini generation (no retrieval)
    │
    ▼
Response builder (attaches sources + season hint)
    │
    ▼
FastAPI JSON response
```

---

## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Retrieval | LangChain + Chroma |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, no API cost) |
| LLM | Gemini 2.0 Flash |
| Local LLM | llama3.2:3b |
| API | FastAPI |
| UI | Streamlit |
| Containerization | Docker |

---

## Data sources

The knowledge base is built from publicly available government documents:

- **Kerala Agricultural University (KAU)** — crop cultivation manuals for rice, banana, coconut, pepper
- **ICAR** — pest management guides, integrated farming advisories

PDFs are chunked, embedded locally using a HuggingFace model, and stored in a Chroma vector database. The database persists across restarts via a Docker volume.

You can also upload your own PDFs through the UI — they get ingested into the same vector store immediately.

---

## Running locally

**Prerequisites:** Docker and Docker Compose installed. A Gemini API key.

```bash
git clone https://github.com/your-username/kerala-agribot
cd kerala-agribot
cp .env.example .env
# add your GOOGLE_API_KEY to .env
```

```bash
docker compose up --build
```

That's it. The API will be at `http://localhost:8000` and the UI at `http://localhost:8501`.

First startup takes a minute longer — the embedding model (~33MB) downloads once and is cached in a Docker volume after that.

---

## API

FastAPI generates interactive docs automatically at `http://localhost:8000/docs`.

**POST** `/ask`
```json
{
  "message": "My rice leaves are turning yellow, what should I do?",
  "history": []
}
```

```json
{
  "answer": "Yellow leaves in rice during Kerala's summer season...",
  "sources": [
    { "source": "KAU Rice Cultivation Manual", "page": 14 }
  ],
  "season_context": "Current Kerala season: Summer. Likely crops: vegetables, sesame.",
  "intent": "agricultural"
}
```

**POST** `/ingest`  
Upload a PDF to add it to the knowledge base. Accepts `multipart/form-data`.

---

## Project structure

```
kerala-agribot/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── graph.py         # LangGraph agent definition
│   ├── chain.py         # RAG chain with query rewriting
│   ├── ingest.py        # PDF ingestion + vector store
│   ├── rag_service.py   # Cached singletons, ask_rag()
│   ├── logger.py        # Logging
│   ├── config.py        # API key configs
│   └── scrape.py        # scarping logics       
├── data/
│   └── raw/             # Source PDFs (KAU, ICAR)
├── rag_database/        # Chroma vector store (git-ignored)
├── ui.py                # Streamlit frontend
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Environment variables

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Everything else is configured in `app/config.py`.

---

## What's not production-ready

This was built as a learning project for Smart India Hackathon. A few things are still rough:

- No authentication on the `/ingest` endpoint — anyone can add documents
- The Chroma vector store isn't backed up automatically
- Response latency depends on Gemini API — no streaming yet
- The season detection is based on the server's system clock, not the user's location

---

## Acknowledgements

Document sources: Kerala Agricultural University, Indian Council of Agricultural Research (ICAR). Both publish their crop advisories as public domain materials.
