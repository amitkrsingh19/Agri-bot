# Imports
import os
from .logger import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import GOOGLE_API_KEY
from langchain_core.prompts import PromptTemplate



REWRITE_PROMPT = PromptTemplate.from_template(
    "Rewrite this farmer's question into a precise agronomic search query. "
    "Keep it under 20 words. Question: {question}\nRewritten:"
)

KERALA_SEASON_CONTEXT = {
    "kharif":  ("June-September",  ["rice", "tapioca", "vegetables"]),
    "rabi":    ("October-January", ["rice", "banana", "pulses"]),
    "summer":  ("February-May",    ["vegetables", "sesame", "groundnut"]),
}


def get_season_hint() -> str:
    from datetime import datetime
    month = datetime.now().month
    if 6 <= month <= 9:
        season, crops = KERALA_SEASON_CONTEXT["kharif"]
    elif 10 <= month <= 1 or month == 1:
        season, crops = KERALA_SEASON_CONTEXT["rabi"]
    else:
        season, crops = KERALA_SEASON_CONTEXT["summer"]
    return f"Current Kerala season: {season}. Likely crops: {', '.join(crops)}."


"""
os.environ.setdefault("USER_AGENT", "TerraAI/1.0 (+https://example.com)")
# Initialize the LLM interface (Gemini via Google GenAI)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)
"""

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:3b", 
    temperature=0.0
)

# Rewrite the query before retrieval
def rewrite_query(query: str) -> str:
    response = llm.invoke(REWRITE_PROMPT.format(question=query))
    return response.content.strip()


# this function builds and return retrieval chain
def build_refine_chain(retriever):
    """Build a simple retrieval-augmented generation chain.
    """

    system_prompt ="""
    You are an expert agricultural advisor for farmers in Kerala, India. 
    Your goal is to provide practical, accurate, and localized advice based on the 
    current season and weather patterns. Always factor in the provided context.
"""
    template_string = """
    Current Kerala Agricultural Context:
    {season_context}
    
    Document Context:
    {context}
    
    Question:
    {input}
    
    Answer:
"""
    prompt_template = PromptTemplate(
        template = template_string,
        input_variables = ["context" , "input"],
        partial_variables = {"season_context": get_season_hint} 
    )

    class SimpleRAGChain:
        def __init__(self, retriever):
            self.retriever = retriever

        def invoke(self, payload):
            query = payload.get("input", "")
            if not query:
                return {"answer": "No query provided."}

            # Retrieve top documents
            query = rewrite_query(query)
            docs = self.retriever.invoke(query)

            context = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
            prompt = prompt_template.format(context=context, input=query)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            sources = [
                {
                    "source" : d.metadata.get("source","unknown"),
                    "page" : d.metadata.get("page",None)
                }
                for d in docs
            ] 

            response = llm.invoke(messages)
            answer = getattr(response, "content", None)
            if answer is None:
                # If the response is not a standard message object, fall back to str()
                answer = str(response)
            
            return {"answer": answer ,"sources" : sources}

    return SimpleRAGChain(retriever)
