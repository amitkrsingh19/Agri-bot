import os

try:
    import streamlit as st
    # Use Streamlit secrets 
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    # Fallback for local dev
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is missing! Add it to .env or Streamlit secrets.")
