import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000/ask"

st.set_page_config(
    page_title="Kerala AgriBot",
    page_icon="🌱",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
<style>
/* Intent badges */
.badge {
    display: inline-block;
    font-size: 11px;
    padding: 2px 10px;
    border-radius: 99px;
    margin-bottom: 6px;
    font-weight: 500;
}
.badge-agri  { background: #E1F5EE; color: #085041; }
.badge-rag   { background: #E6F1FB; color: #0C447C; }
.badge-logic { background: #F1EFE8; color: #444441; }

/* Season pill in header */
.season-pill {
    display: inline-block;
    font-size: 12px;
    padding: 3px 12px;
    border-radius: 99px;
    background: #E1F5EE;
    color: #085041;
}

/* Source item */
.source-item {
    font-size: 12px;
    padding: 4px 10px;
    border: 0.5px solid #e0e0e0;
    border-radius: 6px;
    margin-bottom: 4px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)


# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaskaram! How can I help you with your farm today?",
            "sources": [],
            "season": "",
            "intent": ""
        }
    ]

if "last_season" not in st.session_state:
    st.session_state.last_season = ""


# --- Header ---
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("### 🌱 Kerala AgriBot")
    st.caption("Ecological farming assistant")
with col2:
    if st.session_state.last_season:
        st.markdown(
            f'<div class="season-pill">☀️ {st.session_state.last_season}</div>',
            unsafe_allow_html=True
        )

st.divider()


# --- Intent badge helper ---
INTENT_BADGES = {
    "agricultural": ('badge-agri',  "🌾 Agricultural advice"),
    "rag_chain":    ('badge-rag',   "📄 Document lookup"),
    "logical":      ('badge-logic', "💡 General reasoning"),
}

def render_intent_badge(intent: str):
    if not intent:
        return
    css_class, label = INTENT_BADGES.get(intent, ('badge-logic', intent))
    st.markdown(
        f'<span class="badge {css_class}">{label}</span>',
        unsafe_allow_html=True
    )


# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("intent"):
            render_intent_badge(msg["intent"])

        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    name = src.get("source", "Unknown source")
                    page = src.get("page")
                    label = f"{name} · p.{page}" if page else name
                    st.markdown(
                        f'<div class="source-item">📄 {label}</div>',
                        unsafe_allow_html=True
                    )


# --- Chat input ---
if prompt := st.chat_input("Ask about crops, pests, or schemes..."):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({
        "role": "user", "content": prompt,
        "sources": [], "season": "", "intent": ""
    })

    # Build clean history (role + content only)
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # exclude the message just added
    ]

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                response = requests.post(
                    FASTAPI_URL,
                    json={"message": prompt, "history": clean_history},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                answer       = data.get("answer", "I couldn't process that.")
                sources      = data.get("sources", [])
                season       = data.get("season_context", "")
                intent       = data.get("intent", "")

                # Update season in header
                if season:
                    st.session_state.last_season = season

                render_intent_badge(intent)
                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 {len(sources)} source(s)"):
                        for src in sources:
                            name  = src.get("source", "Unknown source")
                            page  = src.get("page")
                            label = f"{name} · p.{page}" if page else name
                            st.markdown(
                                f'<div class="source-item">📄 {label}</div>',
                                unsafe_allow_html=True
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "season": season,
                    "intent": intent
                })

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Is uvicorn running on port 8000?")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The model may be loading — try again.")
            except requests.exceptions.HTTPError as e:
                st.error(f"Backend error {e.response.status_code}: {e.response.text[:200]}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")