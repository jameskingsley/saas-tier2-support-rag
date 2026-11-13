import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Tier-2 B2B SaaS Support Triage Copilot",
    page_icon="",
    layout="wide",
)

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/api/v1/triage"

# Header
st.title("Tier-2 B2B SaaS Support Triage Copilot")
st.markdown(
    "Enterprise-grade support assistant powered by **FastAPI**, **Qdrant Hybrid Search**, "
    "**Groq (Llama-3)**, and **Langfuse Observability**."
)
st.markdown("---")

# Sidebar Configuration & Telemetry
with st.sidebar:
    st.header("Copilot Settings")
    retrieval_limit = st.slider("Retrieved Knowledge Cards", min_value=2, max_value=8, value=4)
    
    st.markdown("---")
    st.markdown("### System Status")
    st.info("Connected to FastAPI Backend\n\nEndpoint: `http://127.0.0.1:8000`")

# Main Interface Layout
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.subheader("Incoming Support Ticket / Inquiry")
    default_prompt = "Our webhook endpoint is returning 504 gateway timeouts after 10 seconds. How do we fix this?"
    user_query = st.text_area("Paste customer issue or query:", value=default_prompt, height=120)

    if st.button("Run Triage & Generate Resolution", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a valid ticket description.")
        else:
            with st.spinner("Calling FastAPI backend, running hybrid retrieval, and synthesizing triage..."):
                try:
                    payload = {
                        "query": user_query,
                        "limit": retrieval_limit
                    }
                    response = requests.post(API_URL, json=payload, timeout=45)
                    
                    if response.status_code == 200:
                        st.session_state["last_result"] = response.json()
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI backend. Ensure Uvicorn is running (`uvicorn app.api:app --reload`).")
                except Exception as e:
                    st.error(f"Error during request: {e}")

# Display Results if Available
if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    
    st.markdown("---")
    st.subheader("Synthesized Tier-2 Triage Response")
    st.markdown(res["response"])

    # Metrics row
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Prompt Tokens", res["token_usage"]["prompt_tokens"])
    with col_m2:
        st.metric("Completion Tokens", res["token_usage"]["completion_tokens"])
    with col_m3:
        st.metric("Total Tokens", res["token_usage"]["total_tokens"])

    # Retrieved Knowledge Sources Expander
    st.markdown("---")
    st.subheader("Retrieved Knowledge Base Sources (RRF Hybrid Search)")
    for idx, doc in enumerate(res["retrieved_sources"], 1):
        with st.expander(f"Source {idx}: [{doc['source_type'].upper()}] {doc['title']} (Score: {doc['score']:.4f})"):
            st.markdown(f"**Category:** `{doc['category']}` | **Priority:** `{doc['priority']}`")
            st.text(doc["text"])