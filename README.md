# Tier-2 B2B SaaS Support Triage Copilot

An enterprise-grade, production-ready RAG support triage assistant powered by **FastAPI**, **Qdrant Hybrid Search**, **Groq (Llama-3)**, and **Langfuse Observability**. Designed to automate Tier-2 B2B SaaS ticket analysis, reduce mean time to resolution (MTTR), and enforce rigorous evidence-based grounding.

---

## 🚀 Live Deployments
* **Streamlit Frontend UI:** [https://saas-tier2-support-rag-sv6h3mknrt672bqtdacarb.streamlit.app/](https://saas-tier2-support-rag-sv6h3mknrt672bqtdacarb.streamlit.app/)
* **FastAPI Backend & Interactive Docs (Swagger):** [https://saas-tier2-support-rag.onrender.com/docs](https://saas-tier2-support-rag.onrender.com/docs)

---

## 🏗️ Architecture & Tech Stack

* **Backend API:** FastAPI running on Render with Uvicorn.
* **Frontend UI:** Streamlit Cloud interface configured for asynchronous API calls.
* **Vector Storage & Retrieval:** Qdrant Cloud vector database utilizing sparse-dense hybrid search with **Reciprocal Rank Fusion (RRF)** via `fastembed` (`BAAI/bge-small-en-v1.5` + BM25).
* **LLM Inference:** Groq high-performance inference engine.
* **Observability & Evaluation:** Langfuse tracing and LLM-as-a-Judge validation pipelines.

---

## 📂 Project Structure

```text
saas-tier2-support-rag/
├── .github/
│   └── workflows/
│       └── pytest.yml          # CI/CD pipeline for automated test suites
├── app/
│   ├── api.py                  # FastAPI application & /api/v1/triage endpoints
│   └── streamlit_app.py        # Streamlit frontend client interface
├── data/
│   ├── processed/              # Chunked and cleaned knowledge base cards (.jsonl)
│   └── raw/                    # Raw documentation markdown & historical support tickets
├── src/
│   ├── ingestion/
│   │   ├── chunker.py          # Document chunking & metadata tagging
│   │   ├── fetch_data.py       # Knowledge base data gatherer
│   │   └── indexer.py          # Vector embedding generator & Qdrant uploader
│   └── pipeline/               # RAG retrieval, RRF ranking, and Groq synthesis logic
├── evaluation/
│   └── evaluate_copilot.py     # LLM-as-a-judge accuracy and out-of-domain refusal test suite
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render multi-service Blueprint configuration
└── README.md

⚙️ Local Development Setup

1. Clone the Repository & Setup Virtual Environment

git clone [https://github.com/jameskingsley/saas-tier2-support-rag.git](https://github.com/jameskingsley/saas-tier2-support-rag.git)
cd saas-tier2-support-rag

python -m venv .venv

.venv\Scripts\Activate

pip install --upgrade pip

pip install -r requirements.txt

2. Configure Environment Variables

Create a .env file in the root directory with your credentials:

Code snippet

GROQ_API_KEY=your_groq_api_key

QDRANT_URL=your_qdrant_cloud_url

QDRANT_API_KEY=your_qdrant_api_key

QDRANT_COLLECTION_NAME=saas_tier_2_rag

LANGFUSE_PUBLIC_KEY=your_langfuse_public_key

LANGFUSE_SECRET_KEY=your_langfuse_secret_key

LANGFUSE_HOST=[https://cloud.langfuse.com](https://cloud.langfuse.com)

3. Run the Ingestion Pipeline (Populate Qdrant)

python src/ingestion/indexer.py

4. Run the FastAPI Backend Locally


uvicorn app.api:app --reload --host 127.0.0.1 --port 8000

5. Run the Streamlit UI Locally (in a separate terminal)


.venv\Scripts\Activate

streamlit run app/streamlit_app.py

🧪 Testing & Evaluation

Execute the automated test suite to verify in-domain retrieval accuracy, response grounding, and out-of-domain hallucination refusal:


pytest evaluation/evaluate_copilot.py -v

🚀 Deployment Guide

Render Blueprint (FastAPI Backend)

In your Render Dashboard, click New + > Blueprint.

Connect your repository (saas-tier2-support-rag).

Provide your environment secrets (GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, etc.) when prompted.

Click Apply to deploy the service.

Streamlit Cloud (Frontend UI)

Go to Streamlit Community Cloud.

Deploy a new app linked to app/streamlit_app.py in your repository.

Under Advanced settings, set the environment variable:

API_URL: https://saas-tier2-support-rag.onrender.com/api/v1/triage

📄 License
This project is open-source and available under the MIT License.
