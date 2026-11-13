import sys
from pathlib import Path
from groq import Groq
from langfuse import observe, get_client

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import settings
from src.retrieval.hybrid_retriever import HybridRetriever


class SupportCopilotGenerator:
    def __init__(self):
        print("[INFO] Initializing Support Copilot Generator...")
        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.retriever = HybridRetriever()

        # Initialize Langfuse client if keys are provided
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            print("[INFO] Initializing Langfuse Observability...")
            self.langfuse = get_client()
        else:
            self.langfuse = None

    @observe(name="tier2_support_triage")
    def triage_ticket(self, user_query: str, limit: int = 4) -> dict:
        """Runs hybrid retrieval, formats context, generates LLM triage response, and traces via Langfuse."""
        # Retrieve relevant root-cause cards
        retrieved_docs = self.retriever.search(user_query, limit=limit)

        # Build context block from retrieved cards
        context_blocks = []
        for idx, doc in enumerate(retrieved_docs, 1):
            context_blocks.append(
                f"--- SOURCE CARD {idx} [{doc['source_type'].upper()}] ---\n"
                f"Title: {doc['title']}\n"
                f"Category: {doc['category']} | Priority: {doc['priority']}\n"
                f"{doc['text']}\n"
            )
        context_str = "\n".join(context_blocks)

        # Construct Tier-2 Enterprise Support Prompt (Strict Grounding + Abstention Guardrail)
        system_prompt = (
            "You are an expert Tier-2 B2B SaaS Support Triage Copilot. Your job is to analyze "
            "incoming customer issues and provide precise, actionable troubleshooting steps using "
            "ONLY the provided Knowledge Base Root-Cause Cards and technical documentation.\n\n"
            "STRICT GROUNDING RULES:\n"
            "1. Cite specific configuration values, limits, and troubleshooting steps directly from the retrieved cards.\n"
            "2. Do NOT use outside general knowledge to invent product configurations, parameters, or workarounds.\n"
            "3. If the retrieved Knowledge Base cards do NOT contain information relevant to the user's issue, "
            "explicitly state: \"No matching internal documentation was found for this issue,\" explain briefly what "
            "topic is missing from the retrieved cards, and recommend escalating to Tier-3 Engineering.\n\n"
            "When matching documentation IS present, structure your response into clear sections:\n"
            "1. **Identified Root Cause**: What is causing the symptom based on the retrieved cards?\n"
            "2. **Immediate Mitigation / Fix**: Step-by-step resolution instructions for the support engineer.\n"
            "3. **Verification Steps**: How to verify the fix resolved the issue.\n\n"
            "Be concise, technical, and professional."
        )

        user_prompt = (
            f"INCOMING TICKET / QUERY:\n{user_query}\n\n"
            f"RETRIEVED KNOWLEDGE BASE CONTEXT:\n{context_str}\n\n"
            "Provide the Tier-2 Support Triage response:"
        )

        # Call Groq LLM Inference
        chat_completion = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        response_text = chat_completion.choices[0].message.content
        usage = chat_completion.usage

        return {
            "query": user_query,
            "response": response_text,
            "retrieved_sources": retrieved_docs,
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }


if __name__ == "__main__":
    copilot = SupportCopilotGenerator()
    test_query = "Our webhook endpoint is returning 504 gateway timeouts after 10 seconds. How do we fix this?"
    print(f"\n[TEST] Running generation test for query: '{test_query}'\n")
    
    result = copilot.triage_ticket(test_query)
    print("=== COPILOT TRIAGE RESPONSE ===")
    print(result["response"])
    print(f"\n[INFO] Token Usage: {result['token_usage']}")
    if copilot.langfuse:
        copilot.langfuse.flush()
        print("[INFO] Traces flushed to Langfuse.")