import json
import sys
from pathlib import Path
from langfuse import observe

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.generation.generator import SupportCopilotGenerator

EVAL_DATASET = [
    {
        "query": "Our webhook endpoint is returning 504 gateway timeouts after 10 seconds. How do we fix this?",
        "expect_found": True,
    },
    {
        "query": "Our production client is hitting HTTP 429 Too Many Requests errors during peak traffic. How do we resolve rate limit throttling?",
        "expect_found": True,
    },
    {
        "query": "How should we configure database connection pooling limits to avoid PoolTimeout errors?",
        "expect_found": False,  # Out-of-scope test: copilot should flag missing KB docs instead of hallucinating
    },
]


@observe(name="llm_as_a_judge_eval")
def judge_response(
    groq_client, query: str, retrieved_sources: list[dict], response: str, expect_found: bool
) -> dict:
    """Evaluates groundedness and out-of-domain abstention behavior."""
    context_summary = "\n".join(
        [f"- {doc['title']}: {doc['text'][:400]}" for doc in retrieved_sources]
    )

    if expect_found:
        rubric = """Score from 1 to 5 (5 = best):
1. groundedness: Are the technical claims strictly supported by the retrieved cards without fabrication?
2. actionability: Does it clearly state the root cause, step-by-step fix, and verification steps?
Set "passed" to true if both scores are >= 4."""
    else:
        rubric = """The retrieved cards do NOT cover this topic. Evaluate whether the copilot properly abstained from fabricating an answer:
1. groundedness: 5 if it explicitly noted missing internal documentation and avoided inventing product settings; 1 if it hallucinated unverified fixes.
2. actionability: 5 if the abstention/escalation guidance is clear and helpful.
Set "passed" to true if groundedness >= 4."""

    judge_prompt = f"""You are a strict QA evaluator for a Tier-2 B2B SaaS Support Copilot.

CUSTOMER QUERY:
{query}

RETRIEVED KNOWLEDGE BASE CARDS:
{context_summary}

GENERATED COPILOT RESPONSE:
{response}

{rubric}

Return ONLY valid JSON with this exact schema:
{{"groundedness": int, "actionability": int, "passed": bool, "reason": "short 1-sentence explanation"}}"""

    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)


def run_evaluation():
    print("[INFO] Initializing Support Copilot Generator for LLM-as-a-Judge Evaluation...")
    copilot = SupportCopilotGenerator()

    passed = 0
    total = len(EVAL_DATASET)

    for idx, item in enumerate(EVAL_DATASET, 1):
        query = item["query"]
        expect_found = item["expect_found"]
        print(f"\n[EVAL {idx}/{total}] ({'IN-DOMAIN' if expect_found else 'OUT-OF-DOMAIN'}) Query: '{query}'")

        result = copilot.triage_ticket(query)
        verdict = judge_response(
            groq_client=copilot.groq_client,
            query=query,
            retrieved_sources=result["retrieved_sources"],
            response=result["response"],
            expect_found=expect_found,
        )

        status = "PASSED" if verdict.get("passed") else "FAILED"
        if verdict.get("passed"):
            passed += 1

        print(f"  -> {status} | Groundedness: {verdict.get('groundedness')}/5 | Actionability: {verdict.get('actionability')}/5")
        print(f"  -> Judge Notes: {verdict.get('reason')}")
        print(f"  -> Top Retrieved Card: {result['retrieved_sources'][0]['title']}")

    print("\n==============================")
    print(f"EVALUATION COMPLETE: {passed}/{total} passed ({(passed / total) * 100:.1f}%)")
    print("================================")

    if copilot.langfuse:
        copilot.langfuse.flush()
        print("[INFO] Evaluation traces flushed to Langfuse.")


if __name__ == "__main__":
    run_evaluation()