import json
import os
import uuid
from pathlib import Path
import pandas as pd


def process_markdown_docs(docs_dir: str = "data/raw/docs") -> list[dict]:
    """Parses technical Markdown guides into structured knowledge cards."""
    cards = []
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"[WARNING] Docs directory not found: {docs_dir}")
        return cards

    for md_file in docs_path.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        lines = content.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else md_file.stem
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else content

        card_text = (
            f"DOCUMENT TYPE: Technical Reference Guide\n"
            f"TITLE: {title}\n"
            f"TECHNICAL SPECIFICATION & RESOLUTION:\n{body}"
        )

        cards.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc_{md_file.name}")),
            "source_type": "documentation",
            "title": title,
            "category": "API & System Reference",
            "priority": "reference",
            "text": card_text,
            "metadata": {
                "filename": md_file.name,
                "source_path": str(md_file),
            },
        })

    print(f"[INFO] Processed {len(cards)} documentation cards.")
    return cards


def process_support_tickets(tickets_path: str = "data/raw/tickets/support_tickets.parquet") -> list[dict]:
    """Transforms raw support tickets into standardized Root-Cause Cards."""
    cards = []
    if not os.path.exists(tickets_path):
        print(f"[WARNING] Tickets file not found: {tickets_path}")
        return cards

    df = pd.read_parquet(tickets_path)

    for idx, row in df.iterrows():
        subject = str(row.get("subject", "Support Inquiry")).strip()
        body = str(row.get("body", "")).strip()
        answer = str(row.get("answer", "")).strip()
        queue = str(row.get("queue", "General Support")).strip()
        priority = str(row.get("priority", "medium")).strip()
        ticket_type = str(row.get("type", "Incident")).strip()

        # Skip incomplete tickets missing either the problem description or the resolution
        if not body or body == "None" or not answer or answer == "None":
            continue

        card_text = (
            f"DOCUMENT TYPE: Resolved Tier-2 Support Ticket\n"
            f"ISSUE SUMMARY: {subject}\n"
            f"QUEUE / DOMAIN: {queue} | TYPE: {ticket_type} | PRIORITY: {priority}\n"
            f"REPORTED SYMPTOMS:\n{body}\n\n"
            f"VERIFIED ROOT CAUSE & RESOLUTION:\n{answer}"
        )

        cards.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"ticket_{idx}_{subject[:30]}")),
            "source_type": "ticket",
            "title": subject,
            "category": queue,
            "priority": priority.lower(),
            "text": card_text,
            "metadata": {
                "ticket_index": int(idx),
                "ticket_type": ticket_type,
                "queue": queue,
            },
        })

    print(f"[INFO] Processed {len(cards)} resolved support ticket cards.")
    return cards


def build_knowledge_base():
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/kb_cards.jsonl"

    doc_cards = process_markdown_docs()
    ticket_cards = process_support_tickets()
    all_cards = doc_cards + ticket_cards

    with open(output_path, "w", encoding="utf-8") as f:
        for card in all_cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Saved {len(all_cards)} total Root-Cause Cards to {output_path}")


if __name__ == "__main__":
    build_knowledge_base()