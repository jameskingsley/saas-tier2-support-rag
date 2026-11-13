import os
from datasets import load_dataset
import pandas as pd

def fetch_support_tickets():
    """Fetches real-world English customer support tickets from Hugging Face."""
    tickets_path = "data/raw/tickets/support_tickets.parquet"
    if os.path.exists(tickets_path):
        print("[INFO] Support tickets already downloaded.")
        return pd.read_parquet(tickets_path)

    print("[INFO] Downloading support tickets from Hugging Face...")
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    df = ds.to_pandas()
    
    # Filter for English tech/support tickets and take a clean subset of 500 rows
    eng_tickets = df[df["language"] == "en"].head(500)
    eng_tickets.to_parquet(tickets_path, index=False)
    print(f"[SUCCESS] Saved {len(eng_tickets)} support tickets to {tickets_path}")
    return eng_tickets

def clone_sample_docs():
    """Creates structured technical API reference docs in data/raw/docs/guides."""
    docs_dir = "data/raw/docs"
    guides_dir = os.path.join(docs_dir, "guides")
    os.makedirs(guides_dir, exist_ok=True)
    
    if len(os.listdir(guides_dir)) > 0:
        print("[INFO] Documentation files already present.")
        return

    print("[INFO] Creating structured technical API reference docs...")
    sample_docs = {
        "webhooks-timeout.md": "# Webhook Timeout & Retry Limits\n\nAsynchronous payload delivery times out after 10 seconds. Configure retry intervals and gateway proxy headers correctly to prevent HTTP 504 gateway timeouts.",
        "api-rate-limits.md": "# API Rate Limiting & Quotas\n\nStandard tier accounts are restricted to 100 requests per minute. Exceeding this limit returns an HTTP 429 Too Many Requests response header with a Retry-After timestamp.",
        "authentication-tokens.md": "# OAuth2 & Bearer Token Authentication\n\nAll API requests must include a valid Bearer token in the Authorization header. Expired tokens return HTTP 401 Unauthorized; refresh tokens every 24 hours."
    }
    
    for filename, content in sample_docs.items():
        file_path = os.path.join(guides_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"[SUCCESS] Created technical documentation guides in {guides_dir}")

if __name__ == "__main__":
    os.makedirs("data/raw/tickets", exist_ok=True)
    os.makedirs("data/raw/docs", exist_ok=True)
    fetch_support_tickets()
    clone_sample_docs()