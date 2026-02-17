
"""
run_role_check.py

Purpose:
- Load a folder of .txt "documents" (people profiles).
- For each file, call an LLM via OpenAI API.
- Ask the model to classify whether the person is a BAKER based ONLY on the text.
- Force a strict JSON response with:
  - name
  - stated occupation
  - baker_status (SUPPORTED / NOT_FOUND / CONFLICT)
  - evidence snippets
  - reason
- Print results and save to results.json

Why this matters:
- This is the simplest possible "agentic" pattern:
  1) ingest text
  2) apply rules + schema
  3) call model
  4) parse structured output
  5) persist result
"""


# ---- Dependencies: standard libs, config loading, and OpenAI API client ----
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 1) Load environment variables from a local .env file in this directory
#    .env should contain:
#      OPENAI_API_KEY=...
#      OPENAI_MODEL=gpt-4o-mini
load_dotenv()


# 2) Read config from environment variables (with sane defaults)
# ---- Configuration ---------------------------------------------------------
# Load runtime configuration from environment variables.
# Read runtime settings from environment variables.
# This keeps secrets (API keys) and config outside the codebase.
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")

# Fail fast if credentials are missing.
# It's better to stop immediately than make a confusing API call later.
if not API_KEY:
    raise SystemExit(
        "Missing OPENAI_API_KEY. Put it in a .env file or set it in your environment."
    )


# 3) Create an API client
# ---- OpenAI Client ----------------------------------------------------------
# Initialize the OpenAI API client.
# This object handles authentication and sends all requests to the LLM
# (chat completions, embeddings, etc.).
# Think of this as the "brain connection" for the agent.
client = OpenAI(api_key=API_KEY)


# 4) Set where files live and where output will be written
# ---- Input / Output Paths ---------------------------------------------------
# Define where the program reads documents from and where results are saved.
# Keeping paths configurable makes the script predictable and easy to reuse.
SAMPLE_DIR = Path("sample_people")
OUT_PATH = Path("results.json")


# 5) The system prompt is your "policy / guardrails"
#    NOTE: This is where agent behavior is controlled.
#    - We explicitly forbid guessing.
#    - We define what counts as SUPPORTED / CONFLICT / NOT_FOUND.
#    - We require evidence snippets from the document.
#    - We force strict JSON output.
SYSTEM_PROMPT = """You are a strict document-based classifier.
Use ONLY the text provided. Do not assume facts not stated.
Your task: determine whether the person is a BAKER, based on evidence in the text.

Return JSON ONLY matching the schema:
{
  "name": string,
  "stated_occupation": string,
  "baker_status": "SUPPORTED" | "NOT_FOUND" | "CONFLICT",
  "evidence": [string, ...],
  "reason": string
}

Rules:
- SUPPORTED: the document supports that they are a baker (title and/or duties clearly involve baking).
- CONFLICT: the document states "baker" (or equivalent) BUT the duties contradict (e.g., no baking tasks; different role).
- NOT_FOUND: insufficient evidence they are a baker (even if food-related).
- evidence must be 1-3 short verbatim snippets from the document that justify the status.
- If name or occupation is not explicitly present, use "" for those fields.
"""


def analyze_file(text: str) -> dict:
    """
    Core LLM call for a single document.

    Sends the document + system policy to the model and forces a structured
    JSON response. This function acts as the "analysis engine" for the agent.

    Design choices for stability:
    - temperature=0.0      → deterministic, less creative
    - response_format     → enforce valid JSON output
    - system prompt       → defines rules/guardrails
    - strict parsing      → fail if output isn't valid JSON
    """

    resp = client.chat.completions.create(
        model=MODEL,

        # Lower temperature reduces randomness and hallucinations.
        # Good for classification/extraction tasks (bad for creative writing).
        temperature=0.0,

        # Forces the model to output valid JSON only.
        # Prevents free-form text that would break parsing.
        response_format={"type": "json_object"},

        messages=[
            # System message = policy/guardrails (highest priority instructions)
            {"role": "system", "content": SYSTEM_PROMPT},

            # User message = actual document data to analyze
            {"role": "user", "content": f"DOCUMENT:\n{text}"},
        ],
    )

    # Extract JSON string and convert → Python dict
    content = resp.choices[0].message.content
    return json.loads(content)


# ---- Main: Orchestrates the batch analysis pipeline ------------------------
def main():
    """
    Pipeline driver:
    - locate input .txt documents
    - iterate through each file
    - run LLM-based analysis
    - print per-file results
    - write aggregate JSON output to disk
    """

    # 1) Validate input directory exists (fail fast with clear error)
    if not SAMPLE_DIR.exists():
        raise SystemExit(f"Missing folder: {SAMPLE_DIR.resolve()}")

    # 2) Collect input documents (deterministic ordering for reproducibility)
    files = sorted(SAMPLE_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {SAMPLE_DIR.resolve()}")

    # 3) Accumulate structured results for all documents
    results = []

    # 4) Process each document (batch loop)
    for fp in files:
        # Read file contents (errors='ignore' avoids crashing on weird characters)
        text = fp.read_text(encoding="utf-8", errors="ignore")

        # Call the LLM "analysis engine" for this document
        result = analyze_file(text)

        # Add provenance for traceability (helps debugging / audits)
        result["_file"] = fp.name

        # Store result for final output
        results.append(result)

        # Print a human-readable view to the terminal for quick inspection
        print(f"\n=== {fp.name} ===")
        print(json.dumps(result, indent=2))

    # 5) Persist results as one JSON file (useful for demos + downstream steps)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_PATH.resolve()}")


# Standard Python entry point guard:
# - Running:  python run_role_check.py  will execute main()
# - Importing this file elsewhere will NOT auto-run main()
if __name__ == "__main__":
    main()











