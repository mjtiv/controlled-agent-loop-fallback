
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
import time
import argparse

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


# 4) The system prompt is your "policy / guardrails"
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
  "baker_status": "SUPPORTED" | "NOT_FOUND" | "CONFLICT" | "ERROR",
  "evidence": [string, ...],
  "reason": string
}

Rules:
- SUPPORTED: the document supports that they are a baker (title and/or duties clearly involve baking).
- CONFLICT: the document states "baker" (or equivalent) BUT the duties contradict (e.g., no baking tasks; different role).
- ERROR: processing failure occurred outside the document analysis (e.g., API or parsing issue).
- NOT_FOUND: insufficient evidence they are a baker (even if food-related).
- evidence must be 1-3 short verbatim snippets from the document that justify the status.
- If name or occupation is not explicitly present, use "" for those fields.
"""


def analyze_file(text: str, label: str = "") -> dict:
    """
    Core LLM call for a single document with error handling.

    Sends the document + system policy to the model and forces a structured
    JSON response. This function acts as the "analysis engine" for the agent.

    Design choices for stability:
    - temperature=0.0      → deterministic, less creative
    - response_format     → enforce valid JSON output
    - system prompt       → defines rules/guardrails
    - strict parsing      → fail if output isn't valid JSON
    - try/except          → prevents pipeline crashes on API failure

    If something goes wrong (API error, JSON parse error, etc.),
    the function returns a structured ERROR record instead of stopping
    the entire batch run.
    """

    for attempt in range(2):  # try up to 2 times
        try:
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

            # ---- Token Usage Visibility -----------------------------------------------
            # Exposing token usage helps with cost awareness and debugging.
            # Reviewers often want to understand the resource footprint of LLM calls.
            usage = resp.usage
            print(
                f"[tokens:{label}] prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} "
                f"total={usage.total_tokens}"
            )

            # Extract JSON string and convert → Python dict
            content = resp.choices[0].message.content
            result = json.loads(content)

            # ---- Minimal Schema Validation ---------------------------------------------
            # Even though we instructed the model to follow a schema, we do NOT blindly
            # trust the output. Models can occasionally omit fields or change structure.
            #
            # This check enforces a contract boundary between:
            #   (A) the LLM output
            #   (B) our application logic
            #
            # If required fields are missing, we raise an exception so the retry logic
            # can attempt recovery. If the second attempt also fails, the outer error
            # handler returns a structured ERROR record instead of crashing the pipeline.
            #
            # This pattern is important for real-world agents:
            #   prompt rules → model output → program validation → safe system state
            #
            required = {"name", "stated_occupation", "baker_status", "evidence", "reason"}
            
            if not required.issubset(result.keys()):
                raise ValueError("Invalid schema from model")

            return result, usage

        except Exception as e:
            # Return a structured error instead of crashing the pipeline.
            # This allows batch processing to continue for other files.
            if attempt == 1:
                return {
                    "name": "",
                    "stated_occupation": "",
                    "baker_status": "ERROR",
                    "evidence": [],
                    "reason": f"Processing failure: {str(e)}"
                }, None
            
            # Otherwise wait briefly and retry
            time.sleep(1)

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

    # ---- CLI Arguments ------------------------------------------------------
    # In demos, hard-coding paths is fine, but adding CLI args turns this script
    # into a reusable tool. This lets someone run the exact same program against
    # different folders/files without editing code.
    #
    # Example:
    #   python run_role_check.py --input data --output out.json
    #
    # Why this matters (signal):
    # - Separates configuration (paths) from logic (processing)
    # - Improves reproducibility and portability
    # - Makes the script feel like a real utility, not a one-off demo
    #
    # Design choice:
    # - We map args back into SAMPLE_DIR / OUT_PATH so the rest of the pipeline
    #   code stays unchanged.
    #
    parser = argparse.ArgumentParser(
        description="Run document role classification using an LLM."
    )

    parser.add_argument(
        "--input",
        default="sample_people",
        help="Folder containing .txt documents (default: sample_people)",
    )

    parser.add_argument(
        "--output",
        default="results.json",
        help="Output JSON file (default: results.json)",
    )

    args = parser.parse_args()

    # Convert CLI strings into Path objects for safe filesystem handling.
    # This overwrites the defaults so the rest of the script uses the CLI values.
    SAMPLE_DIR = Path(args.input)
    OUT_PATH = Path(args.output)


    # 1) Validate input directory exists (fail fast with clear error)
    if not SAMPLE_DIR.exists():
        raise SystemExit(f"Missing folder: {SAMPLE_DIR.resolve()}")

    # 2) Collect input documents (deterministic ordering for reproducibility)
    files = sorted(SAMPLE_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {SAMPLE_DIR.resolve()}")

    # 3) Accumulate structured results for all documents
    results = []
    total_prompt = 0
    total_completion = 0
    total_tokens = 0

    # 4) Process each document (batch loop)
    for fp in files:
        # Read file contents (errors='ignore' avoids crashing on weird characters)
        text = fp.read_text(encoding="utf-8", errors="ignore")

        # Call the LLM "analysis engine" for this document
        result, usage = analyze_file(text, fp.name)

        if usage:
            total_prompt += usage.prompt_tokens
            total_completion += usage.completion_tokens
            total_tokens += usage.total_tokens

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

    print(
        f"\nTOTAL TOKENS: prompt={total_prompt} "
        f"completion={total_completion} total={total_tokens}"
    )

# Standard Python entry point guard:
# - Running:  python run_role_check.py  will execute main()
# - Importing this file elsewhere will NOT auto-run main()
if __name__ == "__main__":
    main()











