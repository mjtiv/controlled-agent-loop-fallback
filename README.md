# 🔄 Controlled Agent Loop — Enhanced Version

This repository demonstrates a minimal **agentic AI pattern** for document classification using an LLM.

The original version focused on the core pipeline:

1. Load documents  
2. Apply rules via prompt  
3. Call the model  
4. Parse structured output  
5. Persist results  

The updated version keeps this architecture intact while adding **production-style quality control and observability features**.

The goal is not to make the system more complex — but to make it **more reliable, debuggable, and transparent**.

To see the original controlled agent loop — including detailed installation instructions, architecture explanation, and baseline implementation — visit:
https://github.com/mjtiv/controlled-agent-loop/blob/main/README.md

---

## 📁 Project Structure

Place your files in the following layout:

    people_demo/
      sample_people/
        01_amelia_hart_true_baker.txt
        02_marcus_lee_false_baker_label.txt
        03_priya_nair_data_analyst.txt
        04_jose_alvarez_chef_trap_overlap.txt
        05_sarah_kim_flight_attendant.txt
      run_role_check.py
      .env

---

## 🔐 Environment Setup

Create a `.env` file in the root directory:

    OPENAI_API_KEY=your_key_here
    OPENAI_MODEL=gpt-4o-mini

Using `python-dotenv` keeps credentials out of source control and makes
the demo clean and portable.

⚠️ Security Note: This repository does not include a .env file.
API keys are sensitive credentials and must never be committed to source control.
Create your own .env locally and ensure it is listed in .gitignore.

---

## ▶️ Running the Demo

Once your `.env` file is configured and dependencies are installed, run:

python run_role_check_fallback_1.5.py

or

python run_role_check_fallback_1.5.py --input sample_people --output results.json

---

# 🧠 What Changed (High Level)



Compared to the original implementation, the updated script adds:

## ✅ Error Handling + Retry Logic

The agent now retries failed API calls once and returns a structured `"ERROR"` state instead of crashing the pipeline.

This demonstrates a critical real-world principle:

> Agent systems must fail safely and continue processing.

---

## ✅ Schema Validation

Even though the model is instructed to return structured JSON, the system now verifies required fields programmatically before accepting results.

This enforces a **contract boundary** between model output and application logic.

---

## ✅ Token Usage Visibility

The script now prints token usage per file and aggregates totals across the run.

This provides:

- cost awareness  
- debugging insight  
- performance visibility  

Observability is a key requirement in production LLM systems.

---

## ✅ CLI Configurability

Input and output paths are now configurable via command-line arguments:

```bash
run_role_check_fallback_1.5.py --input data --output results.json
```

This transforms the script from a one-off demo into a reusable tool.

---

## ✅ Structured Failure States

Instead of throwing exceptions, the system returns structured `"ERROR"` records when processing fails.

This allows downstream analysis to continue without interruption.

---

# 🔬 Architectural Philosophy

Importantly, the **core agent loop has not changed**.

The system still demonstrates the same fundamental pattern:

```
Document → Rules → Model → Structured Output → Persistence
```

The improvements focus on:

- reliability  
- validation  
- observability  
- reproducibility  

These are the elements that distinguish experimental AI code from deployable systems.

---

# 📁 Version Comparison

## Original Version

- Static paths  
- Single model call  
- No retry  
- No schema verification  
- No token visibility  
- Minimal error handling  

## Enhanced Version

- CLI-driven configuration  
- Retry + safe failure state  
- Schema validation layer  
- Token accounting + totals  
- Improved debugging output  
- More robust pipeline behavior  

See:

- **Original:** `run_role_check_1.0`  
- **Updated:** `run_role_check_fallback_1.5`  

---

# 🚀 Why This Matters

Many “agentic AI” examples focus on adding more steps or complexity.

This repository instead demonstrates a more important insight:

> Real agent systems evolve through reliability and control — not just additional reasoning loops.

The updated version shows how a simple agent can be hardened with:

- guardrails  
- validation  
- observability  
- safe execution patterns  

---

# 🔮 Future Directions (Optional)

Potential extensions include:

- multi-step reasoning loops for ambiguous classifications  
- dynamic rule injection  
- checkpoint/resume processing  
- parallel execution with centralized persistence  

These are intentionally not implemented yet to preserve clarity.

---

# 🧪 Educational Purpose

This project is designed to help engineers understand that:

> Agentic AI systems are often just deterministic pipelines  
> with controlled LLM boundaries

—not magic autonomous entities.

---

# 👍 Summary

The updated version does not change **what** the agent does.

It improves **how safely and transparently it does it**.

That difference is what moves AI code toward production readiness.

---

## 🏗️ Production Insight

Many real-world AI systems are not autonomous agents.

They are controlled pipelines with:

- deterministic orchestration
- bounded model interaction
- validation layers
- observable execution

Understanding this distinction is critical for deploying reliable AI software.

---

## ⚖️ Disclaimer

This repository was created for educational and demonstration purposes only.

Created by **M. Joseph Tomlinson IV**  
Contact: mjtiv@udel.edu  

Feel free to use, modify, adapt, and build upon this project as you see fit.
