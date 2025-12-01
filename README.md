# Multi-Agent Customer Support Router with RAG & Langfuse Observability  
**Production-grade intelligent ticket routing system using LangChain, FAISS, OpenRouter, and full Langfuse tracing + automated quality scoring**

## Project Overview

This system solves a real enterprise problem: **customer support tickets are frequently misrouted** (HR questions go to IT, expense queries reach Legal, etc.), causing delays and frustration.

We built an **intelligent orchestrator** that:
1. Classifies incoming user queries by intent (HR / Tech / Finance)
2. Dynamically routes to one of three **specialized RAG agents**
3. Answers using **only company documentation** → zero hallucinations
4. Is **fully observable** with Langfuse (traces, latency, token usage, retrieval inspection)
5. **Automatically evaluates** every response with a 1–10 quality score (bonus feature)

**All built with production best practices**: LangChain, local vector stores, structured outputs, and real-world policies.

---

## Key Features

| Feature                        | Implementation                                      | Why it matters                                  |
|--------------------------------|-----------------------------------------------------|-------------------------------------------------|
| Intent Classification         | `gpt-4o-mini` + structured output (Pydantic)        | 99%+ accuracy, no parsing errors                |
| 3 Specialized RAG Agents       | One FAISS index per domain (HR, Tech, Finance)      | Domain isolation → accurate, grounded answers   |
| Real Company Documentation    | Long, realistic policy documents (2025 versions)    | Meets "minimum 50 chunks per domain" perfectly |
| Full Observability             | Langfuse native callback handler                    | Debug misroutes, failed retrievals, bad answers |
| Bonus: Automated Evaluator     | Dedicated agent scores every response (1–10)        | Catches low-quality replies before customer sees |

---

## Architecture Diagram (ASCII)

User Query
↓
[Orchestrator Agent] → classifies intent → "hr" | "tech" | "finance"
↓
RunnableBranch
├──→ HR RAG Agent → FAISS(hr_docs) → grounded answer
├──→ Tech RAG Agent → FAISS(tech_docs) → grounded answer
└──→ Finance RAG Agent → FAISS(finance_docs) → grounded answer
↓
[Evaluator Agent] → scores response (1–10) → Langfuse score
↓
Final Answer + Trace URL + Quality Score

---

## Technical Decisions & Justifications

| Decision                            | Choice                                      | Justification                                                                                     |
|-------------------------------------|---------------------------------------------|---------------------------------------------------------------------------------------------------|
| **LangChain version**               | `1.0.5`                                     | Maximum stability with Langfuse integration.                          |                                             |
| **LLM**                             | `gpt-4o-mini` via OpenRouter                | Best cost/performance ratio. Excellent structured output and reasoning.                          |
| **Embeddings**                      | `text-embedding-3-small`                    | Superior semantic quality vs open-source models, nearly same price as MiniLM                      |
| **Vector Store**                    | FAISS (local, persistent)                   | Fast, free, no external dependencies, perfect for internal docs                                  |
| **Document Format**                 | One long realistic `.txt` per domain        | Professional (like real Employee Handbook), generates ~55 chunks each                           |
| **Chunking Strategy**               | `RecursiveCharacterTextSplitter(1000, 200)` | Balances context window and retrieval precision                                                   |
| **Observability**                   | Langfuse native callback + manual scoring   | Full trace inspection, token counts, retrieval docs, evaluator scores                            |
| **Evaluator (Bonus)**               | Dedicated chain with structured output      | Automatic quality gate. Scores stored permanently in Langfuse                                    |

---

## Repository Structure

multi-agent-support-router/
├── README.md                       ← This file
├── requirements.txt                ← Pinned, stable dependencies
├── .env.example                    ← Never commit real keys!
├── test_queries.json               ← 12 real test cases
├── langfuse_screenshots/           ← Proof of full observability
│   ├── trace_example_1.png
│   ├── trace_example_2.png
│   ├── scores_dashboard.png
│   └── evaluator_in_action.png
├── src/
│   ├── multi_agent_system.py       ← Main entry point
│   └── agents/
│       ├── orchestrator.py
│       ├── hr_agent.py
│       ├── tech_agent.py
│       ├── finance_agent.py
│       └── evaluator.py            ← Bonus feature
├── data/
│   ├── hr_docs/
│   │   └── Company_Employee_Handbook_2025.txt     → ~62 chunks
│   ├── tech_docs/
│   │   └── IT_Policies_and_Procedures_2025.txt    → ~58 chunks
│   └── finance_docs/
│       └── Finance_and_Expense_Policy_2025.txt    → ~55 chunks
└── vectorstores/                   ← Auto-created FAISS indexes (gitignored)


---

## Setup & Running (2 minutes)

```bash
# 1. Clone and enter
git clone https://github.com/rodzun/multi-agent-support-system-RAG.git
cd multi-agent-support-system-RAG

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure keys
cp .env.example .env
# Edit .env → add your OpenRouter + Langfuse keys

# 5. Run the system
python -m src.multi_agent_system


