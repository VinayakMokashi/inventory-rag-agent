# Inventory RAG Agent

An inventory planning assistant that answers questions by combining three sources: **policy documents** (PDFs, searched with retrieval), **live data** (a SQLite database, queried with SQL the model writes), and a **calculator** for the policy's safety-stock formula. The model decides which tools a question needs, calls them, and explains its answer with the numbers and documents it used.

Built with [Docling](https://github.com/docling-project/docling) (PDF to Markdown), [Chroma](https://www.trychroma.com/) (vector search), [Groq](https://groq.com/)-hosted `openai/gpt-oss-120b` (tool calling) and [Gradio](https://www.gradio.app/) (chat interface). All the data belongs to a fictional distributor, Acme Distribution.

```
 PDFs ──Docling──► Markdown ──chunk──► Chroma ──── search_documents ──┐
                                                                      │
 SQLite (stock, sales, purchase orders) ────────── run_sql ───────────┼──► agent (gpt-oss-120b on Groq) ──► answer
                                                                      │
 policy formula ─────────────────────────── calculate_reorder_point ──┘
```

A question like *"What is the safety stock for EL-1001 at the service level required by its ABC class?"* needs all three: the service level for the SKU's class is in the policy PDF, the lead time and 13 weeks of sales are in the database, and the formula is the calculator's job.

## Results on the sample questions

From the run saved in [`walkthrough.ipynb`](walkthrough.ipynb), with every answer checked by hand against the documents and the database:

| Question | Tool calls | Result |
| --- | --- | --- |
| Which SKUs are below their reorder point once on-order stock is counted? | 1 SQL query | ✅ EL-1001, PK-3001 and CP-4001, the same three a direct query finds |
| What is the safety stock for EL-1001 at the service level required by its ABC class? | 3 SQL queries, 2 document searches | ✅ 157 units: class A → 99% → z = 2.33 (policy); daily σ = 30.74 / √7 = 11.62 (sales history); lead time 28 days + 20% review buffer = 33.6 days (supplier agreements) |
| How would that change at a 95% service level instead? *(follow-up)* | 1 calculation | ✅ 112 units, reusing the earlier inputs from the chat history |
| Given the November promotion, which suppliers are risky and what can we do about it? | 10, then the step-limit fallback | ⚠️ Names the right risks with sources, but suggests air freight that Delta Electronics does not offer, and drops the 99% target and buffer it used two questions earlier |

What testing found:

- **A first run got the safety stock confidently wrong.** It answered 54 units, with the right formula and a tidy table of sources, but it divided the weekly standard deviation by 7 instead of √7 and never read the 20% buffer rule for a supplier on review. Both rules are now in the system prompt. Nothing in the wrong answer looked wrong; only checking its inputs caught it.
- **Forcing a final answer is harder than it looks.** The broad question first hit the step limit, and the usual fix, `tool_choice="none"` on the last step, was rejected because gpt-oss still tried to call a tool. Its next attempt was cut off by the token limit. The agent now answers from the gathered results in a fresh conversation, under a word limit.
- **Broad planning questions are the weak spot.** They gather more facts than the model then uses consistently, so check the **Tools used** trace before acting on them.

## Repository structure

```
inventory-rag-agent/
├── data/
│   ├── pdfs/                 # inventory policy, supplier agreements, Q4 demand report
│   └── inventory.db          # SQLite: suppliers, products, inventory, sales_history, purchase_orders
├── ingest.py                 # PDFs -> Markdown (Docling) -> chunks -> Chroma index
├── agent.py                  # the three tools, the tool-calling loop and a command-line interface
├── app.py                    # Gradio chat interface with short-term memory
├── walkthrough.ipynb         # every step with its output, from the PDFs to the agent's answers
├── requirements.txt
├── .env.example              # template for your Groq API key
└── LICENSE
```

`ingest.py` writes `data/markdown/` (the converted documents) and `data/chroma_db/` (the index). Both are generated, so they are git-ignored.

## The sample data

| File | Contents |
| --- | --- |
| `inventory_policy.pdf` | ABC classes and their service levels (A 99%, B 95%, C 90%), the safety-stock and reorder-point formulas, EOQ rules, and exceptions such as escalating any SKU with under 7 days of cover |
| `supplier_agreements.pdf` | Lead time, minimum order quantity, payment terms and on-time rate for four suppliers, expedite options, and penalties: Delta Electronics is on performance review, so planners add a 20% lead-time buffer |
| `demand_warehouse_report.pdf` | The Q4 forecast by category, a November promotion on Wireless Sensors and Shipping Boxes, warehouse capacity, and key risks |
| `inventory.db` | 8 products from 4 suppliers, stock on hand and on order with reorder points, 13 weeks of weekly sales from 2026-07-06, and 4 purchase orders |

The database is a snapshot taken on 2026-10-16, so the agent treats that as today's date.

## How it works

### 1. Ingestion (`ingest.py`)

- **PDF to Markdown with Docling.** Docling runs layout analysis and table-structure recognition, so tables come out as Markdown tables instead of jumbled numbers, and headings as Markdown headings.
- **Section-aware chunking.** The Markdown is split at every heading, so a chunk never mixes two sections. Each table becomes one whole chunk, prose is packed into chunks of up to about 800 characters, and every chunk starts with `[document > section]` so it makes sense on its own and can be cited. The three PDFs become 17 chunks (6 tables, 11 text).
- **Indexing.** Chunks are embedded locally with `all-MiniLM-L6-v2` and stored in a persistent Chroma collection with cosine distance. Rebuilding deletes the old collection first, so nothing is duplicated.

### 2. Tools (`agent.py`)

| Tool | What it does |
| --- | --- |
| `search_documents(query, n_results=4)` | Semantic search over the indexed PDFs; returns each chunk with its document and section |
| `run_sql(query)` | Runs one `SELECT` or `WITH` query and returns the columns and up to 100 rows |
| `calculate_reorder_point(avg_daily_demand, demand_std_daily, lead_time_days, service_level=0.95)` | Safety stock `SS = z × σd × √L` and reorder point `ROP = d × L + SS`, with z = 1.28, 1.65 or 2.33 for 90%, 95% or 99% |

`run_sql` has two layers of protection: it refuses anything that does not start with `SELECT` or `WITH`, and it opens the database in read-only mode, so even a write hidden inside a `WITH` query fails with "attempt to write a readonly database". SQLite has no standard-deviation function, so the tool registers `STDEV()` for the demand-variability questions.

### 3. The agent loop (`InventoryAgent.ask`)

1. Send the system prompt, any chat history and the question to the model with the three tool descriptions.
2. Run every tool the model requests and send each result back as a tool message.
3. Repeat until the model answers without requesting a tool, for at most 10 steps.

The system prompt gives the model the database schema, so it can write SQL without exploring, and tells it to look facts up instead of guessing, show its numbers and formula, and cite documents. Two of its rules came from a first run that went wrong (see the results): convert a weekly standard deviation to a daily one by dividing by √7, not 7, and check the supplier agreements before using a lead time.

Four safeguards handle problems seen in practice:

- **Invalid tool arguments.** gpt-oss sometimes writes a formula such as `30.7/(7**0.5)` into a tool call's JSON, and Groq rejects the whole request (`tool_use_failed`). The agent tells the model and lets it try again.
- **Running out of steps.** If the model is still calling tools after 10 steps, the agent asks for an answer from the results gathered so far, in a fresh conversation with those results written out as text. Simply forbidding tools does not work: once a conversation contains tool calls, gpt-oss tries to call another tool even with `tool_choice="none"` or with no tools offered, and Groq rejects the request.
- **Cut-off answers.** gpt-oss's hidden reasoning counts toward the token limit, so a long answer can be cut off. The system prompt asks for answers under 300 words, and if one is still cut off (`finish_reason="length"`), the agent asks again from the gathered results with low reasoning effort.
- **Failing tools.** Bad SQL or a wrong argument comes back to the model as the tool's result, so it can correct itself.

### 4. Chat interface (`app.py`)

A Gradio `ChatInterface` on top of the same agent. Each answer ends with a collapsible **Tools used** list, so you can check every SQL query and calculation behind it. The last 10 messages are passed as history, with the tool lists removed, so follow-up questions work.

Gradio 6 passes chat history as messages whose content is a list of blocks such as `{"type": "text", "text": "..."}`, where older versions used plain strings. `app.py` reads both. Code that keeps only string content silently drops the whole history, and the chat loses its memory without any error.

## Tech stack

- Python 3.10+
- `docling` (PDF to Markdown with layout and table recognition; installs PyTorch)
- `chromadb` with `sentence-transformers` (vector store and local `all-MiniLM-L6-v2` embeddings)
- `groq` (chat completions with tool calling) and `python-dotenv` (loads your API key)
- `gradio` (chat interface)
- `pandas` (tables in the walkthrough notebook)
- Model: `openai/gpt-oss-120b` on Groq

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/VinayakMokashi/inventory-rag-agent.git
   cd inventory-rag-agent
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Docling and sentence-transformers install PyTorch, so this takes a few minutes.

4. **Add your Groq API key.** Copy `.env.example` to `.env` and paste in your key (get a free one at https://console.groq.com/keys):

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

   The scripts load `.env` from the repository root. `.env` is git-ignored and never committed.

## How to run

### 1. Build the index (once)

```bash
python ingest.py
```

```
Converting PDFs to Markdown with Docling...
  demand_warehouse_report.pdf -> 1,692 characters
  inventory_policy.pdf -> 1,977 characters
  supplier_agreements.pdf -> 1,665 characters
Split into 17 chunks (6 tables, 11 text)
Indexed 17 chunks in data/chroma_db/
```

The first run downloads Docling's layout models and the embedding model. Run it again whenever the PDFs change. It needs no API key.

### 2. Ask questions from the command line

```bash
# Three sample questions, printing each tool call as it happens
python agent.py

# Your own question
python agent.py --question "Which SKUs have fewer than 7 days of cover?"
```

### 3. Chat in the browser

```bash
python app.py
```

Open http://127.0.0.1:7860. The example questions under the chat box are a good start. The app runs locally only: Gradio's `share=True` would create a public link that anyone could use with your API key, so it is left off.

### 4. Walkthrough notebook

```bash
pip install jupyter
jupyter notebook walkthrough.ipynb
```

The notebook runs every step, from the sample data to the agent's answers, and is saved with its outputs, so you can read it on GitHub without running it. Running it rebuilds the index and makes about 20 model calls, because each agent question takes several tool rounds.

### Questions to try

- *Which SKUs have fewer than 7 days of cover?* (database, then the policy's escalation rule)
- *What happens if Delta Electronics delivers late?* (supplier agreements and the demand report)
- *Compute the reorder point for the Linear Actuator at the right service level.* (database, policy and calculator)
- *What is the EOQ for PK-3001 using our policy parameters?* (policy formula with database figures)

## Limitations

- **Small, fictional data.** Three short PDFs and 8 SKUs show the approach; a real deployment would need far more documents and checks.
- **The model does the planning arithmetic outside the calculator.** Averages and standard deviations come from SQL, but the model chooses how to combine them, and a wrong choice (like the √7 conversion above) produces a confident, wrong number. A dedicated tool for demand statistics would be safer.
- **Prompt rules are instructions, not checks.** The model can still skip a document it should have read.
- **No evaluation set.** The answers were checked by hand on a few questions; there is no automated test of answer accuracy.
- **Token use.** A broad question can take 10 tool rounds with a growing context, which uses a large share of a free Groq quota.

## License

Released under the [MIT License](LICENSE).
