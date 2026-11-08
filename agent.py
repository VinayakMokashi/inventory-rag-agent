"""
Inventory planning agent: documents + database + calculator (Groq tool calling).

The agent answers planning questions for a fictional distributor, Acme
Distribution, by choosing among three tools:

- search_documents        - semantic search over the policy, supplier and demand
                            PDFs (the Chroma index built by ingest.py)
- run_sql                 - a read-only SELECT on the inventory database
                            (data/inventory.db: stock, sales, purchase orders)
- calculate_reorder_point - safety stock and reorder point with the policy formula

Many questions need all three: the service level comes from the policy PDF,
the demand history from the database, and the arithmetic from the calculator.

The loop: call the model -> run every tool it requests -> send the results
back -> repeat until it answers. If it is still calling tools after 10 steps,
the agent asks for an answer from the results gathered so far.

How to run
----------
1. Copy `.env.example` to `.env` and add your Groq API key.
2. Install dependencies:  pip install -r requirements.txt
3. Build the index once:  python ingest.py
4. Run:                   python agent.py

With no arguments it answers three sample questions and prints each tool call.
Use --question to ask your own, or run app.py for a chat interface.
"""

import argparse
import json
import math
import os
import sqlite3
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import BadRequestError, Groq

from ingest import open_index

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "inventory.db"
MODEL = "openai/gpt-oss-120b"
TODAY = "2026-10-16"  # the sample database is a snapshot taken on this date
MAX_ROWS = 100  # rows returned per SQL query

SCHEMA = """
suppliers(supplier_id, name, lead_time_days, moq, on_time_pct)
products(sku, name, category, abc_class, unit_cost, supplier_id)
inventory(sku, on_hand, on_order, reorder_point)
sales_history(sku, week_start, units_sold)      -- weekly sales, 13 weeks from 2026-07-06
purchase_orders(po_id, sku, qty, order_date, expected_date, status)  -- status: Confirmed / In Transit / Received
STDEV(x) and STDDEV(x) give the sample standard deviation.
"""

SYSTEM_PROMPT = f"""You are an inventory planning assistant for Acme Distribution.
You have three tools:
- search_documents: company policies, supplier agreements and the Q4 demand report (use for rules, formulas, terms, forecasts).
- run_sql: live data in SQLite (use for stock levels, sales, purchase orders). Schema:
{SCHEMA}
- calculate_reorder_point: apply the policy formula.
Guidelines: look up facts with tools instead of guessing; combine documents and data when needed
(e.g. supplier lead time + sales history + policy service level); show key numbers and the formula used;
cite the document name when you rely on a document. Today's date is {TODAY}.
- Sales are recorded weekly. Average daily demand = weekly units / 7, but the standard deviation of daily
  demand = weekly standard deviation / sqrt(7): variability does not scale like the average.
- Before using a supplier's lead time in a calculation, search the supplier agreements for buffers or
  reliability rules that change it.
- When several lookups do not depend on each other, request them in the same step.
- Keep the final answer under 300 words, and write formulas in plain text, not LaTeX."""

# For the last call when the step limit is reached: no tools are mentioned, so the model answers.
FINAL_ANSWER_PROMPT = f"""You are an inventory planning assistant for Acme Distribution. Answer the question from
the information gathered below. Show key numbers and the formula used, cite the document name when you rely on
a document, keep the answer under 300 words, and write formulas in plain text. Today's date is {TODAY}."""
CUT_OFF = "Sorry, my answer was cut off by the token limit. Please try a narrower question."

SAMPLE_QUESTIONS = [
    "Which SKUs are below their reorder point once on-order stock is counted?",
    "What is the safety stock for EL-1001 at the service level required by its ABC class?",
    "Given the November promotion, which suppliers are risky and what can we do about it?",
]

# z-scores for the service levels the policy uses
Z_FACTORS = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}


class SampleStdev:
    """STDEV(x) / STDDEV(x) aggregate for SQLite, which has no built-in standard deviation."""

    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            self.values.append(value)

    def finalize(self):
        return statistics.stdev(self.values) if len(self.values) > 1 else None


def run_sql(query: str):
    """Run one read-only SELECT and return its columns and up to MAX_ROWS rows."""
    if not query.strip().lower().startswith(("select", "with")):
        return {"error": "Only SELECT queries are allowed."}
    try:
        # mode=ro opens the file read-only, so even a query that slipped past the check cannot write
        db = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
        try:
            for name in ("stdev", "stddev"):
                db.create_aggregate(name, 1, SampleStdev)
            cur = db.execute(query)
            return {"columns": [c[0] for c in cur.description], "rows": cur.fetchmany(MAX_ROWS)}
        finally:
            db.close()
    except Exception as e:
        return {"error": str(e)}


def calculate_reorder_point(avg_daily_demand: float, demand_std_daily: float,
                            lead_time_days: float, service_level: float = 0.95):
    """Safety stock SS = z * sigma_d * sqrt(L); reorder point ROP = d * L + SS."""
    z = Z_FACTORS.get(round(service_level, 2), 1.65)
    safety = z * demand_std_daily * math.sqrt(lead_time_days)
    rop = avg_daily_demand * lead_time_days + safety
    return {"z": z, "safety_stock": round(safety, 1), "reorder_point": round(rop, 1)}


TOOLS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Semantic search over inventory policy, supplier agreements and demand/warehouse report PDFs.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "n_results": {"type": "integer", "default": 4}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run a read-only SELECT against the inventory SQLite database.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate_reorder_point",
        "description": "Compute safety stock and reorder point: SS = Z*sigma_d*sqrt(L); ROP = d*L + SS. "
                       "service_level is 0.90, 0.95 or 0.99.",
        "parameters": {"type": "object", "properties": {
            "avg_daily_demand": {"type": "number"}, "demand_std_daily": {"type": "number"},
            "lead_time_days": {"type": "number"}, "service_level": {"type": "number"}},
            "required": ["avg_daily_demand", "demand_std_daily", "lead_time_days"]}}},
]


class InventoryAgent:
    def __init__(self, model=MODEL):
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key.")
        self.client = Groq(max_retries=5)  # reads GROQ_API_KEY; retries ride out free-tier rate limits
        self.model = model
        self.collection = open_index()
        self.tools = {"search_documents": self.search_documents, "run_sql": run_sql,
                      "calculate_reorder_point": calculate_reorder_point}

    def search_documents(self, query: str, n_results: int = 4):
        """Return the chunks closest to the query, with the document and section they came from."""
        res = self.collection.query(query_texts=[query], n_results=n_results)
        return [{"text": d, "source": m["source"], "section": m["section"]}
                for d, m in zip(res["documents"][0], res["metadatas"][0])]

    def ask(self, question, history=None, max_steps=10, on_tool=None):
        """Answer a question, calling tools as needed.

        history: earlier turns as [{"role": "user" | "assistant", "content": str}, ...]
        on_tool: optional callback(name, args), called before each tool runs
        """
        messages = ([{"role": "system", "content": SYSTEM_PROMPT}] + (history or [])
                    + [{"role": "user", "content": question}])
        evidence = []  # every tool call and its result, for the final answer if the steps run out
        for _ in range(max_steps):
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOLS,
                                                           tool_choice="auto", temperature=0.2)
            except BadRequestError as e:
                # gpt-oss sometimes writes a formula into a tool call's JSON arguments, and Groq
                # rejects the call. Tell the model and let it try again.
                if "tool_use_failed" not in str(e):
                    raise
                messages.append({"role": "system", "content": "Your last tool call was rejected because its "
                                 "arguments were not valid JSON. Work out any formula first, then call the "
                                 "tool again with plain numbers."})
                continue

            choice = resp.choices[0]
            msg = choice.message
            if not msg.tool_calls:
                if choice.finish_reason == "length":
                    # gpt-oss's hidden reasoning counts toward the token limit. Ask once more from the
                    # gathered results, with less reasoning.
                    return self.final_answer(question, history, evidence, reasoning_effort="low")
                return msg.content

            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": t.id, "type": "function",
                                             "function": {"name": t.function.name,
                                                          "arguments": t.function.arguments}}
                                            for t in msg.tool_calls]})
            for t in msg.tool_calls:
                try:
                    args = json.loads(t.function.arguments or "{}")
                    if on_tool:
                        on_tool(t.function.name, args)
                    result = self.tools[t.function.name](**args)
                except Exception as e:  # bad arguments or an unknown tool: report it to the model
                    result = {"error": str(e)}
                content = json.dumps(result, default=str)
                messages.append({"role": "tool", "tool_call_id": t.id, "content": content})
                evidence.append(f"{t.function.name}({t.function.arguments}) -> {content}")

        return self.final_answer(question, history, evidence)

    def final_answer(self, question, history, evidence, reasoning_effort="medium"):
        """Answer from the tool results gathered so far, without offering tools.

        Used when the step limit is reached or an answer was cut off. Once a conversation
        contains tool calls, gpt-oss keeps trying to call tools even when tool_choice is "none"
        or no tools are offered, and Groq rejects the request. A fresh conversation with the
        results written out as plain text avoids that.
        """
        notes = "\n".join(f"- {line}" for line in evidence) or "- (no tools were called)"
        messages = ([{"role": "system", "content": FINAL_ANSWER_PROMPT}] + (history or [])
                    + [{"role": "user", "content": f"{question}\n\nInformation gathered so far:\n{notes}\n\n"
                                                   "Answer from this information, and say what is still missing."}])
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.2,
                                                       reasoning_effort=reasoning_effort)
        except BadRequestError:
            return "Sorry, I could not finish within the step limit."
        choice = resp.choices[0]
        return CUT_OFF if choice.finish_reason == "length" else choice.message.content


def main(argv=None):
    # Force UTF-8 stdout so model output with special characters does not crash on Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask the inventory planning agent a question.")
    parser.add_argument("--question", help="question to ask; leave out to run the sample questions")
    args = parser.parse_args(argv)

    agent = InventoryAgent()
    for question in [args.question] if args.question else SAMPLE_QUESTIONS:
        print(f"Q: {question}")
        answer = agent.ask(question, on_tool=lambda name, a: print(f"  tool: {name} {json.dumps(a)}"))
        print(f"\nA: {answer}\n\n{'-' * 70}\n")


if __name__ == "__main__":
    main()
