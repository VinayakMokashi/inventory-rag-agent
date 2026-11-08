"""
Chat interface for the inventory planning agent (Gradio).

Each answer ends with a collapsible "Tools used" list, so you can see how the
agent reached it: which documents it searched, which SQL it ran and which
numbers went into the calculator.

The chat has short-term memory: the last 10 messages are sent with each new
question, so follow-ups such as "and for PK-3001?" work. The tool lists are
removed from that history, because the model only needs the earlier answers.

How to run
----------
1. Build the index once:  python ingest.py
2. Run:                   python app.py
3. Open http://127.0.0.1:7860 in your browser.

The app runs locally only. Gradio's share=True would create a public link that
anyone could use with your API key, so it is left off.
"""

import json

import gradio as gr

from agent import SAMPLE_QUESTIONS, InventoryAgent

HISTORY_MESSAGES = 10  # how many earlier messages the agent sees
TRACE_MARKER = "\n\n<details><summary>Tools used</summary>"

agent = InventoryAgent()


def message_text(content):
    """The text of a chat message. Gradio 6 sends content as a list of blocks, older versions as a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content
                       if isinstance(block, dict) and block.get("type") == "text")
    return ""  # files and components carry no text for the model


def to_history(chat_history):
    """Convert Gradio's chat history into plain user/assistant messages for the model."""
    past = []
    for h in chat_history or []:
        if isinstance(h, dict):
            role, text = h.get("role"), message_text(h.get("content"))
            if role in ("user", "assistant") and text:
                past.append({"role": role, "content": text.split(TRACE_MARKER)[0]})
        else:  # very old Gradio versions: [user, assistant] pairs
            user, assistant = h
            if user:
                past.append({"role": "user", "content": user})
            if assistant:
                past.append({"role": "assistant", "content": assistant.split(TRACE_MARKER)[0]})
    return past[-HISTORY_MESSAGES:]


def chat(message, chat_history):
    trace = []
    answer = agent.ask(message, to_history(chat_history),
                       on_tool=lambda name, args: trace.append(f"`{name}` {json.dumps(args)}"))
    if trace:
        answer += TRACE_MARKER + "\n\n" + "\n\n".join(trace) + "\n\n</details>"
    return answer


demo = gr.ChatInterface(
    chat,
    title="Inventory Planning Assistant",
    description="Ask about stock levels, reorder points, suppliers, policies and the Q4 demand outlook.",
    examples=SAMPLE_QUESTIONS + [
        "How many days of cover do we have for each A-class item?",
        "What is the EOQ for PK-3001 using our policy parameters?",
    ],
)

if __name__ == "__main__":
    demo.launch()
