"""
Build the document index: PDF -> Markdown (Docling) -> chunks -> Chroma.

1. Docling converts each PDF in data/pdfs/ to Markdown. It runs layout
   analysis and table-structure recognition, so tables come out as real
   Markdown tables instead of jumbled text. The Markdown is saved to
   data/markdown/ so you can read exactly what the agent will search.
2. The Markdown is split along its headings. Tables are never split (half a
   table is useless), prose paragraphs are packed into chunks of up to about
   800 characters, and every chunk starts with "[document > section]" so it
   still makes sense when it is retrieved on its own.
3. The chunks are embedded locally with all-MiniLM-L6-v2 and stored in a
   persistent Chroma collection in data/chroma_db/, using cosine distance.

How to run
----------
python ingest.py

Run it once before agent.py or app.py, and again whenever the PDFs change.
The first run downloads Docling's layout models and the embedding model, which
takes a few minutes. Rebuilding replaces the old index, so nothing is duplicated.
"""

import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "data" / "pdfs"
MD_DIR = ROOT / "data" / "markdown"
CHROMA_DIR = ROOT / "data" / "chroma_db"

COLLECTION = "inventory_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 800


def convert_pdfs(pdf_dir=PDF_DIR, md_dir=MD_DIR):
    """Convert every PDF to Markdown with Docling and save it. Returns {file name: characters}."""
    from docling.document_converter import DocumentConverter  # slow to import, so only when needed

    md_dir.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    sizes = {}
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        markdown = converter.convert(str(pdf)).document.export_to_markdown()
        (md_dir / f"{pdf.stem}.md").write_text(markdown, encoding="utf-8")
        sizes[pdf.name] = len(markdown)
    return sizes


def chunk_markdown(text, source, max_chars=MAX_CHARS):
    """Split one Markdown document into section-aware chunks with metadata."""
    chunks, heading, buf = [], "", []

    def emit(body, kind):
        body = body.strip()
        if body:
            label = f"{source} > {heading}" if heading else source
            chunks.append({
                "text": f"[{label}]\n{body}",
                "metadata": {"source": source, "section": heading, "type": kind},
            })

    def flush_text():
        emit("\n\n".join(buf), "text")
        buf.clear()

    for block in re.split(r"\n\s*\n", text):  # blocks are separated by blank lines
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):  # heading: start a new section
            flush_text()
            heading = block.lstrip("# ").strip()
        elif block.startswith("|"):  # table: always its own, whole chunk
            flush_text()
            emit(block, "table")
        else:  # prose: pack paragraphs up to max_chars
            if buf and sum(len(b) for b in buf) + len(block) > max_chars:
                flush_text()
            buf.append(block)
    flush_text()
    return chunks


def load_chunks(md_dir=MD_DIR):
    """Chunk every Markdown file in md_dir."""
    chunks = []
    for md_file in sorted(md_dir.glob("*.md")):
        chunks += chunk_markdown(md_file.read_text(encoding="utf-8"), md_file.stem)
    return chunks


def embedding_function():
    """The local sentence-transformers model Chroma uses to embed chunks and queries."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def build_index(chunks, chroma_dir=CHROMA_DIR):
    """Embed the chunks into a fresh Chroma collection and return it."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch, so nothing is duplicated
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, embedding_function=embedding_function(),
                                          metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection


def open_index(chroma_dir=CHROMA_DIR):
    """Open the collection built by build_index()."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        return client.get_collection(COLLECTION, embedding_function=embedding_function())
    except Exception as exc:
        raise RuntimeError("The document index does not exist yet. Run `python ingest.py` first.") from exc


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("Converting PDFs to Markdown with Docling...")
    for name, size in convert_pdfs().items():
        print(f"  {name} -> {size:,} characters")

    chunks = load_chunks()
    tables = sum(c["metadata"]["type"] == "table" for c in chunks)
    print(f"Split into {len(chunks)} chunks ({tables} tables, {len(chunks) - tables} text)")

    collection = build_index(chunks)
    print(f"Indexed {collection.count()} chunks in {CHROMA_DIR.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
