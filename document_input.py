"""
document_input.py

Extracts text from uploaded documents (.txt, .pdf) so hidden or embedded
instructions can be analyzed by the same detection pipeline as typed
text - this is a direct, real test of the "Indirect Injection" threat
category (malicious instructions hidden inside a document rather than
the direct user message).
"""


def extract_text(uploaded_file) -> str:
    """Takes a Streamlit UploadedFile object, returns its text content.
    Raises ValueError for unsupported file types."""
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: {name}. Use .txt or .pdf.")
