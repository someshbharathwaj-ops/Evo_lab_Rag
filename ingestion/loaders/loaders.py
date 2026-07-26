import fitz
from pathlib import Path
from typing import List, Dict


def clean_text(text: str) -> str:
    """Remove extra whitespace and normalize text"""
    clean_text = text.replace('\n', ' ')
    clean_text = clean_text.replace('\r', ' ')
    clean_text = clean_text.replace('\t', ' ')
    clean_text = clean_text.translate({
        codepoint: None
        for codepoint in range(32)
        if codepoint not in {9, 10, 13}
    })
    clean_text = clean_text.replace('\x7f', '')
    # Remove multiple spaces
    clean_text = ' '.join(clean_text.split())
    return clean_text


def load_pdf(file_path: str) -> List[Dict]:
    """
    Load PDF and extract text from each page.
    
    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    documents = []
    source = str(Path(file_path).resolve())
    try:
        with fitz.open(file_path) as doc:
            for page_number in range(doc.page_count):
                page = doc.load_page(page_number)
                cleaned_text = clean_text(page.get_text())

                if cleaned_text.strip():
                    documents.append({
                        "text": cleaned_text,
                        "metadata": {
                            "source": source,
                            "page": page_number + 1,
                        },
                    })
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDF '{file_path}': {exc}") from exc
    return documents
