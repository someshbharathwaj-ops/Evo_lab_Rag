import fitz
from typing import List, Dict


def clean_text(text: str) -> str:
    """Remove extra whitespace and normalize text"""
    clean_text = text.replace('\n', ' ')
    clean_text = clean_text.replace('\r', ' ')
    clean_text = clean_text.replace('\t', ' ')
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
    doc = fitz.open(file_path)
    
    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        text = page.get_text()
        cleaned_text = clean_text(text)
        
        if cleaned_text.strip():
            documents.append({
                "text": cleaned_text,
                "metadata": {
                    "source": file_path,
                    "page": page_number + 1  # Use 1-indexed page numbers
                }
            })
    
    doc.close()
    return documents