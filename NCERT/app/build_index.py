import os
import re
import subprocess
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

CHAPTER_NAMES = {
    1: "Chemical Reactions and Equations",
    2: "Acids, Bases and Salts",
    3: "Metals and Non-metals",
    4: "Carbon and its Compounds",
    5: "Life Processes",
    6: "Control and Coordination",
    7: "How do Organisms Reproduce?",
    8: "Heredity",
    9: "Light – Reflection and Refraction",
    10: "The Human Eye and the Colourful World",
    11: "Electricity",
    12: "Magnetic Effects of Electric Current",
    13: "Our Environment"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_chunks")

def extract_text_from_pdf(pdf_path):
    cmd = ["pdftotext", pdf_path, "-"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    text = res.stdout
    # Clean up common non-printable / header-footer noise
    text = re.sub(r'\x0c', '\n\n', text) # form feeds
    text = re.sub(r'Rationalised\s+2023-24', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def build_vector_store():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )

    all_docs = []
    print("Extracting and chunking all 13 NCERT chapters...")
    for ch_num, ch_name in sorted(CHAPTER_NAMES.items()):
        pdf_path = os.path.join(PDF_DIR, f"chapter_{ch_num:02d}.pdf")
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping.")
            continue

        raw_text = extract_text_from_pdf(pdf_path)
        chunks = text_splitter.split_text(raw_text)
        print(f"Chapter {ch_num} ({ch_name}): {len(chunks)} chunks created.")

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chapter_number": ch_num,
                    "chapter_name": ch_name,
                    "chunk_id": f"ch{ch_num}_{i}"
                }
            )
            all_docs.append(doc)

    print(f"Total documents to embed: {len(all_docs)}")
    
    # Save processed chunks for inspection
    with open(os.path.join(PROCESSED_DIR, "chunks_summary.json"), "w", encoding="utf-8") as f:
        json.dump([{"content": d.page_content, "metadata": d.metadata} for d in all_docs[:10]], f, indent=2)

    # Use standard all-MiniLM-L6-v2 embedding model (fast, light, high accuracy)
    print("Initializing HuggingFaceEmbeddings ('sentence-transformers/all-MiniLM-L6-v2')...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS index...")
    db = FAISS.from_documents(all_docs, embeddings)
    
    db.save_local(INDEX_DIR)
    print(f"FAISS index successfully saved to {INDEX_DIR}!")

if __name__ == "__main__":
    build_vector_store()
