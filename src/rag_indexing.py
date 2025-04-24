### Indexing part of rag (Converting into embeddings and storing into chromadb)

import os
import json
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from datetime import datetime

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500

def print_progress(message: str):
    """Helper function for consistent progress messages"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 🔄 {message}")

def process_json_files(json_dir: str, text_splitter: RecursiveCharacterTextSplitter) -> List[Dict[str, Any]]:
    """Process forum JSON files into structured documents"""
    documents = []
    print_progress(f"Starting JSON processing for directory: {json_dir}")
    
    for json_file in os.listdir(json_dir):
        if not json_file.endswith('.json'):
            continue
            
        file_path = os.path.join(json_dir, json_file)
        print_progress(f"Processing JSON file: {json_file}")
        
        with open(file_path, 'r') as f:
            try:
                threads = json.load(f)
            except json.JSONDecodeError as e:
                print_progress(f"Error loading {json_file}: {str(e)}")
                continue
            
            print_progress(f"Found {len(threads)} threads in {json_file}")
            
            for thread in threads:
                content = f"Title: {thread['thread_title']}\n"
                content += f"Description: {thread['thread_descp']}\n"
                content += "Comments:\n" + "\n".join(thread['thread_comments'])
                
                chunks = text_splitter.split_text(content)
                documents.extend([
                    {
                        "text": chunk,
                        "metadata": {
                            "source_type": "forum",
                            "dataset": "forums",
                            "file_path": file_path,
                            "chunk_id": i
                        }
                    } for i, chunk in enumerate(chunks)
                ])
                
                print_progress(f"Created {len(chunks)} chunks from thread: {thread['thread_title'][:30]}...")
                
        print_progress(f"Completed processing {json_file} | Total chunks so far: {len(documents)}")
    
    print_progress(f"Finished JSON processing. Total forum chunks: {len(documents)}")
    return documents

def process_pdf_files(pdf_dirs: List[str], text_splitter: RecursiveCharacterTextSplitter) -> List[Dict[str, Any]]:
    """Process PDF files from multiple datasets"""
    documents = []
    
    for dataset_dir in pdf_dirs:
        dataset_name = os.path.basename(dataset_dir)
        print_progress(f"📂 Processing PDF dataset: {dataset_name}")
        
        pdf_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pdf')]
        print_progress(f"Found {len(pdf_files)} PDF files in {dataset_name}")
        
        for pdf_file in pdf_files:
            file_path = os.path.join(dataset_dir, pdf_file)
            print_progress(f"📄 Processing PDF: {pdf_file}")
            
            try:
                reader = PdfReader(file_path)
            except Exception as e:
                print_progress(f"Error reading {pdf_file}: {str(e)}")
                continue
            
            total_pages = len(reader.pages)
            print_progress(f"Found {total_pages} pages in {pdf_file}")
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                except Exception as e:
                    print_progress(f"Error extracting text from page {page_num+1}: {str(e)}")
                    continue
                
                if not text:
                    continue
                
                chunks = text_splitter.split_text(text)
                documents.extend([
                    {
                        "text": chunk,
                        "metadata": {
                            "source_type": "technical_doc",
                            "dataset": dataset_name,
                            "file_path": file_path,
                            "page_number": page_num + 1,
                            "chunk_id": i
                        }
                    } for i, chunk in enumerate(chunks)
                ])
                
                if page_num % 5 == 0:
                    print_progress(f"Page {page_num+1}/{total_pages} | Current chunks: {len(chunks)}")
            
            print_progress(f"Finished {pdf_file} | Total chunks: {len(documents)}")
    
    print_progress(f"✅ PDF processing complete. Total technical chunks: {len(documents)}")
    return documents

def main():
    # Initialize components
    print_progress("Initializing RAG pipeline...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )

    # Process data sources
    print_progress("\n=== Processing Forum Data ===")
    json_docs = process_json_files("dataset/forums", text_splitter)
    
    print_progress("\n=== Processing Technical Documents ===")
    pdf_dirs = [
        "dataset/x1_dataset",
        "dataset/x2_dataset", 
        "dataset/x3_dataset"
    ]
    pdf_docs = process_pdf_files(pdf_dirs, text_splitter)
    
    # Combine documents
    print_progress("\n=== Combining Results ===")
    all_docs = json_docs + pdf_docs
    print_progress(f"Total chunks: {len(all_docs)}")
    print_progress(f"Forum chunks: {len(json_docs)}")
    print_progress(f"Technical chunks: {len(pdf_docs)}")
    
    # Create vector store
    print_progress("\n=== Creating Vector Store ===")
    vector_store = Chroma.from_texts(
        texts=[doc["text"] for doc in all_docs],
        embedding=embeddings,
        metadatas=[doc["metadata"] for doc in all_docs],
        persist_directory=CHROMA_PERSIST_DIR
    )
    vector_store.persist()
    
    print_progress(f"\n✅ Vector store created successfully!")
    print_progress(f"Persisted at: {os.path.abspath(CHROMA_PERSIST_DIR)}")
    print_progress(f"Total documents indexed: {len(all_docs)}")

if __name__ == "__main__":
    print_progress("🚀 Starting RAG Pipeline")
    main()
    print_progress("🏁 Pipeline completed successfully")
