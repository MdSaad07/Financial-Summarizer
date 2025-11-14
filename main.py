#!/usr/bin/env python3
"""
Financial Summarizer - Main Application
This script automates the entire workflow:
1. Scrape PDFs from URLs in urls.txt
2. Process all downloaded PDFs
3. Launch Streamlit chat interface
"""

import subprocess
import sys
import os
import time
import glob

# Paths
BASE_DIR = "/Users/msf/Desktop/Projects/Financial Summarizer"
VENV_PYTHON = os.path.join(BASE_DIR, ".venv/bin/python")
PDF_FOLDER = os.path.join(BASE_DIR, "Files")
SCRAPER_SCRIPT = os.path.join(BASE_DIR, "pdfscrapper.py")
CHAT_SCRIPT = os.path.join(BASE_DIR, "chatpdf.py")

def print_header(message):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {message}")
    print("="*60 + "\n")

def count_pdfs():
    """Count PDF files in the Files folder"""
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    pdf_files = [f for f in pdf_files if not f.endswith('.crdownload')]
    return len(pdf_files)

def run_pdf_scraper():
    """Run the PDF scraper to download files"""
    print_header("STEP 1: Scraping PDFs from URLs")
    
    try:
        result = subprocess.run(
            [VENV_PYTHON, SCRAPER_SCRIPT],
            cwd=BASE_DIR,
            check=True
        )
        
        pdf_count = count_pdfs()
        print(f"\n✅ PDF scraping completed! Found {pdf_count} PDF files.\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during PDF scraping: {e}\n")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  PDF scraping interrupted by user.\n")
        return False

def process_pdfs():
    """Process PDFs and create vector store"""
    print_header("STEP 2: Processing PDFs")
    
    # Import necessary modules
    sys.path.insert(0, BASE_DIR)
    from PyPDF2 import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    
    try:
        # Load all PDFs
        pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        pdf_files = [f for f in pdf_files if not f.endswith('.crdownload')]
        
        if not pdf_files:
            print("⚠️  No PDF files found to process!")
            return False
        
        print(f"📄 Found {len(pdf_files)} PDF files to process...")
        
        # Extract text from PDFs
        print("📖 Extracting text from PDFs...")
        text = ""
        for pdf_path in pdf_files:
            try:
                pdf_reader = PdfReader(pdf_path)
                for page in pdf_reader.pages:
                    text += page.extract_text()
                print(f"  ✓ Processed: {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"  ✗ Error reading {os.path.basename(pdf_path)}: {e}")
        
        # Create text chunks
        print("\n✂️  Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        print(f"  ✓ Created {len(chunks)} text chunks")
        
        # Create vector store
        print("\n🔢 Creating embeddings and vector store...")
        embeddings = OllamaEmbeddings(model="all-minilm")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        # Save vector store
        faiss_path = os.path.join(BASE_DIR, "faiss_index")
        vector_store.save_local(faiss_path)
        print(f"  ✓ Vector store saved to: {faiss_path}")
        
        print("\n✅ PDF processing completed successfully!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during PDF processing: {e}\n")
        return False

def launch_streamlit():
    """Launch the Streamlit chat interface"""
    print_header("STEP 3: Launching Chat Interface")
    
    print("🚀 Starting Streamlit application...")
    print("📱 The chat interface will open in your browser shortly...\n")
    
    try:
        subprocess.run(
            [VENV_PYTHON, "-m", "streamlit", "run", CHAT_SCRIPT],
            cwd=BASE_DIR,
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Application closed by user. Goodbye!\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching Streamlit: {e}\n")

def main():
    """Main execution flow"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "     📊 FINANCIAL SUMMARIZER - AI-Powered PDF Chat      ".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Check if PDFs already exist
    existing_pdfs = count_pdfs()
    if existing_pdfs > 0:
        print(f"\n📁 Found {existing_pdfs} existing PDF(s) in Files folder.")
        response = input("   Do you want to re-download PDFs? (y/n): ").lower().strip()
        
        if response == 'y':
            # Step 1: Scrape PDFs
            if not run_pdf_scraper():
                print("\n⚠️  Continuing with existing PDFs...\n")
        else:
            print("\n✓ Skipping PDF download. Using existing files.\n")
    else:
        print("\n📥 No existing PDFs found.")
        response = input("   Do you want to download PDFs now? (y/n): ").lower().strip()
        
        if response == 'y':
            # Step 1: Scrape PDFs
            if not run_pdf_scraper():
                print("\n⚠️  No PDFs downloaded. You can select PDFs manually in the app.\n")
        else:
            print("\n✓ Skipping PDF download. You can add PDFs to the Files folder manually.\n")
    
    # Skip automatic processing - let user select in Streamlit
    print_header("Launching Chat Interface")
    print("💡 You can select which PDFs to process in the web interface.\n")
    
    # Launch Streamlit
    launch_streamlit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
