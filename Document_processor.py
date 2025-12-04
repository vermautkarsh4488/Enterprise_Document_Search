"""
Handles all the PDF processing stuff - reading files, OCR for scanned docs, chunking, etc.
"""

import os
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pytesseract
from pdf2image import convert_from_path
import fitz  # this is PyMuPDF, weird naming I know
from config import CHUNK_SIZE, CHUNK_OVERLAP, DOC_CATEGORIES


class DocumentProcessor:
    
    def __init__(self):
        # Setup the text splitter with sensible defaults
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]  # split by paragraphs first, then sentences, etc.
        )

    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Quick check to see if a PDF is scanned (image-based) or has actual text.
        If there's barely any text, it's probably scanned.
        """
        doc = fitz.open(pdf_path)
        text_content = ""
        
        for page in doc:
            text_content += page.get_text()
        doc.close()
        
        # If we got less than 100 characters, it's probably a scanned doc
        return len(text_content.strip()) < 100

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        Use OCR to read text from scanned PDFs.
        This is slower but works for image-based documents.
        """
        print(f"  → Using OCR for: {pdf_path}")
        
        # Convert PDF pages to images first
        images = convert_from_path(pdf_path)
        full_text = ""
        
        for i, image in enumerate(images):
            # Run OCR on each page
            text = pytesseract.image_to_string(image)
            full_text += f"\n--- Page {i+1} ---\n{text}"
            
        return full_text

    def load_pdf(self, pdf_path: str, category: str) -> List[Document]:
        """
        Load a single PDF file. Automatically uses OCR if the PDF is scanned.
        Returns a list of Document objects with metadata.
        """
        try:
            # Check if we need OCR
            if self.is_scanned_pdf(pdf_path):
                # Scanned PDF - use OCR
                text = self.extract_text_with_ocr(pdf_path)
                
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "category": category,
                        "filename": os.path.basename(pdf_path),
                        "extraction_method": "OCR"
                    }
                )
                return [doc]
            
            else:
                # Regular PDF with text - use standard loader (much faster)
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                
                # Add our custom metadata to each page
                for doc in docs:
                    doc.metadata["category"] = category
                    doc.metadata["filename"] = os.path.basename(pdf_path)
                    doc.metadata["extraction_method"] = "text"
                    
                return docs
                
        except Exception as e:
            # Don't crash if one file fails, just skip it
            print(f"  ⚠️ Error loading {pdf_path}: {e}")
            return []

    def process_documents(self, base_path: str = "./documents") -> List[Document]:
        """
        Go through all the document folders and process every PDF.
        Returns chunked documents ready for indexing.
        """
        all_docs = []
        
        print("\n📂 Processing documents...")
        print("-" * 40)

        for category in DOC_CATEGORIES:
            category_path = os.path.join(base_path, category)
            
            # Create folder if it doesn't exist
            if not os.path.exists(category_path):
                os.makedirs(category_path)
                print(f"📁 Created folder: {category}/")
                continue

            # Find all PDFs in this category folder
            pdf_files = [f for f in os.listdir(category_path) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                print(f"📁 {category}/ - No PDFs found")
                continue
                
            print(f"\n📁 {category}/ - Found {len(pdf_files)} file(s)")
            
            for filename in pdf_files:
                pdf_path = os.path.join(category_path, filename)
                docs = self.load_pdf(pdf_path, category)
                all_docs.extend(docs)
                print(f"  ✓ {filename}")

        print("-" * 40)
        
        if not all_docs:
            print("⚠️ No documents found to process!")
            return []

        # Now chunk everything into smaller pieces
        print(f"\n✂️ Splitting {len(all_docs)} document(s) into chunks...")
        chunked_docs = self.text_splitter.split_documents(all_docs)
        
        print(f"✅ Created {len(chunked_docs)} chunks total\n")
        return chunked_docs
