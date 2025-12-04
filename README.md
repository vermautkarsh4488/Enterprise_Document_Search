# Document Search Chatbot

Chatbot that searches your company PDFs (HR, Finance, Technical).

## Quick Start

1. `pip install -r requirements.txt`
2. Add OpenAI key to `.env`
3. Put PDFs in `documents/HR/`, `documents/Finance/`, `documents/Technical/`
4. `streamlit run app.py`
5. Click "Re-index" then ask questions

## Tech
Python, Streamlit, LangChain, OpenAI, FAISS, OCR

## Features
- Search PDFs by asking questions
- Works with scanned PDFs too
- Shows source documents
