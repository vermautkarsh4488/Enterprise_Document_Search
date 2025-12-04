"""
Main Streamlit app - this is what users interact with.
Run with: streamlit run app.py
"""

import streamlit as st
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from config import DOC_CATEGORIES


# Page setup
st.set_page_config(
    page_title="Document Search",
    page_icon="📄",
    layout="wide"
)


# Initialize everything (only runs once per session)
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
    st.session_state.rag_engine.load_index()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============ SIDEBAR ============
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Category filter dropdown
    category_filter = st.selectbox(
        "Search in category",
        ["All"] + DOC_CATEGORIES,
        help="Filter results by document type"
    )
    
    st.divider()
    
    # Document management section
    st.subheader("📁 Documents")
    
    # Re-index button
    if st.button("🔄 Re-index Documents", use_container_width=True):
        with st.spinner("Processing... this might take a minute"):
            processor = DocumentProcessor()
            docs = processor.process_documents()
            
            if docs:
                st.session_state.rag_engine.build_index(docs)
                st.success(f"Done! Indexed {len(docs)} chunks")
            else:
                st.warning("No PDFs found. Add some documents first!")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag_engine.clear_memory()
        st.rerun()
    
    st.divider()
    
    # Help text
    st.caption("📌 Add your PDFs here:")
    for cat in DOC_CATEGORIES:
        st.code(f"documents/{cat}/", language=None)


# ============ MAIN CHAT AREA ============
st.title("🔍 Document Search & Q&A")
st.caption("Ask questions about company documents - I'll find the answers and show you where they came from")

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.markdown(f"**{src['category']}** → `{src['filename']}`")
                    st.caption(src["content_preview"])


# Chat input
user_question = st.chat_input("Ask me anything about your documents...")

if user_question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Get and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            # Get the answer
            filter_value = category_filter if category_filter != "All" else None
            response = st.session_state.rag_engine.query(user_question, filter_value)
            
            # Display answer
            st.markdown(response["answer"])
            
            # Display sources
            if response["sources"]:
                with st.expander("📚 Sources"):
                    for src in response["sources"]:
                        st.markdown(f"**{src['category']}** → `{src['filename']}`")
                        st.caption(src["content_preview"])
    
    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response["sources"]
    })
