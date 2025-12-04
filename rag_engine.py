"""
The brain of the app - handles searching and answering questions.
Uses OpenAI for embeddings and chat, FAISS for fast vector search.
"""

import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
    TEMPERATURE, VECTOR_STORE_PATH
)


class RAGEngine:
    
    def __init__(self):
        print("🔧 Initializing RAG engine...")
        
        # Setup OpenAI embeddings (converts text to vectors)
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Setup the chat model
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        
        # These get set up later
        self.vector_store = None
        self.qa_chain = None
        
        # Memory to remember conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def build_index(self, documents: List):
        """
        Create a searchable index from the documents.
        This is the slow part - only needs to run once after adding new docs.
        """
        print("🔨 Building vector index...")
        
        # Convert all documents to vectors and store them
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save to disk so we don't have to rebuild every time
        self.vector_store.save_local(VECTOR_STORE_PATH)
        
        # Setup the QA chain now that we have vectors
        self._setup_qa_chain()
        
        print("✅ Index built and saved!")

    def load_index(self):
        """
        Load a previously saved index from disk.
        Returns True if successful, False if no index exists.
        """
        if not os.path.exists(VECTOR_STORE_PATH):
            print("ℹ️ No existing index found. Please index some documents first.")
            return False
            
        print("📂 Loading existing index...")
        
        self.vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True  # needed for loading pickled data
        )
        
        self._setup_qa_chain()
        print("✅ Index loaded!")
        return True

    def _setup_qa_chain(self):
        """
        Setup the question-answering chain with our custom prompt.
        This tells the AI how to behave when answering questions.
        """
        
        # Custom prompt to make the AI behave like a helpful document assistant
        prompt_template = """You are a helpful assistant that answers questions based on company documents.

Here's the relevant information I found:
{context}

Previous conversation:
{chat_history}

User's question: {question}

Please answer the question based on the information above. A few guidelines:
- Only use information from the provided documents
- Mention which document the information came from (HR, Finance, or Technical)
- If you can't find the answer, just say so - don't make stuff up
- Keep your answer clear and to the point

Your answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )

        # Create the retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="mmr",  # MMR helps get diverse results
                search_kwargs={"k": 5, "fetch_k": 10}  # get top 5 from 10 candidates
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True  # we want to show sources to users
        )

    def query(self, question: str, category_filter: Optional[str] = None) -> dict:
        """
        Ask a question and get an answer with sources.
        Optionally filter by document category.
        """
        
        # Make sure we have an index loaded
        if not self.qa_chain:
            return {
                "answer": "⚠️ No documents indexed yet. Please add some documents and click 'Re-index'.",
                "sources": []
            }

        # Apply category filter if user selected one
        if category_filter and category_filter != "All":
            self.qa_chain.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "filter": {"category": category_filter}
                }
            )

        # Run the query
        result = self.qa_chain.invoke({"question": question})

        # Extract source info to show the user
        sources = []
        seen_files = set()  # avoid showing duplicate sources
        
        for doc in result.get("source_documents", []):
            filename = doc.metadata.get("filename", "Unknown")
            
            # Skip if we already added this file
            if filename in seen_files:
                continue
            seen_files.add(filename)
            
            sources.append({
                "filename": filename,
                "category": doc.metadata.get("category", "Unknown"),
                "content_preview": doc.page_content[:200] + "..."
            })

        return {
            "answer": result["answer"],
            "sources": sources
        }

    def clear_memory(self):
        """Clear the conversation history to start fresh."""
        self.memory.clear()
        print("🗑️ Conversation history cleared")
