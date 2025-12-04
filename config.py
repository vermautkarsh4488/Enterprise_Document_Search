"""
Configuration file for the document search app.
Update these values based on your needs.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Your OpenAI API key (keep this secret!)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Document categories - add more if needed
DOC_CATEGORIES = ["HR", "Finance", "Technical"]

# Chunking settings
# Smaller chunks = more precise but might miss context
# Larger chunks = more context but slower search
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  # overlap helps maintain context between chunks

# Which OpenAI models to use
EMBEDDING_MODEL = "text-embedding-3-small"  # good balance of cost and quality
LLM_MODEL = "gpt-4o-mini"  # fast and cheap, switch to gpt-4o for better answers
TEMPERATURE = 0.2  # lower = more focused answers, higher = more creative

# Where to save the vector database
VECTOR_STORE_PATH = "./vector_store"
