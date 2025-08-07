import os

# Pinecone API Key and Environment
PINECONE_API_KEY = "pcsk_76ktnc_76gH2hWdtNkhLcdmChhmDYvDJXho3bxDNyyVkezpyaHuzFVxCWExLekqSHAgFHg"
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_INDEX_NAME = "brahmi-vector"

# Model settings
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Database settings
VECTOR_DIMENSION = 384  



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "data", "brahmi_data.csv")

EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "embeddings_updated_new.json")

EMBEDDING_METHOD = "sentence-transformer"