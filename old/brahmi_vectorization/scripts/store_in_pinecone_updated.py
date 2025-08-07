import sys
import os
import json
import hashlib


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(ROOT_DIR)

import pinecone
import pandas as pd
import numpy as np
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, EMBEDDINGS_FILE

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
    )


index = pc.Index(PINECONE_INDEX_NAME)

with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    embeddings_dict = json.load(f)

for word, data in embeddings_dict.items():
    word_id = hashlib.md5(word.encode("utf-8")).hexdigest()  # Generate unique ASCII ID
    word_vector = data["word_vector"]
    combined_vector = data["combined_vector"]  # Context-aware embedding

    metadata = {
        "Brahmi_word": word,
        "Meaning": data["meaning"],
        "Context": data["context"]
    }

    # Upsert into Pinecone with the context-aware vector
    index.upsert([(word_id, combined_vector, metadata)])

print("Embeddings stored in Pinecone successfully!")
