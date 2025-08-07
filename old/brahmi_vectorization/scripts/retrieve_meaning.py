import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import json
import pinecone
import numpy as np
import hashlib
from sentence_transformers import SentenceTransformer
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, SENTENCE_TRANSFORMER_MODEL

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)

model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

def get_embedding(word):
    embedding = model.encode(word).tolist()
    print(f"\nGenerated embedding for '{word}': {embedding[:5]}...")  # Print first 10 values
    return embedding


def search_brahmi_word(word, top_k=3):
    """Search for the most similar Brahmi words in Pinecone"""
    query_vector = get_embedding(word)

    # Debugging: Print query vector to ensure it changes for each input
    print(f"\nQuery Vector for '{word}':\n{query_vector[:5]}...")  # Print first 10 values

    # Perform similarity search
    search_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    if search_results and "matches" in search_results:
        print(f"\nSearch results for: {word}")
        for match in search_results["matches"]:
            metadata = match["metadata"]
            brahmi_word = metadata.get("Brahmi_word", "Unknown")
            meaning = metadata.get("Meaning", "Unknown")
            context = metadata.get("Context", "No Context Available")
            score = match["score"]

            print(f"\nBrahmi Word: {brahmi_word}")
            print(f"Meaning: {meaning}")
            print(f"Context: {context}")
            print(f"Similarity Score: {score}")
    else:
        print("No similar words found.")



if __name__ == "__main__":
    # Sample input word
    test_word = "දයා"  
    # test_word = "පරුම"
    search_brahmi_word(test_word)
