import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(ROOT_DIR)

import json
import pandas as pd
import numpy as np
import fasttext
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from config import CSV_FILE, EMBEDDINGS_FILE, EMBEDDING_METHOD, SENTENCE_TRANSFORMER_MODEL

df = pd.read_csv(CSV_FILE)

embeddings_dict = {}

# Select embedding method
if EMBEDDING_METHOD == "fasttext":
    model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    def get_embedding(word):
        return model.get_word_vector(word).tolist()

elif EMBEDDING_METHOD == "word2vec":
    model = Word2Vec.load(WORD2VEC_MODEL_PATH)
    def get_embedding(word):
        return model.wv[word].tolist() if word in model.wv else np.zeros(300).tolist()

elif EMBEDDING_METHOD == "sentence-transformer":
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    def get_embedding(word):
        return model.encode(word).tolist()

else:
    raise ValueError("Invalid embedding method selected!")

# Generate embeddings
for _, row in df.iterrows():
    word = row["brahmi_word"]
    meaning = row["meaning"]
    context = row["context"] if pd.notna(row["context"]) else ""


    embeddings_dict[word] = {
        "vector": get_embedding(word),
        "meaning": meaning,
        "context": context
    }

# Save embeddings to JSON
with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
    json.dump(embeddings_dict, f, indent=4, ensure_ascii=False)

print(f"Embeddings saved to {EMBEDDINGS_FILE}")
