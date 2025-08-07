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


if EMBEDDING_METHOD == "fasttext":
    model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    def get_embedding(text):
        return model.get_word_vector(text).tolist()

elif EMBEDDING_METHOD == "word2vec":
    model = Word2Vec.load(WORD2VEC_MODEL_PATH)

    def get_embedding(text):
        return model.wv[text].tolist() if text in model.wv else np.zeros(300).tolist()

elif EMBEDDING_METHOD == "sentence-transformer":
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    def get_embedding(text):
        return model.encode(text).tolist()

else:
    raise ValueError("Invalid embedding method selected!")

# Generate embeddings
for _, row in df.iterrows():
    word = str(row["brahmi_word"]).strip()
    meaning = str(row["meaning"]).strip()
    context = str(row["context"]).strip() if pd.notna(row["context"]) else ""

    # Create a combined text representation (Brahmi word + context)
    if context:
        combined_text = f"{word} [SEP] {context}"  # Using [SEP] to separate word and context
    else:
        combined_text = word

    # Generate embeddings for both word-only and word + context
    word_embedding = get_embedding(word)  # Word-only embedding
    combined_embedding = get_embedding(combined_text)  # Word + context embedding

    embeddings_dict[word] = {
        "word_vector": word_embedding,  # Embedding for word only
        "combined_vector": combined_embedding,  # Embedding for word + context
        "meaning": meaning,
        "context": context
    }

with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
    json.dump(embeddings_dict, f, indent=4, ensure_ascii=False)

print(f"Embeddings saved to {EMBEDDINGS_FILE}")
