from sentence_transformers import SentenceTransformer

# Load your current model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # Change if using another model

# Check embeddings for Sinhala and English words
sin_word1 = "දයා"
sin_word2 = "පරුමක"
eng_word1 = "compassion"
eng_word2 = "leader"

sin_embedding1 = model.encode(sin_word1)
sin_embedding2 = model.encode(sin_word2)
eng_embedding1 = model.encode(eng_word1)
eng_embedding2 = model.encode(eng_word2)

# Print first 10 values
print("Sinhala Embedding 1:", sin_embedding1[:5])
print("Sinhala Embedding 2:", sin_embedding2[:5])
print("English Embedding 1:", eng_embedding1[:5])
print("English Embedding 2:", eng_embedding2[:5])
