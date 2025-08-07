import fasttext

model_path = "/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/models/cc.si.300.bin"
model = fasttext.load_model(model_path)

words = ["මම", "තමා", "සිංහල"]
for word in words:
    vector = model.get_word_vector(word)
    print(f"Vector for '{word}': {vector}")