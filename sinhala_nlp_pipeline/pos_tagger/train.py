import pycrfsuite
from pos_tagger.rules import RULES
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
training_data_path = os.path.join(CURRENT_DIR, '..', 'data', 'tagged_data.txt')
model_path = os.path.join(CURRENT_DIR, '..', 'models', 'model.crf')

def read_tagged_data(file_path):
    tagged_sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line:
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    word, tag = parts
                    sentence.append((word, tag))
                else:
                    print(f"Warning: Skipping invalid line - {line}")
            else:
                if sentence:
                    tagged_sentences.append(sentence)
                    sentence = []
        if sentence:
            tagged_sentences.append(sentence)
    return tagged_sentences

def word2features(sent, i):
    word = sent[i][0] if isinstance(sent[i], tuple) else sent[i]
    features = {
        'word': word,
        'is_first': i == 0,
        'is_last': i == len(sent) - 1,
        'is_name_like': word in ['තිස්ස', 'තිස්සා', 'සුමන', 'සේන', 'වේලු', 'නාග', 'ශිව', 'ආනන්ද'],
        'is_place_like': word in ['කලල', 'පියගැට', 'පෙළ', 'ලෙන', 'ගුහාව'],
        'is_title_like': word in ['ගෘහපති', 'ප්‍රධානී', 'භාන්ඩාගාරික', 'තෙරුන්'],
    }
    for pattern, tag in RULES:
        if pattern.match(word):
            features[f'matches_{tag}'] = True
    if i > 0:
        prev_word = sent[i-1][0] if isinstance(sent[i-1], tuple) else sent[i-1]
        features['prev_word'] = prev_word
    if i < len(sent) - 1:
        next_word = sent[i+1][0] if isinstance(sent[i+1], tuple) else sent[i+1]
        features['next_word'] = next_word
    return features

def train_crf_tagger(tagged_sentences):
    X_train = [[word2features(sent, i) for i in range(len(sent))] for sent in tagged_sentences]
    y_train = [[t for w, t in sent] for sent in tagged_sentences]
    trainer = pycrfsuite.Trainer(verbose=True)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 1.0,  # L1 penalty (sparsity)
        'c2': 1e-3,  # L2 penalty (smoothing)
        'max_iterations': 50,
    })
    trainer.train(model_path)
    return trainer

if __name__ == "__main__":
    tagged_sentences = read_tagged_data(training_data_path)
    train_crf_tagger(tagged_sentences)
    print(f"CRF model trained and saved as {model_path}")