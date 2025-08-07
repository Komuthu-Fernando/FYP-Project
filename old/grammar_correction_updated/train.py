# import nltk

# def read_tagged_data(file_path):
#     """
#     Read tagged data from a file, where each line contains 'word tag'.
#     Sentences are separated by blank lines.
#     """
#     tagged_sentences = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         sentence = []
#         for line in f:
#             line = line.strip()
#             if line:
#                 parts = line.split()
#                 if len(parts) == 2:
#                     word, tag = parts
#                     sentence.append((word, tag))
#                 else:
#                     print(f"Warning: Skipping invalid line - {line}")
#             else:
#                 if sentence:
#                     tagged_sentences.append(sentence)
#                     sentence = []
#         if sentence:
#             tagged_sentences.append(sentence)
#     return tagged_sentences

# def train_pos_tagger(tagged_sentences):
#     """
#     Train a POS tagger using Unigram and Bigram models with backoff.
#     """
#     unigram_tagger = nltk.UnigramTagger(tagged_sentences)
#     bigram_tagger = nltk.BigramTagger(tagged_sentences, backoff=unigram_tagger)
#     return bigram_tagger


import pycrfsuite
from rules import RULES
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
training_data_path = os.path.join(CURRENT_DIR, 'data', 'tagged_data.txt')
model_path = os.path.join(CURRENT_DIR, 'model.crf')

def read_tagged_data(file_path):
    """
    Read tagged data from a file, where each line contains 'word(s) tag'.
    Sentences are separated by blank lines.
    """
    tagged_sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line:
                # Split on the last space to handle multi-word tokens
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
        'is_name_like': word in ['තිස්ස', 'සුමන', 'සේන', 'වේලු', 'නාග', 'ශිව', 'ආනන්ද'],  # Gazetteer
        'is_place_like': word in ['කලල', 'පියගැට පෙළ'],
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
    """
    Train a CRF tagger using the tagged sentences.
    """
    # Prepare training data
    X_train = [[word2features(sent, i) for i in range(len(sent))] for sent in tagged_sentences]
    y_train = [[t for w, t in sent] for sent in tagged_sentences]
    
    # Train the CRF model
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 1.0,  # L1 penalty (sparsity)
        'c2': 1e-3,  # L2 penalty (smoothing)
        'max_iterations': 50,  # Stop earlier if convergence
    })
    trainer.train('model.crf')  # Save the model to a file
    return trainer

if __name__ == "__main__":
    # For testing the training process
    tagged_sentences = read_tagged_data(training_data_path)
    train_crf_tagger(tagged_sentences)
    print("CRF model trained and saved as 'model.crf'")