import os
import pycrfsuite
from pos_tagger.rules import RULES
from pos_tagger.train import read_tagged_data, train_crf_tagger, word2features

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
training_data_path = os.path.join(CURRENT_DIR, '..', 'data', 'tagged_data.txt')
model_path = os.path.join(CURRENT_DIR, '..', 'models', 'model.crf')

def hybrid_tagger(sentence):
    tagged_sentence = []
    for word in sentence:
        for pattern, tag in RULES:
            if pattern.match(word):
                tagged_sentence.append((word, tag))
                break
        else:
            tagged_sentence.append((word, None))
    
    tagger = pycrfsuite.Tagger()
    if not os.path.exists(model_path):
        tagged_sentences = read_tagged_data(training_data_path)
        train_crf_tagger(tagged_sentences)
    tagger.open(model_path)
    
    features = [word2features(sentence, i) for i in range(len(sentence))]
    crf_tags = tagger.tag(features)
    
    final_tagged_sentence = []
    for (word, rule_tag), crf_tag in zip(tagged_sentence, crf_tags):
        final_tag = rule_tag if rule_tag else crf_tag
        final_tagged_sentence.append((word, final_tag))
    
    return final_tagged_sentence

if __name__ == "__main__":
    sentence = ['තිස්සා', 'පුජ', 'ලෙන', 'සන්ඝයා']
    tagged = hybrid_tagger(sentence)
    print(f"Input: {sentence}")
    print(f"Tagged: {tagged}")