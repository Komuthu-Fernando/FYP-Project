# import os
# from rules import RULES
# from train import read_tagged_data, train_pos_tagger

# # Get the current script directory
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# training_data_path = os.path.join(CURRENT_DIR, 'data', 'tagged_data.txt')

# def hybrid_tagger(sentence, rules, pos_tagger):
#     """
#     Hybrid POS tagger using regex-based rules and a trained tagger.
#     """
#     tagged_sentence = []
#     for word in sentence:
#         # Apply rule-based tagging first
#         for pattern, tag in rules:
#             if pattern.match(word):
#                 tagged_sentence.append((word, tag))
#                 break
#         else:
#             # Use trained POS tagger with fallback
#             tag = pos_tagger.tag([word])[0][1] or 'UNK'
#             tagged_sentence.append((word, tag))
#     return tagged_sentence

# if __name__ == "__main__":
#     # Read and train
#     tagged_sentences = read_tagged_data(training_data_path)
#     pos_tagger = train_pos_tagger(tagged_sentences)

#     # Example sentence
#     sentence = ['මම', 'ගෙදර', 'සත්වයන්ට', 'යනවා']
    
#     # Tag the sentence
#     tagged = hybrid_tagger(sentence, RULES, pos_tagger)
#     print(tagged)

import os
import pycrfsuite
from rules import RULES
from train import read_tagged_data, train_crf_tagger, word2features

# Get the current script directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
training_data_path = os.path.join(CURRENT_DIR, 'data', 'tagged_data.txt')
model_path = os.path.join(CURRENT_DIR, 'model.crf')
output_path = os.path.join(CURRENT_DIR, 'tagger_output.txt')

def hybrid_tagger(sentence, rules):
    """
    Hybrid POS tagger using regex-based rules and a trained CRF tagger.
    """
    tagged_sentence = []
    
    # First, apply rule-based tagging
    for word in sentence:
        for pattern, tag in rules:
            if pattern.match(word):
                tagged_sentence.append((word, tag))
                break
        else:
            # If no rule matches, append with a placeholder tag (to be replaced by CRF)
            tagged_sentence.append((word, None))
    
    # Use CRF tagger for words not tagged by rules
    tagger = pycrfsuite.Tagger()
    if not os.path.exists(model_path):
        raise FileNotFoundError("CRF model not found. Please train the model first.")
    tagger.open(model_path)
    
    # Prepare features for the full sentence
    features = [word2features(sentence, i) for i in range(len(sentence))]
    crf_tags = tagger.tag(features)
    
    # Combine rule-based and CRF tags
    final_tagged_sentence = []
    for (word, rule_tag), crf_tag in zip(tagged_sentence, crf_tags):
        final_tag = rule_tag if rule_tag else crf_tag  # Prefer rule tag if it exists
        final_tagged_sentence.append((word, final_tag))
    
    return final_tagged_sentence

def crf_only_tagger(sentence):
    """
    POS tagger using only the trained CRF model, bypassing rules.
    """
    tagger = pycrfsuite.Tagger()
    if not os.path.exists(model_path):
        raise FileNotFoundError("CRF model not found. Please train the model first.")
    tagger.open(model_path)
    
    # Prepare features for the full sentence
    features = [word2features(sentence, i) for i in range(len(sentence))]
    crf_tags = tagger.tag(features)
    
    # Return word-tag pairs
    return [(word, tag) for word, tag in zip(sentence, crf_tags)]

if __name__ == "__main__":
    # Read and train (only if model doesn't exist)
    if not os.path.exists(model_path):
        tagged_sentences = read_tagged_data(training_data_path)
        train_crf_tagger(tagged_sentences)
        print("CRF model trained and saved.")

    # Example sentence
    sentence = ['ප්‍රධානී', 'ශිවගේ,', 'ගුහාව,', 'සතර', 'දිසාවේ', 'සංඝයාට', 'දෙනු', 'ලැබේ']
    
    # Tag the sentence with CRF only
    crf_tagged = crf_only_tagger(sentence)
    
    # Tag the sentence with hybrid tagger
    hybrid_tagged = hybrid_tagger(sentence, RULES)
    
    # Save output to a text file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("CRF-only tagged sentence:\n")
        f.write(str(crf_tagged) + "\n\n")
        f.write("Hybrid tagged sentence:\n")
        f.write(str(hybrid_tagged) + "\n")
    
    print(f"Output saved to {output_path}")