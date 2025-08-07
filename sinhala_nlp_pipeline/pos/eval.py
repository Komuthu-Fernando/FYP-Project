import os
import stanza
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# Define directories and paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models')
TEST_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'pos', 'si_custom.test.in.conllu')
pos_model_path = os.path.join(MODEL_DIR, 'si_custom_nocharlm_tagger.pt')
pos_pretrain_path = os.path.join(MODEL_DIR, 'si_custom_pretrain.pt')

# Load the trained model with both tokenize and pos processors
nlp = stanza.Pipeline(
    lang='si',
    processors='tokenize,pos',
    dir=MODEL_DIR,
    package='si_custom',
    pos_model_path=pos_model_path,
    pos_pretrain_path=pos_pretrain_path,
    download_method=stanza.DownloadMethod.REUSE_RESOURCES,
    use_cache=False,
    tokenize_pretokenized=True
)

# Function to read CoNLL file and extract gold tags and words manually
def read_conllu(file_path):
    gold_words = []
    gold_tags = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence = []
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                if current_sentence:
                    for item in current_sentence:
                        parts = item.split('\t')
                        if len(parts) >= 4 and parts[1] and parts[3]:
                            gold_words.append(parts[1])
                            gold_tags.append(parts[3])
                    current_sentence = []
            elif not line.startswith('#'):
                current_sentence.append(line)
        if current_sentence:
            for item in current_sentence:
                parts = item.split('\t')
                if len(parts) >= 4 and parts[1] and parts[3]:
                    gold_words.append(parts[1])
                    gold_tags.append(parts[3])
    print("Extracted Gold Words (sample):", gold_words[:10])
    print("Extracted Gold Tags (sample):", gold_tags[:10])
    return gold_words, gold_tags

# Function to predict POS tags with post-processing
def predict_pos_tags(words):
    doc = nlp([words])
    pred_tags = []
    for sent in doc.sentences:
        for word in sent.words:
            tag = word.upos
            pred_tags.append(tag)
    print("Predicted Tags (sample):", pred_tags[:10])
    return pred_tags

# Read test data
gold_words, gold_tags = read_conllu(TEST_FILE)

# Predict POS tags
pred_tags = predict_pos_tags(gold_words)

# Filter out rare tags (PROP, NUM, PRON, ADJ)
valid_indices = [i for i in range(len(gold_tags)) if gold_tags[i] not in ['PROP', 'NUM', 'PRON', 'ADJ']]
gold_tags_filtered = [gold_tags[i] for i in valid_indices]
pred_tags_filtered = [pred_tags[i] for i in valid_indices]
gold_words_filtered = [gold_words[i] for i in valid_indices]

# Ensure lengths match
min_length = min(len(gold_tags_filtered), len(pred_tags_filtered))
gold_tags_filtered = gold_tags_filtered[:min_length]
pred_tags_filtered = pred_tags_filtered[:min_length]

# Calculate and print evaluation metrics
print("POS Tagging Evaluation Report (excluding PROP, NUM, PRON, ADJ):")
report = classification_report(gold_tags_filtered, pred_tags_filtered, zero_division=0, output_dict=True)
print(classification_report(gold_tags_filtered, pred_tags_filtered, zero_division=0))

# Additional statistics
accuracy = np.mean([p == g for p, g in zip(pred_tags_filtered, gold_tags_filtered)])
print(f"Overall Accuracy (excluding PROP, NUM, PRON, ADJ): {accuracy:.4f}")

# Generate and save chart
tags = ["ADP", "CCONJ", "NOUN", "PART", "PROPN", "VERB"]
precision = [report[tag]['precision'] for tag in tags]
recall = [report[tag]['recall'] for tag in tags]
f1 = [report[tag]['f1-score'] for tag in tags]

x = np.arange(len(tags))  # Label locations
width = 0.25  # Width of the bars

plt.figure(figsize=(10, 6))
plt.bar(x - width, precision, width, label='Precision', color="#272168", alpha=0.7)
plt.bar(x, recall, width, label='Recall', color="#4D6EC3", alpha=0.7)
plt.bar(x + width, f1, width, label='F1-Score', color="#77B0F0", alpha=0.7)

plt.xlabel('POS Tags')
plt.ylabel('Score')
plt.title('POS Tagging Performance')
plt.ylim(0, 1.1)
plt.xticks(x, tags)
plt.legend()
plt.grid(True, alpha=0.3)

# Save the chart as PNG
chart_path = os.path.join(CURRENT_DIR, 'pos_tagging_performance.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"Chart saved as {chart_path}")

# Save results
with open(os.path.join(CURRENT_DIR, 'pos_eval_results.txt'), 'w', encoding='utf-8') as f:
    f.write("POS Tagging Evaluation Report (excluding PROP, NUM, PRON, ADJ):\n")
    f.write(classification_report(gold_tags_filtered, pred_tags_filtered, zero_division=0))
    f.write(f"\nOverall Accuracy (excluding PROP, NUM, PRON, ADJ): {accuracy:.4f}\n")