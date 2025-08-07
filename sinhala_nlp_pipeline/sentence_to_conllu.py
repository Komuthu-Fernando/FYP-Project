import random
from pathlib import Path
import re

# Enhanced regex rules for Sinhala POS tagging
RULES = [
    (re.compile(r'.*ට$'), 'N-DAT'),          # Dative case nouns (e.g., සංඝයාට)
    (re.compile(r'.*ය$'), 'NOUN'),           # Singular nouns (e.g., සංඝයා)
    (re.compile(r'.*වා$'), 'VERB'),          # Verbs (e.g., දෙනවා)
    (re.compile(r'.*න$'), 'VERB'),           # Verb roots (e.g., දෙන)
    (re.compile(r'තිස්ස|තිශ|සුමන ගුප්ත|ගාමිණී ධම්මරාජ|සෝනගුප්ත|සමනක|සේන|සමුද්ද|සුමන|මිත්තදේව|මුලගුප්ත|උත්තිය|නන්දක|වේලු|කනිසත|නාග|ගුටක තිස්ස|අභිජි ගුත්ත|ශිව|ආනන්ද|නන්දික|මානාපදස්සන|උපාලි|දානාටක|තිස්සා'), 'PROPN'),  
    (re.compile(r'.*යි$'), 'ADJ'),           # Adjectives (e.g., හොඳයි)
    (re.compile(r'.*ම$'), 'ADJ'),            # Adjectives (e.g., ලස්සනම)
    (re.compile(r'පුජ|පූජා|දෙනු|ලදි'), 'VERB'),  # Specific verbs
    (re.compile(r'ගෘහපති|ප්‍රධානී|භාන්ඩාගාරික|තෙරුන්'), 'TITLE'),  # Titles/roles
    (re.compile(r'සහ|හා|ද'), 'CONJ'),        # Conjunctions
]

# Function to determine POS tag based on regex rules
def get_pos_tag(token):
    for pattern, tag in RULES:
        if pattern.match(token):
            return tag
    return 'NOUN'  # Default tag for unmatched tokens

# Basic SOV reordering and dependency assignment
def basic_sov_reorder(tagged_sentence, dependencies=None):
    subjects = [word for word, tag in tagged_sentence if tag in ['PROPN', 'TITLE', 'PRON']]
    verbs = [word for word, tag in tagged_sentence if tag == 'VERB']
    direct_objects = [word for word, tag in tagged_sentence if tag in ['NOUN', 'PLACE'] and word not in subjects]
    indirect_objects = [word for word, tag in tagged_sentence if tag == 'N-DAT' and word not in subjects]
    descriptors = [word for word, tag in tagged_sentence if tag == 'NOUN' and word not in subjects + direct_objects + indirect_objects]
    adjectives = [word for word, tag in tagged_sentence if tag == 'ADJ']
    conjunctions = [word for word, tag in tagged_sentence if tag == 'CONJ']
    
    if dependencies:
        root_verbs = [d['word'] for d in dependencies if d['head'] == 0 and d['word'] in verbs]
        if root_verbs:
            reordered = [w for w in subjects + descriptors + adjectives + direct_objects + indirect_objects + conjunctions if w not in root_verbs] + root_verbs
            return reordered
    
    return subjects + descriptors + adjectives + direct_objects + indirect_objects + conjunctions + verbs

# Function to assign basic dependencies based on SOV order
def assign_dependencies(tokens, tagged):
    dependencies = []
    verb_idx = -1
    for i, (word, tag) in enumerate(tagged, 1):
        if tag == 'VERB':
            verb_idx = i
            dependencies.append({'word': word, 'head': 0, 'deprel': 'root'})
        else:
            head = verb_idx if verb_idx != -1 else 0  # Default to verb or 0 if no verb found yet
            deprel = 'dep'  # Default dependency relation
            if tag in ['PROPN', 'TITLE', 'PRON']:
                deprel = 'nsubj'  # Nominal subject
            elif tag == 'NOUN':
                deprel = 'obj'  # Direct object
            elif tag == 'N-DAT':
                deprel = 'iobj'  # Indirect object
            elif tag == 'ADJ':
                deprel = 'amod'  # Adjectival modifier
            elif tag == 'CONJ':
                deprel = 'cc'  # Coordinating conjunction
            dependencies.append({'word': word, 'head': head, 'deprel': deprel})
    return dependencies

# Function to convert a sentence to CoNLL-U format
def sentence_to_conllu(sentence, sent_id):
    # Basic tokenization by splitting on whitespace
    tokens = sentence.strip().split()
    # Tag each token
    tagged = [(token, get_pos_tag(token)) for token in tokens]
    # Assign dependencies based on SOV order
    dependencies = assign_dependencies(tokens, tagged)
    conllu_lines = [f"# sent_id = {sent_id}", f"# text = {sentence.strip()}"]
    
    # Generate CoNLL-U lines
    for i, (token, tag) in enumerate(tagged, 1):
        dep = dependencies[i-1]
        # Columns: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
        line = f"{i}\t{token}\t{token}\t{tag}\t_\t_\t{dep['head']}\t{dep['deprel']}\t_\t_"
        conllu_lines.append(line)
    conllu_lines.append("")  # Empty line to separate sentences
    return "\n".join(conllu_lines)

# Read sentences from file in root directory
input_file = "data/direct_translation.txt"
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

try:
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"Error: {input_file} not found in current directory!")
    exit(1)

# Shuffle sentences for random splitting
random.seed(42)  # For reproducibility
random.shuffle(sentences)

# Calculate split sizes: 80% train, 10% dev, 10% test
total = len(sentences)
train_size = int(total * 0.8)
dev_size = int(total * 0.1)
test_size = total - train_size - dev_size

# Split sentences
train_sentences = sentences[:train_size]
dev_sentences = sentences[train_size:train_size + dev_size]
test_sentences = sentences[train_size + dev_size:]

# Convert and write to CoNLL-U files
def write_conllu(sentences, split_name, output_dir):
    output_file = output_dir / f"{split_name}_enhanced.conllu"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, sent in enumerate(sentences, 1):
            conllu_text = sentence_to_conllu(sent, f"{split_name}-{i}")
            f.write(conllu_text + "\n")
    print(f"Wrote {len(sentences)} sentences to {output_file}")

# Write train, dev, and test files
write_conllu(train_sentences, "train", output_dir)
write_conllu(dev_sentences, "dev", output_dir)
write_conllu(test_sentences, "test", output_dir)

print(f"Processed {total} sentences: {train_size} train, {dev_size} dev, {test_size} test")