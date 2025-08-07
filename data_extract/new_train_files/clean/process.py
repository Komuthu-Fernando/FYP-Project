import re

# List of titles provided (including compound titles)
titles = [
    'ගණකාධිකාරී', 'ප්‍රධානි', 'ගම් ප්‍රධානි', 'සිව', 'ස්වාමි', 'උපාසිකා', 
    'උපාසක', 'ගෘහපති', 'බ්‍රාහ්මණ', 'අයුෂ්මන්ත', 'මහරජු', 'භාණ්ඩාගාරික', 
    'වෙළෙන්දා', 'සමූහයේ', 'කෝෂ්ඨාගාරික', 'අචාර්ය', 'සෙන්පති', 'ප්‍රධානිප්‍රධානි', 
    'වෙස්ස', 'රජ', 'ආචාර්ය', 'රජු'
]

# Transformation rules for last letters of PROPN following TITLE-FEM
letter_transform = {
    'ග': 'ගා', 'ල්': 'ලී', 'ස': 'සා', 'න': 'නා', 'ර': 'රා', 'ත': 'තා', 'ය': 'යා'
}

def is_title(word, xpos):
    """Check if the word is a title based on XPOS tag."""
    if xpos in ['TITLE-MASC', 'TITLE-FEM']:
        return True
    return False

def replace_words(word):
    """Replace specific words as per requirements."""
    if word == 'ප්‍රමුඛ':
        return 'ප්‍රධානි'
    elif word == 'වෙළඳ':
        return 'වෙළෙන්දා'
    elif word == 'ආයුෂ්මන්ත':
        return 'නැගෙනහිර'
    return word

def transform_sentence(words, pos_tags, xpos_tags):
    """Apply transformation rules to the sentence and return if adjusted."""
    transformed_words = words.copy()
    adjusted = False
    i = 0
    while i < len(words):
        # Rule 1: Handle පැමිණියාවූත් නොපැමිණියාවූත් සතරදෙස
        if (i < len(words) - 2 and 
            words[i] == 'පැමිණියාවූත්' and 
            words[i + 1] == 'නොපැමිණියාවූත්' and 
            words[i + 2] == 'සතරදෙස'):
            transformed_words[i:i + 3] = ['සතරදෙස', 'පැමිණියාවූත්', 'නොපැමිණියාවූත්']
            adjusted = True
            i += 3
            continue

        # Rule 2: Handle ADJ TITLE PROPN → PROPN ADJ TITLE
        if (i < len(words) - 2 and 
            pos_tags[i] == 'ADJ' and 
            is_title(words[i + 1], xpos_tags[i + 1]) and 
            pos_tags[i + 2] == 'PROPN'):
            title_words = [words[i + 1]]
            title_start = i + 1
            # Check for compound title like ගම් ප්‍රධානි
            if words[i + 1] == 'ගම්' and i + 2 < len(words) - 1 and words[i + 2] == 'ප්‍රධානි':
                title_words.append(words[i + 2])
                propn = words[i + 3] if i + 3 < len(words) else None
                if propn and pos_tags[i + 3] == 'PROPN':
                    transformed_words[i:i + 4] = [propn, words[i], words[i + 1], words[i + 2]]
                    # Apply TITLE-FEM transformation if applicable
                    if xpos_tags[i + 1] == 'TITLE-FEM':
                        last_letter = propn[-1]
                        if last_letter in letter_transform:
                            transformed_words[i] = propn[:-1] + letter_transform[last_letter]
                    adjusted = True
                    i += 4
                    continue
            else:
                propn = words[i + 2]
                transformed_words[i:i + 3] = [propn, words[i], words[i + 1]]
                # Apply TITLE-FEM transformation if applicable
                if xpos_tags[i + 1] == 'TITLE-FEM':
                    last_letter = propn[-1]
                    if last_letter in letter_transform:
                        transformed_words[i] = propn[:-1] + letter_transform[last_letter]
                adjusted = True
                i += 3
                continue

        # Rule 3: Handle ගම් ප්‍රධානි PROPN → PROPN ගම් ප්‍රධානි
        if (i < len(words) - 2 and 
            words[i] == 'ගම්' and 
            words[i + 1] == 'ප්‍රධානි' and 
            pos_tags[i + 2] == 'PROPN'):
            propn = words[i + 2]
            transformed_words[i:i + 3] = [propn, 'ගම්', 'ප්‍රධානි']
            # Apply TITLE-FEM transformation if applicable
            if xpos_tags[i + 1] == 'TITLE-FEM':
                last_letter = propn[-1]
                if last_letter in letter_transform:
                    transformed_words[i] = propn[:-1] + letter_transform[last_letter]
            adjusted = True
            i += 3
            continue

        # Rule 4: Handle TITLE PROPN → PROPN TITLE
        if (i < len(words) - 1 and 
            is_title(words[i], xpos_tags[i]) and 
            pos_tags[i + 1] == 'PROPN'):
            title_words = [words[i]]
            title_start = i
            propn = words[i + 1]
            transformed_words[i:i + 2] = [propn, title_words[0]]
            # Apply TITLE-FEM transformation if applicable
            if xpos_tags[i] == 'TITLE-FEM':
                last_letter = propn[-1]
                if last_letter in letter_transform:
                    transformed_words[i] = propn[:-1] + letter_transform[last_letter]
            adjusted = True
            i += 2
            continue

        i += 1
    return transformed_words, adjusted

def process_conllu(input_file, output_file):
    """Process the CoNLL-U file, write the output, and count adjusted sentences."""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    adjusted_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        sentence = []
        pos_tags = []
        xpos_tags = []
        dep_rels = []
        current_sentence = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('# text ='):
                # New sentence starts
                if current_sentence:
                    # Apply word replacements to input sentence
                    input_sentence = [replace_words(word) for word in current_sentence]
                    # Process previous sentence
                    transformed, was_adjusted = transform_sentence(input_sentence, pos_tags, xpos_tags)
                    if was_adjusted:
                        adjusted_count += 1
                    # Write to file
                    f.write(f"input: {' '.join(input_sentence)} | POS: {' '.join(pos_tags)} | XPOS: {' '.join(xpos_tags)} | Dep: {' '.join(dep_rels)} → output: {' '.join(transformed)}\n")
                # Reset for new sentence
                current_sentence = []
                pos_tags = []
                xpos_tags = []
                dep_rels = []
                sentence = line[len('# text = '):].strip()
            elif line and not line.startswith('#'):
                # Token line
                fields = line.split('\t')
                if len(fields) >= 7:
                    word = fields[1]
                    pos = fields[3]
                    xpos = fields[4] if fields[4] != '_' else '_'
                    dep = fields[7]
                    current_sentence.append(word)
                    pos_tags.append(pos)
                    xpos_tags.append(xpos)
                    dep_rels.append(dep)
        
        # Process the last sentence
        if current_sentence:
            input_sentence = [replace_words(word) for word in current_sentence]
            transformed, was_adjusted = transform_sentence(input_sentence, pos_tags, xpos_tags)
            if was_adjusted:
                adjusted_count += 1
            f.write(f"input: {' '.join(input_sentence)} | POS: {' '.join(pos_tags)} | XPOS: {' '.join(xpos_tags)} | Dep: {' '.join(dep_rels)} → output: {' '.join(transformed)}\n")
    
    # Print the number of adjusted sentences
    print(f"Number of sentences adjusted (excluding word replacements): {adjusted_count}")

# Run the script
process_conllu('test.conllu', 'test_new.txt')