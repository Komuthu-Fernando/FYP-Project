import os
import re

# Mapping from your custom tags to UPOS
pos_map = {
    'PROP': 'PROPN', 'N': 'NOUN', 'V': 'VERB', 'PLACE': 'NOUN', 'TITLE': 'NOUN',
    'ADJ': 'ADJ', 'CONJ': 'CONJ', 'N-DAT': 'NOUN'
}

def enhance_conllu(input_file, output_file, custom_tag_index=None):
    """
    Enhance a CONLL-U file by mapping custom tags to UPOS tags and setting lemma as word form.
    
    Args:
        input_file: Path to the input CONLL-U file
        output_file: Path to the output file
        custom_tag_index: Index of the column containing custom tags to map (None to use UPOS)
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.startswith('#'):
                outfile.write(line)
            elif line.strip():
                fields = line.strip().split('\t')
                if len(fields) >= 4:  # Ensure minimum fields (ID, FORM, LEMMA, UPOS)
                    word = fields[1]
                    lemma = word  # Set lemma as word form
                    fields[2] = lemma  # Update LEMMA
                    
                    # Determine UPOS based on custom_tag_index or preserve existing UPOS
                    if custom_tag_index is not None and custom_tag_index < len(fields):
                        custom_tag = fields[custom_tag_index].strip()
                        upos = pos_map.get(custom_tag, fields[3])  # Map custom tag, fallback to existing UPOS
                    else:
                        upos = fields[3]  # Preserve existing UPOS if no custom tag index specified
                    fields[3] = upos  # Update UPOS
                    
                    outfile.write('\t'.join(fields) + '\n')
            else:
                outfile.write('\n')

if __name__ == "__main__":
    # Default behavior: Preserve existing UPOS tags and set lemma as word
    enhance_conllu('data/train.conllu', 'data/train_enhanced.conllu')
    enhance_conllu('data/dev.conllu', 'data/dev_enhanced.conllu')
    enhance_conllu('data/test.conllu', 'data/test_enhanced.conllu')