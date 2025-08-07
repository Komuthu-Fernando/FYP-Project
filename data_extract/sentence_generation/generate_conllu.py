# import re

# def conllu_to_sentences_data(file_path):
#     """
#     Convert a CoNLL-U dataset file to a sentences_data list format.
#     Each sentence is a list of tuples (id, form, lemma, pos, features).
#     """
#     sentences_data = []
#     current_sentence = []
    
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
    
#     for line in lines:
#         line = line.strip()
        
#         # Skip empty lines (sentence boundaries)
#         if not line:
#             if current_sentence:
#                 sentences_data.append(current_sentence)
#                 current_sentence = []
#             continue
        
#         # Skip comments (e.g., # text = ...)
#         if line.startswith('#'):
#             continue
        
#         # Parse token line
#         fields = line.split('\t')
#         if len(fields) >= 6:  # Ensure valid CoNLL-U line
#             token_id = fields[0]
#             form = fields[1]
#             lemma = fields[2]
#             pos = fields[3]
#             features = fields[5] if fields[5] != '_' else '_'
            
#             # Create tuple for the token
#             token_tuple = (token_id, form, lemma, pos, features)
#             current_sentence.append(token_tuple)
    
#     # Append the last sentence if it exists
#     if current_sentence:
#         sentences_data.append(current_sentence)
    
#     return sentences_data

# def generate_conllu(file_path):
#     """
#     Generate CoNLL-U format from a CoNLL-U file.
#     Reads the file, converts to sentences_data format, and assigns dependency relations.
#     """
#     # Convert CoNLL-U file to sentences_data format
#     sentences_data = conllu_to_sentences_data(file_path)
    
#     output = []
    
#     for sentence in sentences_data:
#         # Extract the text of the sentence for the # text line
#         text = " ".join(token[1] for token in sentence)
#         output.append(f"# text = {text}")
        
#         # Initialize variables for dependency parsing
#         tokens = []
#         for token in sentence:
#             token_id, form, lemma, pos, features = token
#             # Map features to CoNLL-U format (if any)
#             feats = "_" if not features else features
#             # Placeholder for dependency head and relation
#             deprel = "_"
#             head = "0"
            
#             # Dependency parsing logic based on POS and context
#             if pos == "NOUN" and features in ["TITLE-MASC", "TITLE-FEM"]:
#                 deprel = "appos"  # Titles often act as appositions to proper names
#                 head = str(int(token_id) + 1) if int(token_id) < len(sentence) else "0"
#             elif pos == "PROPN":
#                 deprel = "nmod" if "ලෙණ" in [t[1] for t in sentence] else "root"
#                 head = str([t[0] for t in sentence if t[1] == "ලෙණ"][0]) if "ලෙණ" in [t[1] for t in sentence] else "0"
#             elif pos == "NOUN" and form == "ලෙණ":
#                 deprel = "root"
#                 head = "0"
#             elif pos == "VERB":
#                 deprel = "acl"  # Verbs often modify the main noun (ලෙණ)
#                 head = str([t[0] for t in sentence if t[1] == "ලෙණ"][0]) if "ලෙණ" in [t[1] for t in sentence] else "0"
#                 if "compound:svc" in [t[4] for t in sentence if t[0] > token_id]:
#                     deprel = "compound:svc"
#                     head = str(int(token_id) + 1)
#             elif pos == "NOUN" and form == "සංඝයා":
#                 deprel = "obl"
#                 head = str([t[0] for t in sentence if t[1] == "ලෙණ"][0]) if "ලෙණ" in [t[1] for t in sentence] else "0"
#             elif pos == "NOUN" and form == "පුත්":
#                 deprel = "compound"
#                 head = str(int(token_id) - 1) if int(token_id) > 1 else "0"
#             elif pos == "ADP" and form == "වන":
#                 deprel = "case"
#                 head = str(int(token_id) - 1)
#             elif pos == "CCONJ" and form == "සහ":
#                 deprel = "cc"
#                 head = str(int(token_id) + 1)
#             elif pos == "PART" and form == "ද":
#                 deprel = "advmod"
#                 head = str(int(token_id) + 1)
#             elif pos == "NOUN" and form == "සතරදෙස":
#                 deprel = "compound"
#                 head = str(int(token_id) + 1)
#             elif pos == "PUNCT":
#                 deprel = "punct"
#                 head = str(int(token_id) - 1)
#             else:
#                 deprel = "nmod"  # Default for other nouns or proper names
#                 head = str([t[0] for t in sentence if t[1] == "ලෙණ"][0]) if "ලෙණ" in [t[1] for t in sentence] else "0"
            
#             # CoNLL-U format: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
#             tokens.append(f"{token_id}\t{form}\t{lemma}\t{pos}\t_\t{feats}\t{head}\t{deprel}\t_\t_")
        
#         output.extend(tokens)
#         output.append("")  # Empty line between sentences
    
#     return "\n".join(output)

# # Example usage
# if __name__ == "__main__":
#     # Input CoNLL-U file path
#     input_file = "si_custom.dev.in.conllu"
    
#     # Generate CoNLL-U output
#     conllu_output = generate_conllu(input_file)
    
#     # Write to output file
#     output_file = "output.conllu"
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write(conllu_output)
    
#     # Print the output for verification
#     print(conllu_output)


import re

def conllu_to_sentences_data(file_path):
    """
    Convert a CoNLL-U dataset file to a sentences_data list format.
    Each sentence is a list of tuples (id, form, lemma, upos, xpos, features).
    """
    sentences_data = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines (sentence boundaries)
        if not line:
            if current_sentence:
                sentences_data.append(current_sentence)
                current_sentence = []
            continue
        
        # Skip comments (e.g., # text = ...)
        if line.startswith('#'):
            continue
        
        # Parse token line
        fields = line.split('\t')
        if len(fields) >= 6:  # Ensure valid CoNLL-U line
            token_id = fields[0]
            form = fields[1]
            lemma = fields[2]
            upos = fields[3]
            xpos = fields[4] if len(fields) > 4 and fields[4] != '_' else '_'  # Extract XPOS if available
            features = fields[5] if len(fields) > 5 and fields[5] != '_' else '_'
            
            # Create tuple for the token
            token_tuple = (token_id, form, lemma, upos, xpos, features)
            current_sentence.append(token_tuple)
    
    # Append the last sentence if it exists
    if current_sentence:
        sentences_data.append(current_sentence)
    
    return sentences_data

def generate_conllu(file_path):
    """
    Generate CoNLL-U format from a CoNLL-U file.
    Reads the file, converts to sentences_data format, and assigns dependency relations.
    Includes XPOS field when appropriate.
    """
    # Convert CoNLL-U file to sentences_data format
    sentences_data = conllu_to_sentences_data(file_path)
    
    output = []
    
    for sentence in sentences_data:
        # Extract the text of the sentence for the # text line
        text = " ".join(token[1] for token in sentence)
        output.append(f"# text = {text}")
        
        # Initialize variables for dependency parsing
        tokens = []
        for token in sentence:
            token_id, form, lemma, upos, xpos, features = token
            # Map features to CoNLL-U format (if any)
            feats = "_" if not features else features
            # Use XPOS if available; otherwise, derive it from UPOS or set to '_'
            xpos_out = xpos if xpos != '_' else upos if upos in ["NOUN", "PROPN", "VERB", "ADP", "CCONJ", "PART", "ADJ"] else '_'
            # Placeholder for dependency head and relation
            deprel = "_"
            head = "0"
            
            # Dependency parsing logic based on UPOS and context
            if upos == "NOUN" and features in ["TITLE-MASC", "TITLE-FEM"]:
                deprel = "appos"  # Titles often act as appositions to proper names
                head = str(int(token_id) + 1) if int(token_id) < len(sentence) else "0"
            elif upos == "PROPN":
                deprel = "nmod" if "ලෙණ" in [t[1] for t in sentence] else "root"
                head = str([t[0] for t in sentence if t[1] == "ලෙණ"][0]) if "ලෙණ" in [t[1] for t in sentence] else "0"
            elif upos == "NOUN" and form == "ලෙණ":
                deprel = "root"
                head = "0"
            elif upos == "VERB":
                deprel = "acl"  # Verbs often modify the main noun (ලෙණ)
                head = str([t[0] for t in sentence if t[1] == "ලෙණ"][0]) if "ලෙණ" in [t[1] for t in sentence] else "0"
                if "compound:svc" in [t[5] for t in sentence if t[0] > token_id]:
                    deprel = "compound:svc"
                    head = str(int(token_id) + 1)
            elif upos == "NOUN" and form == "සංඝයා":
                deprel = "obl"
                head = str([t[0] for t in sentence if t[1] == "ලෙණ"][0]) if "ලෙණ" in [t[1] for t in sentence] else "0"
            elif upos == "NOUN" and form == "පුත්":
                deprel = "compound"
                head = str(int(token_id) - 1) if int(token_id) > 1 else "0"
            elif upos == "ADP" and form == "වන":
                deprel = "case"
                head = str(int(token_id) - 1)
            elif upos == "CCONJ" and form == "සහ":
                deprel = "cc"
                head = str(int(token_id) + 1)
            elif upos == "PART" and form == "ද":
                deprel = "advmod"
                head = str(int(token_id) + 1)
            elif upos == "NOUN" and form == "සතරදෙස":
                deprel = "compound"
                head = str(int(token_id) + 1)
            elif upos == "PUNCT":
                deprel = "punct"
                head = str(int(token_id) - 1)
            else:
                deprel = "nmod"  # Default for other nouns or proper names
                head = str([t[0] for t in sentence if t[1] == "ලෙණ"][0]) if "ලෙණ" in [t[1] for t in sentence] else "0"
            
            # CoNLL-U format: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
            tokens.append(f"{token_id}\t{form}\t{lemma}\t{upos}\t{xpos_out}\t{feats}\t{head}\t{deprel}\t_\t_")
        
        output.extend(tokens)
        output.append("")  # Empty line between sentences
    
    return "\n".join(output)

# Example usage
if __name__ == "__main__":
    # Input CoNLL-U file path
    input_file = "si_custom.dev.in.conllu"
    
    # Generate CoNLL-U output
    conllu_output = generate_conllu(input_file)
    
    # Write to output file
    output_file = "output.conllu"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(conllu_output)
    
    # Print the output for verification
    print(conllu_output)