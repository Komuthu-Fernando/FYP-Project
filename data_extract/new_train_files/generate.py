# import random

# # Vocabulary lists based on the dataset
# titles = ["ප්‍රධානි", "උපාසක", "උපාසිකා", "ගෘහපති", "ස්වාමි", "ආයුෂ්මන්ත", "තෙරුණ්", "භාණ්ඩාගාරික", "අමාත්‍ය", "ආචාර්ය"]
# names = ["තිස්ස", "සුමන", "නාග", "ඵුස්ස", "උත්තර", "දත්ත", "ධම්මරක්ඛිත", "සිව", "මිත්ත", "කණ්හ", "සොණ", "අභය"]
# relationships = ["පුත්", "භාර්යාව", "දියණිය", "සහෝදරයා"]
# verbs = ["දෙන ලදී", "කරවන ලද", "පවරා දෙන ලදී"]
# modifiers = ["පැමිණියාවූත් නොපැමිණියාවූත් සතරදෙස", ""]
# sangha = "සංඝයා"

# # Function to generate a random sentence
# def generate_sentence():
#     structure = random.choice([
#         "[TITLE] [NAME] ලෙණ [SANGA]",
#         "[NAME] ලෙණ [SANGA]",
#         "[TITLE] [NAME] [RELATION] [TITLE] [NAME] ලෙණ [SANGA] [VERB]",
#         "[TITLE] [NAME] [RELATION] [TITLE] [NAME] ලෙණ [SANGA]",
#         "[NAME] [RELATION] [TITLE] [NAME] ලෙණ [SANGA]",
#         "[TITLE] [NAME] ලෙණ [MODIFIER] [SANGA]",
#         "[TITLE] [NAME] [RELATION] [NAME] ලෙණ [SANGA]"
#     ])
    
#     title = random.choice(titles)
#     name1 = random.choice(names)
#     name2 = random.choice(names)
#     relation = random.choice(relationships)
#     verb = random.choice(verbs)
#     modifier = random.choice(modifiers)
    
#     sentence = structure.replace("[TITLE]", title).replace("[NAME]", name1, 1)
#     sentence = sentence.replace("[NAME]", name2).replace("[RELATION]", relation)
#     sentence = sentence.replace("[SANGA]", sangha).replace("[VERB]", verb).replace("[MODIFIER]", modifier)
    
#     return sentence.strip()

# # Function to generate CoNLL-U entry with a single root and valid heads
# def generate_conllu_entry(sentence):
#     tokens = sentence.split()
#     conllu_lines = [f"# text = {sentence}"]
#     token_id = 1
#     root_id = None
#     max_id = len(tokens)  # Maximum valid token ID
    
#     # Identify the root (ලෙණ)
#     for i, token in enumerate(tokens, 1):
#         if token == "ලෙණ":
#             root_id = i
#             break
    
#     if not root_id:
#         # Fallback: make the last token the root if ලෙණ is not found
#         root_id = max_id
    
#     # Assign heads and dependencies
#     for i, token in enumerate(tokens, 1):
#         lemma = token  # Simplified: using token as lemma
#         pos = "NOUN" if token in ["ලෙණ", sangha] or token in relationships else "PROPN" if token in names else "VERB" if token in verbs else "ADP" if token == "වන" else "NOUN"
        
#         # Dependency and head assignment
#         if token == "ලෙණ":
#             dep = "root"
#             head = 0
#         elif token == sangha:
#             dep = "obl"
#             head = root_id
#         elif token in verbs:
#             dep = "acl"
#             head = root_id
#         elif token in relationships:
#             dep = "compound"
#             # Prefer linking to the next token (name) if it exists and is a name, else root
#             head = i + 1 if i < max_id and tokens[i] in names else root_id
#         elif token == "වන":
#             dep = "case"
#             # Link to the previous token (relationship) if it exists, else root
#             head = i - 1 if i > 1 and tokens[i-2] in relationships else root_id
#         elif token in titles:
#             dep = "appos"
#             # Link to the next token (name) if it exists and is a name, else root
#             head = i + 1 if i < max_id and tokens[i] in names else root_id
#         else:  # Names or other tokens
#             dep = "nmod"
#             head = root_id
        
#         # Validate head: must be 0 or within range [1, max_id]
#         if head != 0 and (head < 1 or head > max_id):
#             head = root_id  # Fallback to root if invalid
        
#         conllu_lines.append(f"{token_id}\t{token}\t{lemma}\t{pos}\t_\t_\t{head}\t{dep}\t_\t_")
#         token_id += 1
#     conllu_lines.append("")
#     return "\n".join(conllu_lines)

# # Generate multiple entries
# def generate_dataset(num_entries):
#     dataset = []
#     for _ in range(num_entries):
#         sentence = generate_sentence()
#         conllu_entry = generate_conllu_entry(sentence)
#         dataset.append(conllu_entry)
#     return "\n".join(dataset)

# # Generate 10 synthetic entries
# output = generate_dataset(5000)

# # Write to file
# with open("synthetic_conllu.conllu", "w", encoding="utf-8") as f:
#     f.write(output)

# print("Generated 10 synthetic CoNLL-U entries with a single root (head=0) and valid head values per sentence, saved to 'synthetic_conllu.conllu'.")


import random

# --- Expanded Vocabulary (merged from both versions) ---
HP_TITLE_M = ['ප්‍රධාණියා', 'ප්‍රධානි', 'තෙරුණ්', 'උපාසක', 'ගෘහපති', 'ස්වාමි', 'දෙවියන්', 'බ්‍රාහ්මණ']
HP_TITLE_F = ['උපාසිකා', 'භාර්යාව']
RESTRICTED_TITLES = ['තෙරුණ්', 'උපාසක', 'දෙවියන්', 'උපාසිකා', 'භික්ෂුණිය', 'භික්ෂුව', 'තෙරණුවන්', 'තේරුණ්']
TITLE_M = HP_TITLE_M #+ ['අචාර්ය', 'අයුෂ්මන්ත', 'අධ්‍යක්ෂක', 'අමාත්‍ය', 'ආචාර්ය', 'ඇමති', 'කම්මල්කරු', 'කුමාර', 'කෝෂ්ඨාගාරික', 'ගණකාධිකාරී', 'තෙරණුවන්', 'තේරුණ්', 'දේශක', 'පාලක', 'පියාණන්', 'භාණ්ඩාගාරික', 'භුක්තිවිඳින්නා', 'මහරජ', 'මහරජු', 'රජ', 'රජු', 'රන්කරු', 'වඩුවා', 'වෙළඳ', 'සෙන්පති']
TITLE_F = HP_TITLE_F #+ ['ප්‍රමුඛාවිය', 'භික්ෂුණිය', 'භික්ෂුව']
TITLES = TITLE_M + TITLE_F

HP_NAME_M = ['තිස්ස', 'සුමන', 'නාග', 'සොණ', 'සිව', 'ඵුස්ස', 'දත්ත', 'උත්තර', 'සුමනගුත්ත', 'උත්තිය', 'මහාතිස්ස', 'තිස්සගුත්ත']
NAME_M = HP_NAME_M # + ['අනුරාධ', 'අභය', 'ආනන්ද', 'උපරාජ', 'කචලි', 'කණ්හ', 'කුණිකතිස්ස', 'කුමාරදත්ත', 'ගාමණි', 'ගුත්ත', 'ගොතම', 'චලිතිස්ස', 'ජිත', 'දින්න', 'ධම්ම', 'නාගසේන', 'පින්ගල', 'පින්ගල සුමන', 'පුණ්ණ', 'බුද්ධරක්ඛිත', 'මහරෙත', 'මිත්ත', 'යසස්සි', 'රොහක', 'විශාක', 'සමණ']
NAME_F = ['අනුරාධි','උත්තියා', 'තිස්සා', 'ගාමණී','දත්තා','නාගා','පරිජා','මාලි','රෙවා', 'වෙසිලි', 'සාමිකා']
NAMES = NAME_M + NAME_F

HP_RELATION_M = ['පුත්', 'කුමරු', 'ශිෂ්‍ය']
RELATION_M = HP_RELATION_M #+ ["දියණිය", "සහෝදරයා", 'පියාණන්']
RELATION_F = ["භාර්යාව", 'කුමරිය', 'සොයුරිය']
RELATIONS = RELATION_M + RELATION_F

PLACE_NAME = ['මනාපදස්සන', 'කණිකණටිය', 'චලල', 'ශුදස්සන']
VERBS = ["දෙන ලදී", "කරවන ලද", "පවරා දෙන ලදී"]
MODIFIERS = ["පැමිණියාවූත් නොපැමිණියාවූත්", ""]
ADJECTIVES = ["භක්තිමත්"]
SANGA = "සංඝයා"

# --- Input Sentence Patterns from Original ---
TEMPLATES = [
    "{t1} {n1} ලෙණ " + SANGA,
    "{t1} {n1} {p} ලෙණ",
    # "{t1} {n1} පියගැටපෙළ",
    "{n1} ලෙණ " + SANGA,
    "{t1} {n1} {r1} {t2} {n2} ලෙණ " + SANGA + " {v}",
    "{t1} {n1} {r1} {t2} {n2} ලෙණ",
    "{t1} {n1} {r1} {t2} {n2} ලෙණ " + SANGA,
    "{n1} {r1} {t1} {n2} ලෙණ " + SANGA,
    "{t1} {n1} ලෙණ {m1} " + SANGA,
    "{t1} {n1} ලෙණ {m1} සතරදෙස " + SANGA,
    "{t1} {n1} ලෙණ සතරදිග" + SANGA,
    "{t1} {n1} {r1} {n2} ලෙණ " + SANGA,
    "{t1} {n1} {r1} {n2} {t2} {r1} {n3} ලෙණ " + SANGA,
    "{t1} {n1} {r1} {t2} {n2} {r1} {n3} ලෙණ",
    "{t1} {n1} සහ {t2} {n2} {r1} {t2} {n3} ලෙණ " + SANGA,
    "{t1} {n1} {r1} {n2} ලෙණ",
    "භක්තිමත් {t1} {n1} ලෙණ",
    "භක්තිමත් {n1} ලෙණ",                                    
]

# --- Sentence Filler ---
def fill_template(template):
    n1 = random.choice(NAMES)
    t1 = random.choice(TITLES)
    n2 = random.choice(NAMES)
    t2 = random.choice(TITLES)
    n3 = random.choice(NAMES)
    r1 = random.choice(RELATIONS)
    v = random.choice(VERBS)
    m1 = random.choice(MODIFIERS)
    p = random.choice(PLACE_NAME)

    return template.format(
        n1=n1, n2=n2, n3=n3, t1=t1, t2=t2,
        r1=r1, v=v, m1=m1, p=p
    ).strip()

# --- Sentence Generator ---
def generate_sentence():
    pattern = random.choice(TEMPLATES)
    return fill_template(pattern)

# --- CoNLL-U Generator ---
def generate_conllu_entry(sentence):
    tokens = sentence.split()
    conllu_lines = [f"# text = {sentence}"]
    root_id = next((i for i, tok in enumerate(tokens, 1) if tok == "ලෙණ"), len(tokens))

    for i, token in enumerate(tokens, 1):
        lemma = token

        # POS tagging
        if token in ADJECTIVES:
            pos = "ADJ"
        elif token in VERBS:
            pos = "VERB"
        elif token in NAMES:
            pos = "PROPN"
        elif token in [SANGA, "ලෙණ", "පියගැටපෙළ"] or token in RELATIONS:
            pos = "NOUN"
        elif token in PLACE_NAME:
            pos = "PROPN"
        elif token == "වන":
            pos = "ADP"
        else:
            pos = "NOUN"

        # XPOS tagging
        if token in TITLE_F:
            xpos = "TITLE-FEM"
        elif token in TITLE_M:
            xpos = "TITLE-MASC"
        elif token in ADJECTIVES:
            xpos = "ADJ"
        elif token in PLACE_NAME:
            xpos = "N-LOC"
        else:
            xpos = "_"

        # Dependency tagging
        if token == "ලෙණ":
            dep = "root"
            head = 0
        elif token in ADJECTIVES:
            dep = "amod"
            head = i + 1 if i < len(tokens) else root_id
        elif token == SANGA:
            dep = "obl"
            head = root_id
        elif token in VERBS:
            dep = "acl"
            head = root_id
        elif token in RELATIONS:
            dep = "compound"
            head = i + 1 if i < len(tokens) and tokens[i] in NAMES else root_id
        elif token == "වන":
            dep = "case"
            head = i - 1 if i > 1 else root_id
        elif token in TITLES:
            dep = "appos"
            head = i + 1 if i < len(tokens) and tokens[i] in NAMES else root_id
        elif token in NAMES or token in PLACE_NAME:
            dep = "nmod"
            head = root_id
        else:
            dep = "dep"
            head = root_id

        if head != 0 and (head < 1 or head > len(tokens)):
            head = root_id

        conllu_lines.append(f"{i}\t{token}\t{lemma}\t{pos}\t{xpos}\t_\t{head}\t{dep}\t_\t_")

    conllu_lines.append("")
    return "\n".join(conllu_lines)

# --- Dataset Generator ---
def generate_dataset(num_entries):
    dataset = []
    for _ in range(num_entries):
        sentence = generate_sentence()
        conllu = generate_conllu_entry(sentence)
        dataset.append(conllu)
    return "\n".join(dataset)

# --- Generate and Save ---
output = generate_dataset(5000)

with open("synthetic_conllu_new_new.conllu", "w", encoding="utf-8") as f:
    f.write(output)

print("✅ Generated 5000 CoNLL-U entries using enriched vocabulary and sentence patterns.")
