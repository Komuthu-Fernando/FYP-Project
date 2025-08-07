import re

# RULES = [
#     (re.compile(r'.*ට$'), 'N'),          # Dative case nouns 
#     (re.compile(r'.*ය$'), 'N'),          # Some singular nouns 
#     (re.compile(r'.*වā$'), 'V'),         # Verbs 
#     (re.compile(r'.*න$'), 'V'),          # Verb roots 
#     (re.compile(r'මම|ඔබ|අපි|ඇය|එය'), 'PRON'),  # Pronouns
#     (re.compile(r'.*යි$'), 'ADJ'),       # Adjectives 
#     (re.compile(r'.*ම$'), 'ADJ'),        # Adjectives 
# ]


RULES = [
    (re.compile(r'.*ට$'), 'N-DAT'),
    (re.compile(r'.*ය$'), 'N'),
    (re.compile(r'.*වා$'), 'V'),
    (re.compile(r'.*න$'), 'V'),
    (re.compile(r'මම|ඔබ|අපි|ඇය|එය'), 'PRON'),
    (re.compile(r'.*යි$'), 'ADJ'),
    (re.compile(r'.*ම$'), 'ADJ'),
    (re.compile(r'පුජා$'), 'V'),
    (re.compile(r'ගෘහපති|ප්‍රධානී|භාන්ඩාගාරික|තෙරුන්'), 'TITLE'), 
    (re.compile(r'සහ'), 'CONJ'),            
]


# N: Common nouns 
# PROP: Proper names (personal names, e.g., "තිස්ස", "සුමන").
# PLACE: Place names 
# N-DAT: Dative nouns (e.g., "සංඝයාට").
# V: Verbs (e.g., "පූජා", "දෙනු").
# ADJ: Adjectives (e.g., "ප්‍රධානී").
# PRON: Pronouns 
# CONJ: Conjunctions (e.g., "සහ").
# TITLE: Titles or roles (e.g., "ගෘහපති", "තෙරුන්").
