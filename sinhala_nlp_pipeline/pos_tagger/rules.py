import re

# Enhanced regex rules for Sinhala POS tagging
RULES = [
    (re.compile(r'.*ට$'), 'N-DAT'),          # Dative case nouns (e.g., සංඝයාට)
    (re.compile(r'.*ය$'), 'N'),               # Singular nouns (e.g., සංඝයා)
    (re.compile(r'.*වා$'), 'V'),              # Verbs (e.g., දෙනවා)
    (re.compile(r'.*න$'), 'V'),               # Verb roots (e.g., දෙන)
    (re.compile(r'මම|ඔබ|අපි|ඇය|එය|ඔහු'), 'PRON'),  # Pronouns
    (re.compile(r'.*යි$'), 'ADJ'),            # Adjectives (e.g., හොඳයි)
    (re.compile(r'.*ම$'), 'ADJ'),             # Adjectives (e.g., ලස්සනම)
    (re.compile(r'පුජ|පූජා|දෙනු|ලදි'), 'V'),  # Specific verbs
    (re.compile(r'ගෘහපති|ප්‍රධානී|භාන්ඩාගාරික|තෙරුන්'), 'TITLE'),  # Titles/roles
    (re.compile(r'සහ|හා|ද'), 'CONJ'),         # Conjunctions
]