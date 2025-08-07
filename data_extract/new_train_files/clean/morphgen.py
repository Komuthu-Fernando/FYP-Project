import random

# High-probability entries
HP_TITLE_M = ['ප්‍රධාණියා', 'ප්‍රධානි', 'තෙරුණ්', 'උපාසක', 'ගෘහපති', 'ස්වාමි', 'දෙවියන්', 'බ්‍රාහ්මණ']
HP_TITLE_F = ['උපාසිකා', 'භාර්යාව']
HP_RELATION_M = ['පුත්', 'කුමරු', 'ශිෂ්‍ය']
HP_NAME_M = ['තිස්ස', 'සුමන', 'නාග', 'සොණ', 'සිව', 'ඵුස්ස', 'දත්ත', 'උත්තර', 'සුමනගුත්ත', 'උත්තිය', 'මහාතිස්ස', 'තිස්සගුත්ත']
HP_PLACE_NAME = ['මනාපදස්සන']

# Restricted titles not allowed when relation is involved
RESTRICTED_TITLES = ['තෙරුණ්', 'උපාසක', 'දෙවියන්', 'උපාසිකා', 'භික්ෂුණිය', 'භික්ෂුව', 'තෙරණුවන්', 'තේරුණ්']

# Full lists
TITLE_M = HP_TITLE_M #+ ['අචාර්ය', 'අයුෂ්මන්ත', 'අධ්‍යක්ෂක', 'අමාත්‍ය', 'ආචාර්ය', 'ඇමති', 'කම්මල්කරු', 'කුමාර', 'කෝෂ්ඨාගාරික', 'ගණකාධිකාරී', 'තෙරණුවන්', 'තේරුණ්', 'දේශක', 'පාලක', 'පියාණන්', 'භාණ්ඩාගාරික', 'භුක්තිවිඳින්නා', 'මහරජ', 'මහරජු', 'රජ', 'රජු', 'රන්කරු', 'වඩුවා', 'වෙළඳ', 'සෙන්පති']
TITLE_F = HP_TITLE_F #+ ['ප්‍රමුඛාවිය', 'භික්ෂුණිය', 'භික්ෂුව']
NAME_M = HP_NAME_M #+ ['අනුරාධ', 'අභය', 'ආනන්ද', 'උපරාජ', 'කචලි', 'කණ්හ', 'කුණිකතිස්ස', 'කුමාරදත්ත', 'ගාමණි', 'ගුත්ත', 'ගොතම', 'චලිතිස්ස', 'ජිත', 'දින්න', 'ධම්ම', 'නාගසේන', 'පින්ගල', 'පින්ගල සුමන', 'පුණ්ණ', 'බුද්ධරක්ඛිත', 'මහරෙත', 'මිත්ත', 'යසස්සි', 'රොහක', 'විශාක', 'සමණ']
NAME_F = ['අනුරාධි','උත්තියා', 'තිස්සා', 'ගාමණී','දත්තා','නාගා','පරිජා','මාලි','රෙවා', 'වෙසිලි', 'සාමිකා']
RELATION_M = HP_RELATION_M #+ ["දියණිය", "සහෝදරයා", 'පියාණන්']
RELATION_F = ["භාර්යාව", 'කුමරිය', 'සොයුරිය']
VERB = ["දෙන ලදී", "කරවන ලද", "පවරා දෙන ලදී"]
MODIFIER = ["පැමිණියාවූත් නොපැමිණියාවූත්"]
SANGA = "සංඝයා"
PLACE_NAME = HP_PLACE_NAME #+ ['කණිකණටිය', 'චලල', 'ශුදස්සන']
ADJECTIVE = ['භක්තිමත්']

# Sentence Patterns
sentence_patterns = [
    ("{n1} {t1} ලෙණ " + SANGA, "{n1} න​ම් {t1}ගේ ලෙණ " + SANGA + "ට දෙන ලදී"),
    ("{n1} {t1} {p} ලෙණ", "{n1} න​ම් {t1}ගේ {p} න​ම් ලෙණ"),
    ("{n1} {t1} පියගැටපෙ​ළ", "{n1} න​ම් {t1}ගේ පියගැටපෙ​ළ"),
    ("{n1} ලෙණ " + SANGA, "{n1}ගේ ලෙණ " + SANGA + "ට දෙන ලදී"),
    # ("{n1} {t1} {r1} {n2} {t2} ලෙණ " + SANGA + " {v}", "{n1} නම් {t1}ගේ {r1} ව​න {n2} න​ම් {t2}ගේ ලෙණ " + SANGA + "ට {v}"),
    # ("{n1} {t1} {r1} {n2} {t2} ලෙණ", "{n1} න​ම් {t1}ගේ {r1} ව​න {n2} න​ම් {t2}ගේ ලෙණ"),
    ("{n1} {t1} {r1} {n2} {t2} ලෙණ " + SANGA, "{n1} න​ම් {t1}ගේ {r1} වන {n2} නම් {t2}ගේ ලෙණ " + SANGA + "ට දෙන ලදී"),
    # ("{n1} {r1} {t1} {n2} ලෙණ " + SANGA, "{n1}ගේ {r1} {n2} නම් {t1}ගේ ලෙණ " + SANGA + "ට දෙන ලදී"),
    ("{n1} {t1} ලෙණ {m1} " + SANGA, "{n1} නම් {t1}ගේ ලෙණ {m1} " + SANGA + "ට දෙන ලදී"),
    # ("{n1} {t1} ලෙණ සතරදෙස {m1} " + SANGA, "{n1} නම් {t1}ගේ ලෙණ සතරදෙසි​න් {m1} " + SANGA + "ට දෙන ලදී"),
    # ("{n1} {t1} ලෙණ සතරදිග" + SANGA, "{n1} නම් {t1}ගේ ලෙණ සතරදිග" + SANGA + "ට දෙන ලදී"),
    ("{n1} {t1} {r1} {n2} ලෙණ " + SANGA, "{n1} නම් {t1}​ගේ {r1} වන {n2}​ගේ ලෙණ " + SANGA + "ට දෙන ලදී"),
    # # new patterns
    # ("{n1} {t1} {r1} {n2} {t2} {r1} {n3} ලෙණ " + SANGA, "{n1} නම් {t1}​ගේ {r1} වන {n2}​ගේ {t2} නම් {r1} වන {n3}ගේ ලෙණ " + SANGA + "ට දෙන ලදී"),
    # ("{n1} {t1} {r1} {n2} {t2} {r1} {n3} ලෙණ ", "{n1} නම් {t1}​ගේ {r1} වන {n2}​ගේ {t2} නම් {r1} වන {n3}ගේ ලෙණ"),
    # ("{n1} {t1} {r1} {n2} {t2} ලෙණ ", "{n1} නම් {t1}​ගේ {r1} වන {n2} නම් {t2}ගේ ලෙණ"),
    # ("{n1} {t1} {r1} {n2} {t2} ලෙණ " + SANGA, "{n1} නම් {t1}​ගේ {r1} වන {n2} නම් {t2}ගේ ලෙණ" + SANGA + "ට දෙන ලදී"),
    # ("{n1} {t1} සහ {n2} {t2} {r1} {n3} {t2} ලෙණ " + SANGA, "{n1} න​ම් {t1}ගේ සහ {n2} නම් {t2}ගේ {r1} වන {n3} න​ම් {t2}ගේ ලෙණ " + SANGA + "ට දෙන ලදී"),
    # # ශිෂ්‍ය title should be appear only with තෙරුණ් title
    # ("{n1} {t1} {r1} {n2} ලෙණ", "{n1} න​ම් {t1}ගේ {r1} ව​න {n2}ගේ ලෙණ"),
    # ("භක්තිමත් {n1} {t1} ලෙණ", "භක්තිමත් {n1} න​ම් {t1}ගේ ලෙණ"),
    # ("භක්තිමත් {n1} {t1} ලෙණ", "භක්තිමත් {n1} න​ම් {t1}ගේ ලෙණ"),
    ("භක්තිමත් {n1} ලෙණ", "භක්තිමත් {n1}ගේ ලෙණ"),


]

def weighted_choice(high_prob_list, full_list, high_weight=0.7):
    return random.choice(high_prob_list if random.random() < high_weight else full_list)

def pick_title(gender='m', allow_restricted=True):
    titles = TITLE_M if gender == 'm' else TITLE_F
    hp_titles = HP_TITLE_M if gender == 'm' else HP_TITLE_F

    if not allow_restricted:
        titles = [t for t in titles if t not in RESTRICTED_TITLES]
        hp_titles = [t for t in hp_titles if t not in RESTRICTED_TITLES]

    return weighted_choice(hp_titles, titles)

def pick_name_and_title(gender='m', allow_restricted=True):
    if gender == 'm':
        return weighted_choice(HP_NAME_M, NAME_M), pick_title('m', allow_restricted)
    else:
        return random.choice(NAME_F), pick_title('f', allow_restricted)

def generate_sentences(n=100):
    sentences = []
    for _ in range(n):
        pattern, output_template = random.choice(sentence_patterns)
        includes_relation = "{r1}" in pattern

        # First person
        n1, t1 = pick_name_and_title('m', allow_restricted=not includes_relation)

        # Handle special relation-case constraint
        if includes_relation:
            # Pick a relation, but if it's 'ශිෂ්‍ය', force t1 to be 'තෙරුණ්'
            r1 = weighted_choice(HP_RELATION_M, RELATION_M)
            if r1 == "ශිෂ්‍ය":
                t1 = "තෙරුණ්"
        else:
            r1 = ""

        # Second person
        if "{n2}" in pattern:
            gender2 = random.choices(['m', 'f'], weights=[0.7, 0.3])[0]
            n2, t2 = pick_name_and_title(gender2, allow_restricted=not includes_relation)
        else:
            n2 = t2 = ""

        # Third person (if present)
        if pattern.count("{n1}") > 1 and "{n3}" in pattern:
            n3, _ = pick_name_and_title('m', allow_restricted=True)
        else:
            n3 = n1

        if "{t2}" not in pattern:
            t2 = t1
        if pattern.count("{t1}") > 1 and "{t2}" in pattern and t2 == t1:
            _, t2 = pick_name_and_title('m')

        v = random.choice(VERB)
        m1 = random.choice(MODIFIER)
        p = weighted_choice(HP_PLACE_NAME, PLACE_NAME)

        input_sent = pattern.format(n1=n1, n2=n2, n3=n3, t1=t1, t2=t2, r1=r1, v=v, m1=m1, p=p)
        output_sent = output_template.format(n1=n1, n2=n2, n3=n3, t1=t1, t2=t2, r1=r1, v=v, m1=m1, p=p)

        sentences.append(f"input: {input_sent} → output: {output_sent}\n")

    return sentences

# Generate and save
sentences = generate_sentences(5000)
with open("newmophtrain.txt", "w", encoding='utf-8') as f:
    f.writelines(sentences)

print("✅ 5000 sentences generated. Check 'generated_sentences.txt'")
