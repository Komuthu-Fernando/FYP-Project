import json
import re

RULES_FILE = "/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/grammar_correction/data/rules.json"

# Load grammar rules from JSON file
with open(RULES_FILE, "r", encoding="utf-8") as file:
    RULES = json.load(file)

def apply_rule_based_correction(sentence):
    """
    Applies predefined grammar correction rules to a Sinhala sentence.
    """
    for rule in RULES:
        sentence = re.sub(rule["pattern"], rule["replacement"], sentence)
    return sentence

if __name__ == "__main__":
    sample_sentence = "මම යන්නෙ පාසල"
    corrected_sentence = apply_rule_based_correction(sample_sentence)
    print("Rule-Based Correction:", corrected_sentence)
