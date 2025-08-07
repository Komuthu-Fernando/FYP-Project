from reorder.rules import basic_sov_reorder
from dep_parser.dep_parser import parse_dependencies
import logging

logging.basicConfig(level=logging.INFO)

def hybrid_reorder(tagged_sentence):
    dependencies = parse_dependencies(tagged_sentence)
    if not dependencies:
        logging.warning("Falling back to rule-based reordering.")
        reordered = basic_sov_reorder(tagged_sentence)
    else:
        reordered = basic_sov_reorder(tagged_sentence, dependencies)
    return reordered

if __name__ == "__main__":
    tagged_sentence = [('තිස්සා', 'PROP'), ('පුජ', 'V'), ('ලෙන', 'PLACE'), ('සන්ඝයා', 'N')]
    reordered = hybrid_reorder(tagged_sentence)
    print(f"Tagged: {tagged_sentence}")
    print(f"Reordered: {reordered}")