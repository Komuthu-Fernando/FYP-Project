from pos_tagger.tagger import hybrid_tagger
from reorder.run_reorder import hybrid_reorder
from morph_completion.morph_complete import morph_complete
from feedback.llm_feedback import get_llm_feedback, save_feedback
import logging

logging.basicConfig(level=logging.INFO)

def sinhala_nlp_pipeline(sentence):
    # Step 1: POS Tagging
    tagged = hybrid_tagger(sentence)
    logging.info(f"POS Tagged: {tagged}")
    
    # Step 2 & 3: Dependency Parsing (used in reordering)
    reordered = hybrid_reorder(tagged)
    logging.info(f"Reordered: {reordered}")
    
    # Step 4: Morphological Completion
    # completed = morph_complete(reordered)
    # logging.info(f"Morphologically Completed: {completed}")
    
    # Step 5: LLM Feedback
    # deps = parse_dependencies(tagged)
    # corrected, feedback = get_llm_feedback(sentence, tagged, deps, completed)
    # save_feedback(sentence, tagged, deps, completed, corrected, feedback)
    # logging.info(f"Corrected: {corrected}")
    # logging.info(f"Feedback: {feedback}")
    
    return reordered

if __name__ == "__main__":
    sentence = ['තිස්සා', 'පුජ', 'ලෙන', 'සන්ඝයා']
    result = sinhala_nlp_pipeline(sentence)
    print(result)
    # print(f"Final Output: {' '.join(result)}")