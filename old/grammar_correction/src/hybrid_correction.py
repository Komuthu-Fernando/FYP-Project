from rule_based_correction import apply_rule_based_correction
from fine_tuned_llm import correct_with_llm

def hybrid_grammar_correction(sentence):
    """
    Applies rule-based corrections first, then refines the output using LLM.
    """
    rule_corrected = apply_rule_based_correction(sentence)
    final_correction = correct_with_llm(rule_corrected)
    return final_correction

if __name__ == "__main__":
    sample_sentence = "මම යන්නෙ පාසල"
    corrected_sentence = hybrid_grammar_correction(sample_sentence)
    print("Hybrid Correction:", corrected_sentence)
