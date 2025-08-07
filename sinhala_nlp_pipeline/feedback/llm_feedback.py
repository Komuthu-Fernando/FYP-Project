from pos_tagger.tagger import hybrid_tagger
from dep_parser.dep_parser import parse_dependencies
import logging

logging.basicConfig(level=logging.INFO)

def get_llm_feedback(sentence, tagged, dependencies, completed):
    prompt = f"""
Input: {' '.join(sentence)}
POS: {', '.join([f'{w} → {t}' for w, t in tagged])}
DepParse: {', '.join([f'{d["word"]} ({d["deprel"]} of {d["head"]})' for d in dependencies if dependencies])}
Completed: {' '.join(completed)}
Fix the sentence to be grammatically correct and meaningful in Sinhala.
"""
    # Simulate LLM feedback (replace with actual Grok or other LLM API call)
    logging.info("Simulating LLM feedback...")
    corrected = ['තිස්සාගේ', 'ලෙන', 'සන්ඝයාට', 'පූජා', 'කරන', 'ලදි']
    feedback = "Added possessive 'ගේ' to තිස්සා, dative 'ට' to සන්ඝයා, and adjusted verb form to past tense 'කරන ලදි'."
    return corrected, feedback

def save_feedback(sentence, tagged, dependencies, completed, corrected, feedback):
    with open('feedback_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"Input: {' '.join(sentence)}\n")
        f.write(f"Tagged: {tagged}\n")
        f.write(f"Dependencies: {dependencies}\n")
        f.write(f"Completed: {' '.join(completed)}\n")
        f.write(f"Corrected: {' '.join(corrected)}\n")
        f.write(f"Feedback: {feedback}\n\n")

if __name__ == "__main__":
    sentence = ['තිස්සා', 'පුජ', 'ලෙන', 'සන්ඝයා']
    tagged = hybrid_tagger(sentence)
    deps = parse_dependencies(tagged)
    completed = ['තිස්සා', 'ලෙන', 'සන්ඝයා', 'පුජ']
    corrected, feedback = get_llm_feedback(sentence, tagged, deps, completed)
    save_feedback(sentence, tagged, deps, completed, corrected, feedback)
    print(f"Corrected: {corrected}")
    print(f"Feedback: {feedback}")