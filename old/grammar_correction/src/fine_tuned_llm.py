from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load fine-tuned mT5 model
model_path = "../models/fine_tuned_mT5"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def correct_with_llm(sentence):
    """
    Uses the fine-tuned mT5 model to correct Sinhala grammar errors.
    """
    input_text = f"grammar correction: {sentence}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    sample_sentence = "මම යන්නෙ පාසල"
    corrected_sentence = correct_with_llm(sample_sentence)
    print("LLM Correction:", corrected_sentence)
