# # Script to extract the 185th sentence from a CoNLL-U file
# def extract_sentence(file_path, sentence_index):
#     sentences = []
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if line.startswith('# text ='):
#                     # Extract the sentence after '# text = '
#                     sentence = line[len('# text = '):].strip()
#                     sentences.append(sentence)
        
#         # Check if the requested index is valid
#         if sentence_index < 1 or sentence_index > len(sentences):
#             return f"Error: The file contains only {len(sentences)} sentences, but you requested sentence {sentence_index}."
        
#         # Return the sentence at the specified index (adjusting for 1-based indexing)
#         return sentences[sentence_index - 1]
    
#     except FileNotFoundError:
#         return "Error: The file 'synthetic_conllu.conllu' was not found."
#     except Exception as e:
#         return f"Error: An unexpected error occurred: {str(e)}"

# # Specify the file path and desired sentence index
# file_path = "si_custom.train.in.conllu"
# sentence_index = 185

# # Extract and print the 185th sentence
# result = extract_sentence(file_path, sentence_index)
# print(f"Sentence {sentence_index}: {result}")


import conllu

# Path to your CoNLL-U training file
input_file = "/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/data/si_custom.train.in.conllu"

# Parse the CoNLL-U file
with open(input_file, "r", encoding="utf-8") as f:
    data = f.read()
    sentences = conllu.parse(data)

# Iterate through sentences to find the one where token ID 3 has head 8
for sent_idx, sentence in enumerate(sentences, 1):
    for token in sentence:
        if token["id"] == 3 and token["head"] == 8:
            print(f"Found problematic sentence at index {sent_idx}:")
            print(f"Sentence text: {sentence.metadata.get('text', 'No text metadata')}")
            print("\nTokens in the sentence:")
            for t in sentence:
                print(t)
            print("\n")
            break