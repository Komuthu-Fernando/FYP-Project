import re

def conllu_to_sentences_data(file_path, output_file):
    """
    Convert a CoNLL-U dataset file to a sentences_data list format and save to a text file
    in the format of a Python list of lists of tuples: (id, form, lemma, pos, features).
    
    Args:
        file_path (str): Path to the CoNLL-U file.
        output_file (str): Path to the output text file.
    
    Returns:
        list: List of sentences, where each sentence is a list of tuples (id, form, lemma, pos, features).
    """
    sentences_data = []
    current_sentence = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines (sentence boundaries)
                if not line:
                    if current_sentence:
                        sentences_data.append(current_sentence)
                        current_sentence = []
                    continue
                
                # Skip comment lines (e.g., # text, # sent_id)
                if line.startswith("#"):
                    continue
                
                # Parse token line
                fields = line.split("\t")
                if len(fields) >= 6:  # Ensure valid token line
                    token_id = fields[0]
                    form = fields[1]
                    lemma = fields[2]
                    pos = fields[3]
                    # Features are in column 5, use "_" if empty
                    features = fields[5] if fields[5] != "_" else "_"
                    
                    # Create tuple for the token
                    token_tuple = (token_id, form, lemma, pos, features)
                    current_sentence.append(token_tuple)
                else:
                    print(f"Warning: Skipping invalid line in {file_path}: {line}")
        
        # Append the last sentence if it exists
        if current_sentence:
            sentences_data.append(current_sentence)
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except UnicodeDecodeError:
        print(f"Error: Unable to decode {file_path} with UTF-8 encoding.")
        return []
    except Exception as e:
        print(f"Error: An unexpected error occurred while processing {file_path}: {e}")
        return []
    
    # Save the sentences_data to a text file in the specified Python list format
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("sentences_data = [\n")
            for i, sentence in enumerate(sentences_data):
                f.write("    [\n")
                for token in sentence:
                    token_id, form, lemma, pos, features = token
                    f.write(f'        ("{token_id}", "{form}", "{lemma}", "{pos}", "{features}"),\n')
                f.write("    ]")
                if i < len(sentences_data) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]\n")
        print(f"Output successfully saved to {output_file}")
    except Exception as e:
        print(f"Error: Failed to write to {output_file}: {e}")
    
    return sentences_data

# Example usage with a real file
if __name__ == "__main__":
    # Specify the path to your CoNLL-U file and output text file
    input_file = "si_custom.dev.in.conllu"  # Replace with your actual CoNLL-U file path
    output_file = "sentences_data_output.txt"  # Output file path
    
    # Convert the CoNLL-U file to sentences_data format and save to text file
    sentences_data = conllu_to_sentences_data(input_file, output_file)
    
    # Optionally print the result to console for verification
    if sentences_data:
        print("sentences_data = [")
        for i, sentence in enumerate(sentences_data):
            print("    [")
            for token in sentence:
                print(f"        {token},")
            print("    ]" + ("," if i < len(sentences_data) - 1 else ""))
        print("]")