from pathlib import Path

# Define paths
input_file = Path("data/morph_train.txt")
output_file = Path("data/direct_translation.txt")

# Ensure the data directory exists
output_file.parent.mkdir(exist_ok=True)

# Function to extract input sentences
def extract_input_sentences(input_path, output_path):
    input_sentences = []
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "→" in line:
                    # Split on "→" and take the left part (input), remove "input:" and strip whitespace
                    input_part = line.split("→")[0].replace("input:", "").strip()
                    if input_part:
                        input_sentences.append(input_part)
        
        # Write input sentences to output file
        with open(output_path, "w", encoding="utf-8") as f:
            for sentence in input_sentences:
                f.write(sentence + "\n")
        
        print(f"Extracted {len(input_sentences)} input sentences and saved to {output_path}")
    except FileNotFoundError:
        print(f"Error: {input_path} not found!")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

# Execute the extraction
extract_input_sentences(input_file, output_file)