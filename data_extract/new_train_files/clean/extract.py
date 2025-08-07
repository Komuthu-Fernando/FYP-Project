import conllu

def extract_nouns_from_conllu(file_path, output_file):
    male_titles = []
    female_titles = []
    names = []
    places = []
    
    # Read the CoNLL-U file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # Parse the CoNLL-U data
    sentences = conllu.parse(data)
    
    # Process each sentence
    for sentence in sentences:
        for token in sentence:
            # Check for TITLE-MASC (male titles)
            if token['xpos'] and 'TITLE-MASC' in token['xpos']:
                if token['form'] not in male_titles:
                    male_titles.append(token['form'])
            # Check for TITLE-FEM (female titles)
            elif token['xpos'] and 'TITLE-FEM' in token['xpos']:
                if token['form'] not in female_titles:
                    female_titles.append(token['form'])
            # Check for PROPN (names)
            elif token['upos'] == 'PROPN':
                if token['form'] not in names:
                    names.append(token['form'])
            # Check for N-LOC (place names)
            elif token['xpos'] and 'N-LOC' in token['xpos']:
                if token['form'] not in places:
                    places.append(token['form'])
    
    # Sort the lists
    male_titles.sort()
    female_titles.sort()
    names.sort()
    places.sort()
    
    # Append to text file in the requested format
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n")  # Add a newline to separate from existing content
        f.write("male_titles = [")
        f.write(", ".join(f"'{title}'" for title in male_titles))
        f.write("]\n")
        f.write("female_titles = [")
        f.write(", ".join(f"'{title}'" for title in female_titles))
        f.write("]\n")
        f.write("names = [")
        f.write(", ".join(f"'{name}'" for name in names))
        f.write("]\n")
        f.write("places = [")
        f.write(", ".join(f"'{place}'" for place in places))
        f.write("]\n")
    
    return male_titles, female_titles, names, places

# Example usage
if __name__ == "__main__":
    file_path = "all.conllu"
    output_file = "nouns_output.txt"
    male_titles, female_titles, names, places = extract_nouns_from_conllu(file_path, output_file)
    print(f"Results appended to {output_file}")