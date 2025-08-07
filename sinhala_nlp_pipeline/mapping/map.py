import pandas as pd

def map_words_to_meanings(word_list):
    # Read the cleaned Excel file
    df = pd.read_excel('dictionary.xlsx')
    
    # Create a dictionary for mapping Brhami Word to Meaning
    word_map = dict(zip(df['Brhami Word'], df['Meaning']))
    
    # Initialize lists for results and missing words
    meanings = []
    missing_words = []
    
    # Map each word to its meaning
    for word in word_list:
        if word in word_map:
            meanings.append(word_map[word])
        else:
            missing_words.append(word)
    
    # Check for missing words
    if missing_words:
        error_message = f"Error: The following words were not found in the Excel file: {', '.join(missing_words)}"
        with open('error_log.txt', 'w') as f:
            f.write(error_message)
        return None
    
    # Join meanings into a single sentence
    sentence = ' '.join(meanings)
    return sentence

# Example usage
word_list = ['තිශ', 'ශගශ', 'ලෙණෙ']  # Replace with your list of words
result = map_words_to_meanings(word_list)

if result:
    print(f"Sentence: {result}")
else:
    print("Mapping failed. Check error_log.txt for details.")