import random
from typing import List, Tuple

def read_conllu(file_path: str) -> List[List[str]]:
    """
    Read a CoNLL-U file and return a list of sentences, where each sentence is a list of lines.
    Each sentence includes comment lines (e.g., # text) and token lines.
    """
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                current_sentence.append(line)
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def split_conllu(input_file: str, train_ratio: float = 0.8, dev_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Split a CoNLL-U file into training, dev, and test sets.
    
    Args:
        input_file: Path to the input CoNLL-U file.
        train_ratio: Proportion of data for training (default: 0.8).
        dev_ratio: Proportion of data for dev (default: 0.1).
        test_ratio: Proportion of data for test (default: 0.1).
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_sentences, dev_sentences, test_sentences), where each is a list of sentences.
    """
    # Validate ratios
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Read sentences from the input file
    sentences = read_conllu(input_file)
    total_sentences = len(sentences)
    
    # Calculate sizes for each split
    train_size = int(total_sentences * train_ratio)
    dev_size = int(total_sentences * dev_ratio)
    test_size = total_sentences - train_size - dev_size  # Ensure all sentences are used
    
    # Shuffle sentences
    random.seed(seed)
    random.shuffle(sentences)
    
    # Split sentences
    train_sentences = sentences[:train_size]
    dev_sentences = sentences[train_size:train_size + dev_size]
    test_sentences = sentences[train_size + dev_size:]
    
    return train_sentences, dev_sentences, test_sentences

def write_conllu(sentences: List[List[str]], output_file: str):
    """
    Write a list of sentences to a CoNLL-U file.
    
    Args:
        sentences: List of sentences, where each sentence is a list of lines.
        output_file: Path to the output CoNLL-U file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for line in sentence:
                f.write(line + '\n')
            f.write('\n')  # Empty line between sentences

def main():
    # Configuration
    input_file = "synthetic_conllu.conllu"
    train_file = "si_custom.train.in.conllu"
    dev_file = "si_custom.dev.in.conllu"
    test_file = "si_custom.test.in.conllu"
    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1
    seed = 42
    
    # Split the CoNLL-U file
    train_sentences, dev_sentences, test_sentences = split_conllu(
        input_file, train_ratio, dev_ratio, test_ratio, seed
    )
    
    # Write the splits to files
    write_conllu(train_sentences, train_file)
    write_conllu(dev_sentences, dev_file)
    write_conllu(test_sentences, test_file)
    
    # Print summary
    print(f"Total sentences: {len(train_sentences) + len(dev_sentences) + len(test_sentences)}")
    print(f"Training sentences: {len(train_sentences)} ({train_ratio*100:.1f}%)")
    print(f"Dev sentences: {len(dev_sentences)} ({dev_ratio*100:.1f}%)")
    print(f"Test sentences: {len(test_sentences)} ({test_ratio*100:.1f}%)")
    print(f"Output written to: {train_file}, {dev_file}, {test_file}")

if __name__ == "__main__":
    main()