import re

def convert_conllu_to_text(input_file, output_file):
    sentences = []
    current_sentence = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.startswith('# text = '):
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                    current_sentence = []
            elif line.strip() and not line.startswith('#'):
                word = line.split('\t')[1]
                current_sentence.append(word)
        if current_sentence:
            sentences.append(' '.join(current_sentence))
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for sentence in sentences:
            outfile.write(sentence + '\n')

if __name__ == "__main__":
    convert_conllu_to_text('si_custom.train.in.conllu', 'train.txt')
    convert_conllu_to_text('si_custom.dev.in.conllu', 'dev.txt')
    convert_conllu_to_text('si_custom.test.in.conllu', 'test.txt')