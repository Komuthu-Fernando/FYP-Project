import os

def convert_conllu_to_lemma_format(input_file, output_in, output_gold):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_in, 'w', encoding='utf-8') as f_in_out, \
         open(output_gold, 'w', encoding='utf-8') as f_gold_out:
        sentence = []
        for line in f_in:
            line = line.strip()
            if not line or line.startswith('#'):
                if sentence:
                    # Write .in.conllu (ID, FORM, LEMMA, with minimal structure)
                    for token in sentence:
                        lemma = token[2] if token[2] and token[2] != '_' else '_'
                        if lemma == '_':
                            print(f"Warning: Missing lemma for word '{token[1]}' in {input_file}")
                        f_in_out.write(f"{token[0]}\t{token[1]}\t{lemma}\t_\t_\t_\t0\troot\t_\t_\n")
                    f_in_out.write("\n")
                    # Write .gold.conllu (full CONLL-U with all fields)
                    for token in sentence:
                        f_gold_out.write(f"{token[0]}\t{token[1]}\t{token[2]}\t{token[3]}\t{token[4]}\t{token[5]}\t{token[6]}\t{token[7]}\t{token[8]}\t{token[9]}\n")
                    f_gold_out.write("\n")
                    sentence = []
            elif line:
                fields = line.split('\t')
                if len(fields) >= 10:  # Ensure all 10 columns are present
                    sentence.append(fields)
                else:
                    print(f"Warning: Invalid line in {input_file}: {line}")
        if sentence:
            # Write remaining sentence
            for token in sentence:
                lemma = token[2] if token[2] and token[2] != '_' else '_'
                if lemma == '_':
                    print(f"Warning: Missing lemma for word '{token[1]}' in {input_file}")
                f_in_out.write(f"{token[0]}\t{token[1]}\t{lemma}\t_\t_\t_\t0\troot\t_\t_\n")
            f_in_out.write("\n")
            for token in sentence:
                f_gold_out.write(f"{token[0]}\t{token[1]}\t{token[2]}\t{token[3]}\t{token[4]}\t{token[5]}\t{token[6]}\t{token[7]}\t{token[8]}\t{token[9]}\n")
            f_gold_out.write("\n")
    print(f"Converted {input_file} to {output_in} and {output_gold}")

# Paths
base_dir = "/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline"
data_dir = f"{base_dir}/data/lemma/si_custom"
os.makedirs(data_dir, exist_ok=True)

files = {
    "train": f"{base_dir}/data/train_enhanced.conllu",
    "dev": f"{base_dir}/data/dev_enhanced.conllu",
    "test": f"{base_dir}/data/test_enhanced.conllu"
}

for split, filepath in files.items():
    if os.path.exists(filepath):
        convert_conllu_to_lemma_format(
            filepath,
            f"{data_dir}/{split}.in.conllu",
            f"{data_dir}/{split}.gold.conllu"
        )
    else:
        print(f"Warning: {filepath} not found")