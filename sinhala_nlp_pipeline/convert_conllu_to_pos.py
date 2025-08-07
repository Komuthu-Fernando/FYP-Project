import os

def convert_conllu_to_pos_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        sentence = []
        for line in f_in:
            line = line.strip()
            if not line or line.startswith('#'):
                if sentence:
                    for i, token in enumerate(sentence, 1):
                        # Set first token as root (HEAD = 0), others point to previous token
                        head = '0' if i == 1 else str(i - 1)
                        # Write 10 fields: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
                        f_out.write(f"{token[0]}\t{token[1]}\t{token[2]}\t{token[3]}\t_\t_\t{head}\t_\t_\t_\n")
                    f_out.write("\n")
                    sentence = []
            elif line:
                fields = line.split('\t')
                if len(fields) >= 4:  # Ensure minimum fields (ID, FORM, LEMMA, UPOS)
                    sentence.append(fields)
        if sentence:
            for i, token in enumerate(sentence, 1):
                head = '0' if i == 1 else str(i - 1)
                f_out.write(f"{token[0]}\t{token[1]}\t{token[2]}\t{token[3]}\t_\t_\t{head}\t_\t_\t_\n")
            f_out.write("\n")
    print(f"Converted {input_file} to {output_file} with 10-field format")

# Paths
base_dir = "/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline"
pos_dir = f"{base_dir}/data/pos"
os.makedirs(pos_dir, exist_ok=True)

files = {
    "train": f"{base_dir}/data/train_enhanced.conllu",
    "dev": f"{base_dir}/data/dev_enhanced.conllu",
    "test": f"{base_dir}/data/test_enhanced.conllu"
}

for split, filepath in files.items():
    if os.path.exists(filepath):
        convert_conllu_to_pos_format(filepath, f"{pos_dir}/si_custom.{split}.in.conllu")
        print(f"Converted {split} data to {pos_dir}/si_custom.{split}.in.conllu")
    else:
        print(f"Warning: {filepath} not found")