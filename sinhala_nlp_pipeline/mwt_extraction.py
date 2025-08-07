import json
from pathlib import Path

def extract_mwt_from_conllu(conllu_path):
    mwt_data = {}
    with open(conllu_path, 'r', encoding='utf-8') as f:
        current_mwt = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if current_mwt:
                    mwt_data[current_mwt[0]['id']] = current_mwt
                    current_mwt = []
                continue
            fields = line.split('\t')
            if len(fields) >= 10 and '-' in fields[0]:  # MWT range (e.g., 2-3)
                start, end = map(int, fields[0].split('-'))
                token = {'id': fields[0], 'form': fields[1], 'lemma': fields[2], 'upos': fields[3]}
                current_mwt.append(token)
            elif current_mwt:  # End of MWT block
                mwt_data[current_mwt[0]['id']] = current_mwt
                current_mwt = []
    if current_mwt:
        mwt_data[current_mwt[0]['id']] = current_mwt
    return mwt_data

# Paths
tokenize_dir = "/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/data/tokenize"
conllu_files = {
    "train": "si_custom.train.gold.conllu",
    "dev": "si_custom.dev.gold.conllu",
    "test": "si_custom.test.gold.conllu"  # Replace with actual test file if available
}

# Extract and save MWT data
for split, conllu_file in conllu_files.items():
    full_path = Path(tokenize_dir) / conllu_file
    if full_path.exists():
        mwt_data = extract_mwt_from_conllu(full_path)
        output_path = Path(tokenize_dir) / f"si_custom-ud-{split}-mwt.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mwt_data, f, ensure_ascii=False, indent=2)
        print(f"Generated {output_path}")
    else:
        print(f"Warning: {full_path} not found, skipping MWT extraction for {split}")

if __name__ == "__main__":
    pass  # Run manually or via command line