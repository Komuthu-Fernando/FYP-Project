import os
import subprocess
import stanza
from pathlib import Path

def download_dataset():
    print("Downloading UD_Sinhala-STB dataset...")
    if not os.path.exists("UD_Sinhala-STB"):
        subprocess.run(["git", "clone", "https://github.com/UniversalDependencies/UD_Sinhala-STB.git"])
    os.makedirs("data/si_custom", exist_ok=True)
    os.rename("UD_Sinhala-STB/si_stb-ud-train.conllu", "data/si_custom/train.conllu")
    os.rename("UD_Sinhala-STB/si_stb-ud-dev.conllu", "data/si_custom/dev.conllu")
    os.rename("UD_Sinhala-STB/si_stb-ud-test.conllu", "data/si_custom/test.conllu")

def run_training():
    print("\nTraining Tokenizer...")
    subprocess.run([
        "python", "-m", "stanza.utils.training.run_tokenizer",
        "depparse",
        "--shorthand", "si_custom",
        "--train_file", "data/si_custom/train.conllu",
        "--dev_file", "data/si_custom/dev.conllu",
        "--test_file", "data/si_custom/test.conllu",
        "--lang", "si"
    ])

    print("\nTraining POS Tagger and Lemmatizer...")
    subprocess.run([
        "python", "-m", "stanza.utils.training.run_pos",
        "--shorthand", "si_custom",
        "--train_file", "data/si_custom/train.conllu",
        "--dev_file", "data/si_custom/dev.conllu",
        "--test_file", "data/si_custom/test.conllu",
        "--lang", "si"
    ])

    print("\nTraining Dependency Parser...")
    subprocess.run([
        "python", "-m", "stanza.utils.training.run_depparse",
        "--shorthand", "si_custom",
        "--train_file", "data/si_custom/train.conllu",
        "--dev_file", "data/si_custom/dev.conllu",
        "--test_file", "data/si_custom/test.conllu",
        "--lang", "si"
    ])

def parse_sample_sentence():
    print("\nLoading custom Sinhala NLP pipeline...")
    nlp = stanza.Pipeline(
        lang='si',
        package='custom',
        dir='stanza_resources',
        processors='tokenize,pos,lemma,depparse'
    )

    sentence = "මම ගෙදරට යමි."
    doc = nlp(sentence)

    print("\nParsed Dependencies:")
    for sent in doc.sentences:
        for word in sent.words:
            print(f"{word.id}\t{word.text}\thead={word.head}\tdeprel={word.deprel}")

if __name__ == "__main__":
    download_dataset()
    run_training()
    parse_sample_sentence()
