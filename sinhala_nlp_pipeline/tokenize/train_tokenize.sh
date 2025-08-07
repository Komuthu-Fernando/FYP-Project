#!/bin/bash
CURRENT_DIR=$(dirname "$(realpath "$0")")
STANZA_DIR="/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/stanza"
TOKENIZE_DIR="$CURRENT_DIR/../data/tokenize"
SAVE_DIR="$CURRENT_DIR/../models"

mkdir -p "$SAVE_DIR"
export TOKENIZE_DATA_DIR="$TOKENIZE_DIR"
python3 "$STANZA_DIR/stanza/utils/training/run_tokenizer.py" \
python3 "$STANZA_DIR/stanza/utils/training/run_tokenizer.py" \
  --label_file "$TOKENIZE_DIR/si_custom-ud-train.toklabels" \
  --txt_file "$TOKENIZE_DIR/si_custom.train.txt" \
  --dev_label_file "$TOKENIZE_DIR/si_custom-ud-dev.toklabels" \
  --dev_txt_file "$TOKENIZE_DIR/si_custom.dev.txt" \
  --dev_conll_gold "$TOKENIZE_DIR/si_custom.dev.gold.conllu" \
  --lang si \
  --shorthand si_custom \
  --mode train \
  --max_steps 1000 \
  --save_dir "$SAVE_DIR" \
  --save_name "si_custom_tokenizer.pt" \
  --max_seqlen 100