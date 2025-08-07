#!/bin/bash
CURRENT_DIR=$(dirname "$(realpath "$0")")
STANZA_DIR="/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/stanza"
DATA_DIR="$CURRENT_DIR/../data/lemma/si_custom"
MODEL_DIR="$CURRENT_DIR/../models"

mkdir -p "$MODEL_DIR"
python3 "$STANZA_DIR/stanza/utils/training/run_lemma.py" \
  --train_file "$DATA_DIR/train.in.conllu" \
  --eval_file "$DATA_DIR/dev.in.conllu" \
  --test_file "$DATA_DIR/test.in.conllu" \
  --gold_file "$DATA_DIR/dev.gold.conllu" \
  --output_dir "$MODEL_DIR" \
  --lang si \
  --shorthand si_custom \
  --mode train \