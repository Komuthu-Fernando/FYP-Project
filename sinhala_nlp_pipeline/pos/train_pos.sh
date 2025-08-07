#!/bin/bash
CURRENT_DIR=$(dirname "$(realpath "$0")")
STANZA_DIR="/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/stanza"
POS_DIR="$CURRENT_DIR/../data/pos"
TRAIN_FILE="$POS_DIR/si_custom.train.in.conllu"
DEV_FILE="$POS_DIR/si_custom.dev.in.conllu"
TEST_FILE="$POS_DIR/si_custom.test.in.conllu"
MODEL_DIR="$CURRENT_DIR/../models"

mkdir -p "$MODEL_DIR"
python3 "$STANZA_DIR/stanza/utils/training/run_pos.py" --train_file "$TRAIN_FILE" --eval_file "$DEV_FILE" --test_file "$TEST_FILE" --output_dir "$MODEL_DIR" --lang si --shorthand si_custom --mode train --max_steps 1000 --wordvec_file "$MODEL_DIR/cc.si.300.vec"