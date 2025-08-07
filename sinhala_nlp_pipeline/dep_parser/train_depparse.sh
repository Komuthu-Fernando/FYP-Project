#!/bin/bash
# Training script for Stanza dependency parser for Sinhala
CURRENT_DIR=$(dirname "$(realpath "$0")")
STANZA_DIR="/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/stanza"  # Path to your cloned Stanza directory
TRAIN_FILE="$CURRENT_DIR/../data/si_custom.train.in.conllu"
DEV_FILE="$CURRENT_DIR/../data/si_custom.dev.in.conllu"
TEST_FILE="$CURRENT_DIR/../data/si_custom.test.in.conllu"
MODEL_DIR="$CURRENT_DIR/../models"

# Ensure the model directory exists
mkdir -p "$MODEL_DIR"


python3 "$STANZA_DIR/stanza/utils/training/run_depparse.py" \
  --train_file "$TRAIN_FILE" \
  --eval_file "$DEV_FILE" \
  --test_file "$TEST_FILE" \
  --output_dir "$MODEL_DIR" \
  --lang si \
  --shorthand si_custom \
  --mode train \
  --batch_size 32 \
  --max_steps 5000 \
  --wordvec_file "$MODEL_DIR/cc.si.300.vec" \
  --output_file "$MODEL_DIR/si_custom_eval_predictions.conllu"

