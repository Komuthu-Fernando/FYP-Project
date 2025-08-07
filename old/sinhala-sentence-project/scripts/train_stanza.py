import stanza
from stanza.utils.conll import CoNLL
import os
import argparse
import shutil

# Check Stanza version
print(f"Stanza version: {stanza.__version__}")

def setup_training_paths():
    """Setup paths required for training"""
    # Base paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Create required directories
    os.makedirs(os.path.join(data_dir, 'pos'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'depparse'), exist_ok=True)
    
    # Setup paths dictionary
    paths = {
        'BASE_DIR': base_dir,
        'DATA_DIR': data_dir,
        'POS_DATA_DIR': os.path.join(data_dir, 'pos'),
        'DEPPARSE_DATA_DIR': os.path.join(data_dir, 'depparse'),
        'TRAIN_FILE': os.path.join(base_dir, 'data/raw/conllu_train.conllu'),
    }
    print("Paths dictionary:", paths)
    return paths

def prepare_training_data(train_file, output_dir, file_type):
    """Copy training file to appropriate directory with correct naming"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'si_test.{file_type}.train.in')
    shutil.copy2(train_file, output_file)
    return output_file

def train_stanza_model(train_file, model_dir):
    """Train a Stanza model for POS tagging and dependency parsing"""
    # Get paths
    paths = setup_training_paths()
    
    # Verify train file exists
    if not os.path.exists(paths['TRAIN_FILE']):
        raise FileNotFoundError(f"Training file not found: {paths['TRAIN_FILE']}")
    print(f"Training file: {paths['TRAIN_FILE']}")
    
    # Prepare data for training
    pos_train_file = prepare_training_data(paths['TRAIN_FILE'], paths['POS_DATA_DIR'], 'pos')
    depparse_train_file = prepare_training_data(paths['TRAIN_FILE'], paths['DEPPARSE_DATA_DIR'], 'depparse')
    
    # Create output directories
    pos_model_dir = os.path.join(model_dir, 'pos')
    depparse_model_dir = os.path.join(model_dir, 'depparse')
    os.makedirs(pos_model_dir, exist_ok=True)
    os.makedirs(depparse_model_dir, exist_ok=True)
    
    # Train POS tagger
    print("Training POS tagger...")
    pos_trainer = stanza.models.pos.trainer.Trainer(
        wordvec_dir=None,
        train_file=pos_train_file,
        eval_file=pos_train_file,  # Using same file for evaluation
        output_file=os.path.join(pos_model_dir, 'model.pt'),
        lang='si',
        mode='train'
    )
    pos_trainer.train()
    
    # Train dependency parser
    print("Training dependency parser...")
    depparse_trainer = stanza.models.depparse.trainer.Trainer(
        wordvec_dir=None,
        train_file=depparse_train_file,
        eval_file=depparse_train_file,  # Using same file for evaluation
        output_file=os.path.join(depparse_model_dir, 'model.pt'),
        lang='si',
        mode='train'
    )
    depparse_trainer.train()

# Run training
os.makedirs('models/stanza_model', exist_ok=True)
train_stanza_model('data/raw/conllu_train.conllu', 'models/stanza_model')