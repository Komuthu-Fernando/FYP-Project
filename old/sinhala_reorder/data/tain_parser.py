import os
import json
import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.processor import ProcessorRequirementsException
from stanza.resources.common import DEFAULT_MODEL_DIR

def initialize_sinhala_resources(model_path):
    """Initialize Sinhala language resources configuration"""
    resources = {
        "si": {
            "lang": "si",
            "packages": {
                "default": {
                    "tokenize": {
                        "model_path": os.path.join(model_path, "tokenize", "default.pt"),
                        "config_path": os.path.join(model_path, "tokenize", "config.json")
                    },
                    "pos": {
                        "model_path": os.path.join(model_path, "pos", "default.pt"),
                        "config_path": os.path.join(model_path, "pos", "config.json")
                    },
                    "lemma": {
                        "model_path": os.path.join(model_path, "lemma", "default.pt"),
                        "config_path": os.path.join(model_path, "lemma", "config.json")
                    },
                    "depparse": {
                        "model_path": os.path.join(model_path, "depparse", "default.pt"),
                        "config_path": os.path.join(model_path, "depparse", "config.json")
                    }
                }
            },
            "default_processors": {
                "tokenize": "default",
                "pos": "default",
                "lemma": "default",
                "depparse": "default"
            },
            "default_dependencies": {
                "pos": ["tokenize"],
                "lemma": ["tokenize", "pos"],
                "depparse": ["tokenize", "pos", "lemma"]
            }
        }
    }
    
    # Create necessary directories
    for processor in ["tokenize", "pos", "lemma", "depparse"]:
        os.makedirs(os.path.join(model_path, processor), exist_ok=True)
        
        # Create default config files
        config = {
            "model_path": os.path.join(model_path, processor, "default.pt"),
            "batch_size": 32
        }
        with open(os.path.join(model_path, processor, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
    
    # Save resources configuration
    resources_file = os.path.join(model_path, 'resources.json')
    with open(resources_file, 'w', encoding='utf-8') as f:
        json.dump(resources, f, indent=2)
    
    return resources_file

def train_dependency_parser(training_data_filename, model_save_dir):
    """
    Train a Stanza dependency parser using UD-formatted CoNLL-U data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))   
    training_data_path = os.path.join(script_dir, 'data', training_data_filename)
    model_save_path = os.path.join(script_dir, model_save_dir)

    # Ensure model directory exists
    os.makedirs(model_save_path, exist_ok=True)

    try:
        # Initialize Sinhala resources
        resources_file = initialize_sinhala_resources(model_save_path)
        
        # Create configuration for training
        config = {
            'lang': 'si',
            'processors': 'tokenize,pos,lemma,depparse',
            'dir': model_save_path,
            'use_gpu': False,
            'verbose': True,
            'tokenize_pretokenized': True  # Since our training data is already tokenized
        }

        print("Initializing Stanza pipeline for Sinhala...")
        nlp = stanza.Pipeline(**config)

        print(f"Training pipeline on {training_data_path}...")
        nlp.train(
            train_file=training_data_path,
            save_dir=model_save_path,
            ud_eval=False,
            max_steps=10000
        )
        
        print(f"Training completed. Models saved to {model_save_path}")
        
        # Test the trained model
        print("\nTesting the trained model...")
        test_text = "මම පොත කියවමි"
        doc = nlp(test_text)
        print(f"\nTest sentence: {test_text}")
        print("Dependency parse:")
        for sent in doc.sentences:
            for word in sent.words:
                print(f"{word.text}\t{word.upos}\t{word.deprel}\t{word.head}")
        
    except ProcessorRequirementsException as e:
        print(f"Error during training: {str(e)}")
        print("Make sure your training data is in valid CoNLL-U format and contains all required fields.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_dependency_parser('training_data.conllu', 'models/')
