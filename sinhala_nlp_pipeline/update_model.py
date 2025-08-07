import torch
import logging
import os

logging.basicConfig(level=logging.DEBUG)

# Paths
pos_model_path = '/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/models/si/pos/si_custom.pt'

# Load model
model = torch.load(pos_model_path, weights_only=False)
logging.debug(f"Original pretrain path in config: {model['config'].get('pretrain', 'Not set')}")

# Set pretrain to None
model['config']['pretrain'] = None
logging.debug(f"Updated pretrain path in config: {model['config'].get('pretrain', 'Not set')}")

# Save updated model
torch.save(model, pos_model_path)
logging.debug("Model saved with pretrain set to None")