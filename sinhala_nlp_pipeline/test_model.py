# import torch
# import logging


# # Load the model
# model = torch.load('/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/models/si/pos/si_custom.pt')

# # Print top-level keys
# print(model.keys())  # Verify all top-level keys

# # Access the weight dictionary (use 'model' instead of 'state_dict')
# print(model['model'].keys())  # Verify all weight tensors

# # Optional: Print vocabulary and config for debugging
# print("Vocabulary:", model['vocab'])
# print("Config:", model['config'])

import torch
import logging
model = torch.load('/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/models/si/pos/si_custom.pt', weights_only=False)
logging.debug(f"Pretrain path in config: {model['config'].get('pretrain', 'Not set')}")
print(f"Pretrain path in config: {model['config'].get('pretrain', 'Not set')}")