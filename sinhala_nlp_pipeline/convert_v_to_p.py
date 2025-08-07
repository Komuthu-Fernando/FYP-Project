from stanza.stanza.models.common.pretrain import Pretrain
import os

# Define the paths
wordvec_file = "/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/models/cc.si.300.vec"
pretrain_pt_file = "/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/models/si_custom_pretrain.pt"

# Correctly initialize Pretrain with 'filename' and 'vec_filename'
pretrain = Pretrain(filename=pretrain_pt_file, vec_filename=wordvec_file, max_vocab=250000)

# Optionally, print a confirmation message
print(f"Serialized pretrain file will be saved to: {pretrain_pt_file}")