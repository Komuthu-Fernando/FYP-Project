# import stanza
# import os
# import logging

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)

# # Define directories
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'models'))

# def test_pos_model(words, pretokenized=True):
#     try:
#         # Define paths
#         pos_model_path = os.path.join(MODEL_DIR, 'si_custom_nocharlm_tagger.pt')
#         tokenize_model_path = os.path.join(MODEL_DIR, 'si/tokenize/si_custom.pt')
#         wordvec_file = os.path.join(MODEL_DIR, 'cc.si.300.vec')  # Path to pre-trained word vectors
        
#         # Log paths for debugging
#         logging.debug(f"POS model path: {pos_model_path}, Exists: {os.path.exists(pos_model_path)}")
#         logging.debug(f"Tokenize model path: {tokenize_model_path}, Exists: {os.path.exists(tokenize_model_path)}")
#         logging.debug(f"Wordvec file path: {wordvec_file}, Exists: {os.path.exists(wordvec_file)}")
        
#         # Set Stanza logging level
#         logging.getLogger('stanza').setLevel(logging.DEBUG)

#         # Initialize the pipeline with pretrain path
#         nlp = stanza.Pipeline(
#             lang='si',
#             processors='tokenize,pos',
#             tokenize_pretokenized=pretokenized,
#             dir=MODEL_DIR,
#             package='si_custom',
#             pos_model_path=pos_model_path,
#             pos_pretrain_path=wordvec_file,  # Specify the pretrain file
#             download_method=stanza.DownloadMethod.REUSE_RESOURCES,
#             use_cache=False
#         )
#         logging.debug(f"Pipeline initialized successfully: {nlp is not None}")
        
#         # Process the input
#         if pretokenized:
#             pretokenized_input = '\n'.join(words)
#             doc = nlp(pretokenized_input)
#         else:
#             doc = nlp(' '.join(words))
        
#         logging.debug(f"Doc: {doc}")
#         pos_tags = []
#         for sent in doc.sentences:
#             for word in sent.words:
#                 pos_tags.append((word.text, word.upos))
#         return pos_tags
#     except Exception as e:
#         logging.error(f"POS testing failed: {e}")
#         import traceback
#         logging.error(f"Stack trace: {traceback.format_exc()}")
#         return None

# if __name__ == "__main__":
#     words = ['තිස්සා', 'පුජ', 'ලෙන', 'සන්ඝයා']
#     pos_results_pretokenized = test_pos_model(words, pretokenized=True)
#     print(f"Input words (pretokenized): {words}")
#     print(f"POS tags (pretokenized): {pos_results_pretokenized}")


import os
import stanza

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define model directory
# MODEL_DIR = '/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/sinhala_nlp_pipeline/models'
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models')

# Paths to models and pretrain file
# pos_model_path = os.path.join(MODEL_DIR, 'si_custom_nocharlm_tagger.pt')
pos_model_path = os.path.join(CURRENT_DIR, '..', 'models', 'si_custom_nocharlm_tagger.pt')
# pos_pretrain_path = os.path.join(MODEL_DIR, 'si_custom_pretrain.pt')
pos_pretrain_path = os.path.join(CURRENT_DIR, '..', 'models', 'si_custom_pretrain.pt')   # Updated to .pt file
pretokenized = ['බුද්ධරක්ඛිත', 'රජු', 'ලෙණ', 'සතරදෙස', 'පැමිණියාවූත්', 'නොපැමිණියාවූත්', 'සංඝයා'] # Your input words
   
      

# Initialize the pipeline
nlp = stanza.Pipeline(
    lang='si',
    processors='tokenize,pos',
    tokenize_pretokenized=pretokenized,
    dir=MODEL_DIR,
    package='si_custom',
    pos_model_path=pos_model_path,
    pos_pretrain_path=pos_pretrain_path,  # Now points to the serialized .pt file
    download_method=stanza.DownloadMethod.REUSE_RESOURCES,
    use_cache=False
)

# Process the input
doc = nlp(" ".join(pretokenized))
for sent in doc.sentences:
    for word in sent.words:
        print(f"Word: {word.text}, POS: {word.upos}, XPOS: {word.xpos}")