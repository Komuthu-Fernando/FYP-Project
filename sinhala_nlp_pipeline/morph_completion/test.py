# import torch

# print("CUDA Available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))

# import nltk
# nltk.download('punkt_tab')

import torch
print(torch.__version__)  # Should show 2.7.0+cu121 or higher
print(torch.cuda.is_available())  # Should show True
print(torch.version.cuda)  # Should show 12.1 or 12.3
print(torch.cuda.get_device_name(0))  # Should show "NVIDIA GeForce RTX 3070"