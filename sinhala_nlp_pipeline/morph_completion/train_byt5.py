# from transformers import ByT5Tokenizer, T5ForConditionalGeneration
# import torch
# import os
# import random
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import StepLR
# from datetime import datetime
# import evaluate
# import nltk
# import statistics

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'byt5_sinhala')
# DATA_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'morph_train.txt')
# LOG_FILE = os.path.join(CURRENT_DIR, '..', 'logs', 'training_log.txt')

# def prepare_data():
#     """Prepare training and validation data."""
#     data = []
#     input_lengths = []
#     output_lengths = []
#     with open(DATA_FILE, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 parts = line.split(' → ')
#                 if len(parts) == 2 and parts[0].startswith('input:') and parts[1].startswith('output:'):
#                     input_text = parts[0].replace('input: ', '').strip()
#                     output_text = parts[1].replace('output: ', '').strip()
#                     data.append({'input': input_text, 'output': output_text})
#                     input_lengths.append(len(input_text.split()))
#                     output_lengths.append(len(output_text.split()))
#     random.shuffle(data)
#     train_size = int(0.8 * len(data))
#     train_data = data[:train_size]
#     val_data = data[train_size:]
#     print(f"Loaded {len(data)} total examples ({len(train_data)} train, {len(val_data)} val)")
#     log_to_file(f"Loaded {len(data)} total examples ({len(train_data)} train, {len(val_data)} val)")
#     # Log dataset statistics
#     log_to_file(f"Input length stats: mean={statistics.mean(input_lengths):.2f}, max={max(input_lengths)}, min={min(input_lengths)}")
#     log_to_file(f"Output length stats: mean={statistics.mean(output_lengths):.2f}, max={max(output_lengths)}, min={min(output_lengths)}")
#     return {'train': train_data, 'val': val_data}

# def log_to_file(message):
#     """Append message to log file with timestamp."""
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     with open(LOG_FILE, 'a', encoding='utf-8') as f:
#         f.write(f"[{timestamp}] {message}\n")

# def train_byt5():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     log_to_file(f"Starting training on device: {device}")

#     try:
#         from safetensors.torch import load_file
#         print("Using safetensors for model loading")
#     except ImportError:
#         print("safetensors not installed; using default torch loading (ensure PyTorch >= 2.6)")

#     tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-small', use_safetensors=True)
#     model = T5ForConditionalGeneration.from_pretrained('google/byt5-small', use_safetensors=True).to(device)

#     optimizer = AdamW(model.parameters(), lr=1e-4)
#     scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

#     dataset = prepare_data()
#     train_data = dataset['train']
#     val_data = dataset['val']

#     bleu_metric = evaluate.load('bleu')

#     def encode_batch(batch):
#         inputs = [f"input: {item['input']} → output:" for item in batch]
#         targets = [item['output'] for item in batch]
#         input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
#         target_enc = tokenizer(targets, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
#         return {'input_ids': input_enc['input_ids'], 'labels': target_enc['input_ids']}

#     def evaluate_bleu(data, batch_size=8):
#         model.eval()
#         predictions = []
#         references = []
#         with torch.no_grad():
#             for i in range(0, len(data), batch_size):
#                 batch = data[i:i + batch_size]
#                 inputs = [f"input: {item['input']} → output:" for item in batch]
#                 input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
#                 outputs = model.generate(
#                     input_ids=input_enc['input_ids'],
#                     max_length=128,
#                     num_beams=4,
#                     early_stopping=False  # Disabled to avoid premature truncation
#                 )
#                 for j, output in enumerate(outputs):
#                     pred = tokenizer.decode(output, skip_special_tokens=True).strip()
#                     ref = batch[j]['output'].strip()
#                     predictions.append(pred)
#                     references.append([ref])
#         bleu_score = bleu_metric.compute(predictions=predictions, references=references, tokenizer=nltk.word_tokenize)['bleu']
#         return bleu_score

#     batch_size = 8
#     epochs = 10  # Increased for better convergence
#     best_val_loss = float('inf')
#     patience = 3
#     patience_counter = 0

#     os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
#     log_to_file("Training started")

#     model.train()
#     for epoch in range(epochs):
#         total_train_loss = 0
#         for i in range(0, len(train_data), batch_size):
#             batch = train_data[i:i + batch_size]
#             encodings = encode_batch(batch)
#             outputs = model(**encodings)
#             loss = outputs.loss
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             optimizer.zero_grad()
#             total_train_loss += loss.item()
#             print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Train Loss: {loss.item()}")

#         avg_train_loss = total_train_loss / (len(train_data) // batch_size + 1)
#         log_to_file(f"Epoch {epoch + 1}, Average Train Loss: {avg_train_loss}")

#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for i in range(0, len(val_data), batch_size):
#                 batch = val_data[i:i + batch_size]
#                 encodings = encode_batch(batch)
#                 outputs = model(**encodings)
#                 total_val_loss += outputs.loss.item()
#         avg_val_loss = total_val_loss / (len(val_data) // batch_size + 1)
#         log_to_file(f"Epoch {epoch + 1}, Average Validation Loss: {avg_val_loss}")

#         bleu_score = evaluate_bleu(val_data, batch_size)
#         log_to_file(f"Epoch {epoch + 1}, Validation BLEU Score: {bleu_score}")

#         scheduler.step()
#         log_to_file(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             patience_counter = 0
#             model.save_pretrained(MODEL_DIR)
#             tokenizer.save_pretrained(MODEL_DIR)
#             log_to_file(f"New best model saved to {MODEL_DIR} with Val Loss: {best_val_loss}")
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 log_to_file(f"Early stopping triggered after {patience} epochs with no improvement")
#                 break

#         model.train()

#     if patience_counter < patience:
#         model.save_pretrained(MODEL_DIR)
#         tokenizer.save_pretrained(MODEL_DIR)
#         log_to_file(f"Final model saved to {MODEL_DIR}")

# if __name__ == "__main__":
#     nltk.download('punkt')
#     nltk.download('punkt_tab')
#     train_byt5()


# BYT5 Base model training script

from transformers import ByT5Tokenizer, T5ForConditionalGeneration
import torch
import os
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import evaluate
import nltk
import statistics

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'byt5_sinhala')
DATA_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'morph_train.txt')
LOG_FILE = os.path.join(CURRENT_DIR, '..', 'logs', 'training_log.txt')

def prepare_data():
    """Prepare training and validation data without filtering."""
    data = []
    input_lengths = []
    output_lengths = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' → ')
                if len(parts) == 2 and parts[0].startswith('input:') and parts[1].startswith('output:'):
                    input_text = parts[0].replace('input: ', '').strip()
                    output_text = parts[1].replace('output: ', '').strip()
                    data.append({'input': input_text, 'output': output_text})
                    input_lengths.append(len(input_text.split()))
                    output_lengths.append(len(output_text.split()))
    random.shuffle(data)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    print(f"Loaded {len(data)} total examples ({len(train_data)} train, {len(val_data)} val)")
    log_to_file(f"Loaded {len(data)} total examples ({len(train_data)} train, {len(val_data)} val)")
    log_to_file(f"Input length stats: mean={statistics.mean(input_lengths):.2f}, max={max(input_lengths)}, min={min(input_lengths)}")
    log_to_file(f"Output length stats: mean={statistics.mean(output_lengths):.2f}, max={max(output_lengths)}, min={min(output_lengths)}")
    return {'train': train_data, 'val': val_data}

def log_to_file(message):
    """Append message to log file with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

# def train_byt5():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     log_to_file(f"Starting training on device: {device}")

#     try:
#         from safetensors.torch import load_file
#         print("Using safetensors for model loading")
#     except ImportError:
#         print("safetensors not installed; using default torch loading (ensure PyTorch >= 2.6)")

#     tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-base', use_safetensors=True)
#     model = T5ForConditionalGeneration.from_pretrained('google/byt5-base', use_safetensors=True).to(device)

#     optimizer = AdamW(model.parameters(), lr=1e-4)
#     scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

#     dataset = prepare_data()
#     train_data = dataset['train']
#     val_data = dataset['val']

#     bleu_metric = evaluate.load('bleu')

#     def encode_batch(batch):
#         inputs = [f"input: {item['input']} → output:" for item in batch]
#         targets = [item['output'] for item in batch]
#         input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
#         target_enc = tokenizer(targets, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
#         return {'input_ids': input_enc['input_ids'], 'labels': target_enc['input_ids']}

#     def evaluate_bleu(data, batch_size=4):
#         model.eval()
#         predictions = []
#         references = []
#         sample_predictions = []
#         with torch.no_grad():
#             for i in range(0, len(data), batch_size):
#                 batch = data[i:i + batch_size]
#                 inputs = [f"input: {item['input']} → output:" for item in batch]
#                 input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
#                 outputs = model.generate(
#                     input_ids=input_enc['input_ids'],
#                     max_length=256,
#                     num_beams=4,
#                     no_repeat_ngram_size=2,
#                     do_sample=False
#                 )
#                 for j, output in enumerate(outputs):
#                     pred = tokenizer.decode(output, skip_special_tokens=True).strip()
#                     ref = batch[j]['output'].strip()
#                     predictions.append(pred)
#                     references.append([ref])
#                     if i == 0 and j < 5:
#                         sample_predictions.append(f"Sample {j+1}: Input: {batch[j]['input']} → Pred: {pred} → Ref: {ref}")
#         bleu_score = bleu_metric.compute(predictions=predictions, references=references, tokenizer=nltk.word_tokenize)['bleu']
#         for sample in sample_predictions:
#             log_to_file(sample)
#         return bleu_score

#     batch_size = 4
#     gradient_accumulation_steps = 2  # Effective batch size = 4 * 2 = 8
#     epochs = 5
#     best_val_loss = float('inf')
#     patience = 3
#     patience_counter = 0

#     os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
#     log_to_file("Training started")

#     model.train()
#     for epoch in range(epochs):
#         total_train_loss = 0
#         optimizer.zero_grad()
#         for i in range(0, len(train_data), batch_size):
#             batch = train_data[i:i + batch_size]
#             encodings = encode_batch(batch)
#             outputs = model(**encodings)
#             loss = outputs.loss / gradient_accumulation_steps
#             loss.backward()
#             if (i // batch_size + 1) % gradient_accumulation_steps == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#                 optimizer.zero_grad()
#             total_train_loss += loss.item() * gradient_accumulation_steps
#             print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Train Loss: {loss.item() * gradient_accumulation_steps}")

#         avg_train_loss = total_train_loss / (len(train_data) // batch_size + 1)
#         log_to_file(f"Epoch {epoch + 1}, Average Train Loss: {avg_train_loss}")

#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for i in range(0, len(val_data), batch_size):
#                 batch = val_data[i:i + batch_size]
#                 encodings = encode_batch(batch)
#                 outputs = model(**encodings)
#                 total_val_loss += outputs.loss.item()
#         avg_val_loss = total_val_loss / (len(val_data) // batch_size + 1)
#         log_to_file(f"Epoch {epoch + 1}, Average Validation Loss: {avg_val_loss}")

#         bleu_score = evaluate_bleu(val_data, batch_size)
#         log_to_file(f"Epoch {epoch + 1}, Validation BLEU Score: {bleu_score}")

#         scheduler.step()
#         log_to_file(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             patience_counter = 0
#             model.save_pretrained(MODEL_DIR)
#             tokenizer.save_pretrained(MODEL_DIR)
#             log_to_file(f"New best model saved to {MODEL_DIR} with Val Loss: {best_val_loss}")
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 log_to_file(f"Early stopping triggered after {patience} epochs with no improvement")
#                 break

#         model.train()

#     if patience_counter < patience:
#         model.save_pretrained(MODEL_DIR)
#         tokenizer.save_pretrained(MODEL_DIR)
#         log_to_file(f"Final model saved to {MODEL_DIR}")
def train_byt5(): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    log_to_file(f"Starting training on device: {device}")

    try:
        from safetensors.torch import load_file
        print("Using safetensors for model loading")
    except ImportError:
        print("safetensors not installed; using default torch loading")

    tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-base', use_safetensors=True)
    model = T5ForConditionalGeneration.from_pretrained('google/byt5-base', use_safetensors=True).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    dataset = prepare_data()
    train_data = dataset['train']

    def encode_batch(batch):
        inputs = [f"input: {item['input']} → output:" for item in batch]
        targets = [item['output'] for item in batch]
        input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        target_enc = tokenizer(targets, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        return {'input_ids': input_enc['input_ids'], 'labels': target_enc['input_ids']}

    batch_size = 4
    gradient_accumulation_steps = 2
    epochs = 5

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log_to_file("Training started")

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        optimizer.zero_grad()
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            encodings = encode_batch(batch)
            outputs = model(**encodings)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            total_train_loss += loss.item() * gradient_accumulation_steps
            print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Train Loss: {loss.item() * gradient_accumulation_steps}")

        avg_train_loss = total_train_loss / (len(train_data) // batch_size + 1)
        log_to_file(f"Epoch {epoch + 1}, Average Train Loss: {avg_train_loss}")
        scheduler.step()
        log_to_file(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    log_to_file(f"Final model saved to {MODEL_DIR}")


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    train_byt5()