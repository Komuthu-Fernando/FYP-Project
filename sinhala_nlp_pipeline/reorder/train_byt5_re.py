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

# === Paths ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'byt5_reorder')
DATA_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'reorder_train.txt')
LOG_FILE = os.path.join(CURRENT_DIR, '..', 'logs', 'byt5_reorder_training_log.txt')

# === Logging ===
def log_to_file(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

# === Data Preparation ===
def prepare_data():
    data = []
    input_lengths, output_lengths = [], []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' → ' not in line:
                continue
            parts = line.split(' → ')
            input_parts = parts[0].split(' | ')
            if len(parts) != 2 or not input_parts[0].startswith("input:"):
                continue

            input_text = input_parts[0].replace('input: ', '').strip()
            pos = input_parts[1].replace('POS: ', '').strip() if len(input_parts) > 1 else ''
            xpos = input_parts[2].replace('XPOS: ', '').strip() if len(input_parts) > 2 else ''
            dep = input_parts[3].replace('Dep: ', '').strip() if len(input_parts) > 3 else ''
            full_input = f"{input_text} | POS: {pos} | XPOS: {xpos} | Dep: {dep}"
            output_text = parts[1].replace('output: ', '').strip()

            data.append({'input': full_input, 'output': output_text})
            input_lengths.append(len(full_input.split()))
            output_lengths.append(len(output_text.split()))

    random.shuffle(data)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f"Loaded {len(data)} total examples ({len(train_data)} train, {len(val_data)} val)")
    log_to_file(f"Loaded {len(data)} total examples ({len(train_data)} train, {len(val_data)} val)")
    log_to_file(f"Input len: mean={statistics.mean(input_lengths):.2f}, max={max(input_lengths)}, min={min(input_lengths)}")
    log_to_file(f"Output len: mean={statistics.mean(output_lengths):.2f}, max={max(output_lengths)}, min={min(output_lengths)}")

    return {'train': train_data, 'val': val_data}

# === Main Training Function ===
def train_byt5_reorder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    log_to_file(f"Using device: {device}")

    tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-base', use_safetensors=True)
    model = T5ForConditionalGeneration.from_pretrained('google/byt5-base', use_safetensors=True).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    dataset = prepare_data()
    train_data, val_data = dataset['train'], dataset['val']

    bleu_metric = evaluate.load('bleu')
    batch_size = 4
    gradient_accumulation_steps = 2
    epochs = 2
    best_val_loss = float('inf')
    patience, patience_counter = 3, 0

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log_to_file("Training started")

    def encode_batch(batch):
        inputs = [f"input: {item['input']} → output:" for item in batch]
        targets = [item['output'] for item in batch]
        input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        target_enc = tokenizer(targets, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        return {'input_ids': input_enc['input_ids'], 'labels': target_enc['input_ids']}

    def evaluate_bleu(data, batch_size=4):
        model.eval()
        predictions, references, sample_preds = [], [], []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                inputs = [f"input: {item['input']} → output:" for item in batch]
                input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
                outputs = model.generate(input_enc['input_ids'], max_length=256, num_beams=4, no_repeat_ngram_size=2)
                for j, output in enumerate(outputs):
                    pred = tokenizer.decode(output, skip_special_tokens=True).strip()
                    ref = batch[j]['output'].strip()
                    predictions.append(pred)
                    references.append([ref])
                    if i == 0 and j < 5:
                        sample_preds.append(f"Sample {j+1}: Input: {batch[j]['input']} → Pred: {pred} → Ref: {ref}")
        for s in sample_preds:
            log_to_file(s)
        return bleu_metric.compute(predictions=predictions, references=references, tokenizer=nltk.word_tokenize)['bleu']

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        optimizer.zero_grad()
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            encodings = encode_batch(batch)
            loss = model(**encodings).loss / gradient_accumulation_steps
            loss.backward()

            if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            total_train_loss += loss.item() * gradient_accumulation_steps
            print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")

        avg_train_loss = total_train_loss / (len(train_data) // batch_size + 1)
        log_to_file(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # === Validation ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                encodings = encode_batch(batch)
                total_val_loss += model(**encodings).loss.item()
        avg_val_loss = total_val_loss / (len(val_data) // batch_size + 1)
        log_to_file(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

        bleu = evaluate_bleu(val_data, batch_size)
        log_to_file(f"Epoch {epoch + 1}, Validation BLEU: {bleu:.4f}")

        scheduler.step()
        log_to_file(f"Epoch {epoch + 1}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            log_to_file(f"Best model saved to {MODEL_DIR}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_to_file("Early stopping triggered.")
                break

        model.train()

    if patience_counter < patience:
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        log_to_file("Final model saved.")

if __name__ == "__main__":
    nltk.download('punkt')
    train_byt5_reorder()
