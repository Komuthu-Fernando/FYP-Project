# from transformers import MT5ForConditionalGeneration, MT5Tokenizer
# import torch
# import os
# import random
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import StepLR

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'mt5_reorder')
# DATA_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'reorder_train.txt')

# # def prepare_data():
# #     data = []
# #     with open(DATA_FILE, 'r', encoding='utf-8') as f:
# #         for line in f:
# #             line = line.strip()
# #             if line:
# #                 parts = line.split(' → ')
# #                 if len(parts) == 2:
# #                     input_parts = parts[0].split(' | ')
# #                     input_text = input_parts[0].replace('input: ', '').strip()
# #                     pos = input_parts[1].replace('POS: ', '').strip() if len(input_parts) > 1 else ''
# #                     dep = input_parts[2].replace('Dep: ', '').strip() if len(input_parts) > 2 else ''
# #                     output_text = parts[1].replace('output: ', '').strip()
# #                     data.append({
# #                         'input': input_text,
# #                         'pos': pos,
# #                         'dep': dep,
# #                         'output': output_text
# #                     })
# #     random.shuffle(data)
# #     train_size = int(0.8 * len(data))
# #     train_data = data[:train_size]
# #     val_data = data[train_size:]
# #     print(f"Loaded {len(data)} total examples ({len(train_data)} train, {len(val_data)} val)")
# #     return {'train': train_data, 'val': val_data}
# def prepare_data():
#     data = []
#     with open(DATA_FILE, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 parts = line.split(' → ')
#                 if len(parts) == 2:
#                     input_parts = parts[0].split(' | ')
#                     input_text = input_parts[0].replace('input: ', '').strip()
#                     pos = input_parts[1].replace('POS: ', '').strip() if len(input_parts) > 1 else ''
#                     xpos = input_parts[2].replace('XPOS: ', '').strip() if len(input_parts) > 2 else ''
#                     dep = input_parts[3].replace('Dep: ', '').strip() if len(input_parts) > 3 else ''
#                     output_text = parts[1].replace('output: ', '').strip()
#                     data.append({
#                         'input': input_text,
#                         'pos': pos,
#                         'xpos': xpos,
#                         'dep': dep,
#                         'output': output_text
#                     })
#     random.shuffle(data)
#     train_size = int(0.8 * len(data))
#     train_data = data[:train_size]
#     val_data = data[train_size:]
#     print(f"Loaded {len(data)} total examples ({len(train_data)} train, {len(val_data)} val)")
#     return {'train': train_data, 'val': val_data}

# def train_mt5():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
#     model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small').to(device)

#     optimizer = AdamW(model.parameters(), lr=5e-5)
#     scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

#     dataset = prepare_data()
#     train_data = dataset['train']
#     val_data = dataset['val']

#     def encode_batch(batch):
#         inputs = [f"reorder: {item['input']} | POS: {item['pos']} | Dep: {item['dep']}" for item in batch]
#         targets = [item['output'] for item in batch]
#         input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=50, return_tensors='pt').to(device)
#         target_enc = tokenizer(targets, padding=True, truncation=True, max_length=50, return_tensors='pt').to(device)
#         return {'input_ids': input_enc['input_ids'], 'labels': target_enc['input_ids']}

#     batch_size = 8
#     epochs = 5
#     best_val_loss = float('inf')
#     patience = 2
#     patience_counter = 0

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
#         print(f"Epoch {epoch + 1}, Average Train Loss: {avg_train_loss}")

#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for i in range(0, len(val_data), batch_size):
#                 batch = val_data[i:i + batch_size]
#                 encodings = encode_batch(batch)
#                 outputs = model(**encodings)
#                 total_val_loss += outputs.loss.item()
#         avg_val_loss = total_val_loss / (len(val_data) // batch_size + 1)
#         print(f"Epoch {epoch + 1}, Average Validation Loss: {avg_val_loss}")

#         scheduler.step()
#         print(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             patience_counter = 0
#             model.save_pretrained(MODEL_DIR)
#             tokenizer.save_pretrained(MODEL_DIR)
#             print(f"New best model saved to {MODEL_DIR} with Val Loss: {best_val_loss}")
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"Early stopping triggered after {patience} epochs with no improvement.")
#                 break

#         model.train()

#     if patience_counter < patience:
#         model.save_pretrained(MODEL_DIR)
#         tokenizer.save_pretrained(MODEL_DIR)
#         print(f"mT5 model saved to {MODEL_DIR}")

# if __name__ == "__main__":
#     train_mt5()


from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
import os
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'mt5_reorder')
DATA_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'reorder_train.txt')

def prepare_data():
    data = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' → ')
                if len(parts) == 2:
                    input_parts = parts[0].split(' | ')
                    input_text = input_parts[0].replace('input: ', '').strip()
                    pos = input_parts[1].replace('POS: ', '').strip() if len(input_parts) > 1 else ''
                    xpos = input_parts[2].replace('XPOS: ', '').strip() if len(input_parts) > 2 else ''
                    dep = input_parts[3].replace('Dep: ', '').strip() if len(input_parts) > 3 else ''
                    output_text = parts[1].replace('output: ', '').strip()
                    data.append({
                        'input': input_text,
                        'pos': pos,
                        'xpos': xpos,
                        'dep': dep,
                        'output': output_text
                    })
    random.shuffle(data)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    print(f"Loaded {len(data)} total examples ({len(train_data)} train, {len(val_data)} val)")
    return {'train': train_data, 'val': val_data}

def train_mt5():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small').to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    dataset = prepare_data()
    train_data = dataset['train']
    val_data = dataset['val']

    def encode_batch(batch):
        inputs = [f"reorder: {item['input']} | POS: {item['pos']} | XPOS: {item['xpos']} | Dep: {item['dep']}" for item in batch]
        targets = [item['output'] for item in batch]
        input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=50, return_tensors='pt').to(device)
        target_enc = tokenizer(targets, padding=True, truncation=True, max_length=50, return_tensors='pt').to(device)
        return {'input_ids': input_enc['input_ids'], 'labels': target_enc['input_ids']}

    batch_size = 8
    epochs = 10  
    best_val_loss = float('inf')
    patience = 3  
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            encodings = encode_batch(batch)
            outputs = model(**encodings)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
            print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Train Loss: {loss.item()}")

        avg_train_loss = total_train_loss / (len(train_data) // batch_size + 1)
        print(f"Epoch {epoch + 1}, Average Train Loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                encodings = encode_batch(batch)
                outputs = model(**encodings)
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / (len(val_data) // batch_size + 1)
        print(f"Epoch {epoch + 1}, Average Validation Loss: {avg_val_loss}")

        scheduler.step()
        print(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"New best model saved to {MODEL_DIR} with Val Loss: {best_val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

        model.train()

    if patience_counter < patience:
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"mT5 model saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_mt5()