from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

# Load model and tokenizer
model_name = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)  # Use safetensors

# Load and encode data
def load_data(file_path):
    inputs, targets = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            inp, tgt = line.strip().split('\t')
            inputs.append(inp)
            targets.append(tgt)
    return inputs, targets

train_inputs, train_targets = load_data('data/processed/mt5_train_data.txt')
test_inputs, test_targets = load_data('data/processed/mt5_test_data.txt')

# Tokenize data
train_encodings = tokenizer(train_inputs, truncation=True, padding=True, max_length=50)
train_labels = tokenizer(train_targets, truncation=True, padding=True, max_length=50)
test_encodings = tokenizer(test_inputs, truncation=True, padding=True, max_length=50)
test_labels = tokenizer(test_targets, truncation=True, padding=True, max_length=50)

# Create dataset
class SinhalaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = SinhalaDataset(train_encodings, train_labels)
test_dataset = SinhalaDataset(test_encodings, test_labels)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='models/mt5_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=10,
    evaluation_strategy='steps',
    save_steps=500,
    save_total_limit=2,
)

# Train model
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()

# Save model
model.save_pretrained('models/mt5_model')
tokenizer.save_pretrained('models/mt5_model')