import os
import re
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import ByT5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from evaluate import load
from sklearn.metrics import f1_score

# === Paths ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'byt5_reorder')
EVAL_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'evaluation_reorder.txt')
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'byt5_reorder_eval_results.txt')
PLOT_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'byt5_reorder_score_distribution.png')

# === Load model ===
def load_byt5():
    try:
        from safetensors.torch import load_file
        print("Using safetensors for model loading")
    except ImportError:
        print("safetensors not installed; falling back to torch default")
    tokenizer = ByT5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, use_safetensors=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded to {device}")
    return model, tokenizer, device

# === Load eval data ===
def load_eval_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'input: (.*?) → output: (.*)', line.strip())
            if match:
                data.append({'input': match.group(1).strip(), 'output': match.group(2).strip()})
    print(f"Loaded {len(data)} pairs")
    return data

# === Prediction ===
def reorder_sentence(model, tokenizer, device, sentence):
    input_text = f"input: {sentence} → output:"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=256).to(device)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        max_length=256,
        num_beams=4,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Evaluation ===
def evaluate_model(data):
    model, tokenizer, device = load_byt5()
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    chrf_metric = load("chrf")
    smoothie = SmoothingFunction().method4

    exact_matches = 0
    bleu_scores, meteor_scores, chrf_scores = [], [], []
    # rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    f1_scores = []
    results = []

    for item in tqdm(data, desc="Evaluating"):
        inp = item['input']
        ref = item['output']
        pred = reorder_sentence(model, tokenizer, device, inp)

        ref_tokens = ref.split()
        pred_tokens = pred.split()

        # Metrics
        exact_match = ref.strip() == pred.strip()
        if exact_match:
            exact_matches += 1

        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        meteor = single_meteor_score(ref.split(), pred.split())
        chrf = chrf_metric.compute(predictions=[pred], references=[ref])['score']
        # rouge = scorer.score(ref, pred)
        # rouge = scorer.score(ref.replace(" ", ""), pred.replace(" ", ""))

        # F1 Score (word-level)
        common_tokens = list(set(ref_tokens) | set(pred_tokens))
        ref_vec = [1 if token in ref_tokens else 0 for token in common_tokens]
        pred_vec = [1 if token in pred_tokens else 0 for token in common_tokens]
        try:
            f1 = f1_score(ref_vec, pred_vec)
        except:
            f1 = 0.0

        # Save scores
        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        chrf_scores.append(chrf)
        # rouge_scores['rouge1'].append(rouge['rouge1'].fmeasure)
        # rouge_scores['rouge2'].append(rouge['rouge2'].fmeasure)
        # rouge_scores['rougeL'].append(rouge['rougeL'].fmeasure)
        f1_scores.append(f1)

        results.append({
            "input": inp,
            "expected": ref,
            "predicted": pred,
            "exact": exact_match,
            "bleu": bleu,
            "meteor": meteor,
            "chrf": chrf,
            # "rouge1": rouge['rouge1'].fmeasure,
            # "rouge2": rouge['rouge2'].fmeasure,
            # "rougeL": rouge['rougeL'].fmeasure,
            "f1": f1
        })

    total = len(data)
    avg = lambda scores: sum(scores) / total if total else 0

    print(f"\nEvaluation complete: {total} samples")
    print(f"Exact Match: {exact_matches}/{total} = {exact_matches/total:.2%}")
    print(f"Avg BLEU: {avg(bleu_scores):.4f}")
    print(f"Avg METEOR: {avg(meteor_scores):.4f}")
    print(f"Avg chrF: {avg(chrf_scores):.4f}")
    # print(f"Avg ROUGE-1: {avg(rouge_scores['rouge1']):.4f}")
    # print(f"Avg ROUGE-2: {avg(rouge_scores['rouge2']):.4f}")
    # print(f"Avg ROUGE-L: {avg(rouge_scores['rougeL']):.4f}")
    print(f"Avg F1: {avg(f1_scores):.4f}")

    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"Input: {r['input']}\n")
            f.write(f"Expected: {r['expected']}\n")
            f.write(f"Predicted: {r['predicted']}\n")
            f.write(f"Exact Match: {r['exact']}\n")
            f.write(f"BLEU: {r['bleu']:.4f}, METEOR: {r['meteor']:.4f}, chrF: {r['chrf']:.4f}\n")
            # f.write(f"ROUGE-1: {r['rouge1']:.4f}, ROUGE-2: {r['rouge2']:.4f}, ROUGE-L: {r['rougeL']:.4f}, F1: {r['f1']:.4f}\n\n")

    # Plot distributions
    plt.figure(figsize=(15, 10))
    metric_names = ['BLEU', 'METEOR', 'chrF', 'F1']
    all_scores = [bleu_scores, meteor_scores, chrf_scores, f1_scores]

    for i, (metric, scores) in enumerate(zip(metric_names, all_scores)):
        plt.subplot(3, 3, i+1)
        plt.hist(scores, bins=20, edgecolor='black')
        plt.title(f'{metric} Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()

if __name__ == "__main__":
    data = load_eval_data(EVAL_FILE)
    evaluate_model(data)
