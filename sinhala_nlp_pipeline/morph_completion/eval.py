# import os
# import torch
# import unicodedata
# from transformers import ByT5Tokenizer, T5ForConditionalGeneration
# from evaluate import load as load_metric
# from nltk.translate.bleu_score import corpus_bleu
# from tqdm import tqdm

# # ========== Configuration ==========
# MODEL_DIR = os.path.join("..", "models", "byt5_sinhala", "byt_base_old")
# TEST_DATA = [
#     ("සුමනගුත්ත පාලක ලෙණ පැමිණියාවූත් නොපැමිණියාවූත් සංඝයා", "සුමනගුත්ත නම් පාලකගේ ලෙණ පැමිණියාවූත් නොපැමිණියාවූත් සංඝයාට දෙන ලදී"),
#     ("ධම්ම තෙරුණ් ලෙණ සංඝයා", "ධම්ම න​ම් තෙරුණ්ගේ ලෙණ සංඝයාට දෙන ලදී"),
#     ("තිස්ස තෙරුණ් පියගැටපෙ​ළ", "තිස්ස න​ම් තෙරුණ්ගේ පියගැටපෙ​ළ"),
#     ("සුමන ස්වාමි ලෙණ සංඝයා", "සුමන න​ම් ස්වාමිගේ ලෙණ සංඝයාට දෙන ලදී"),
#     ("මහාතිස්ස ලෙණ සංඝයා", "මහාතිස්සගේ ලෙණ සංඝයාට දෙන ලදී")
# ]

# # ========== Load Model ==========
# tokenizer = ByT5Tokenizer.from_pretrained(MODEL_DIR)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# # ========== Normalization ==========
# def normalize_text(text):
#     return unicodedata.normalize("NFC", text).replace("\u200b", "").replace("\u00A0", " ").strip()

# # ========== Inference ==========
# def generate_output(text):
#     input_text = f"input: {text} → output:"
#     input_ids = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=256).input_ids.to(device)
#     output_ids = model.generate(
#         input_ids=input_ids,
#         max_length=256,
#         num_beams=4,
#         temperature=0.7,
#         top_k=50,
#         do_sample=True,
#         early_stopping=True
#     )
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# # ========== Predict ==========
# predictions = []
# references = []
# inputs = []

# print("Generating predictions...")
# for inp, ref in tqdm(TEST_DATA):
#     pred = generate_output(inp)
#     inputs.append(inp)
#     predictions.append(normalize_text(pred))
#     references.append(normalize_text(ref))

# # ========== ROUGE ==========
# rouge = load_metric("rouge")
# rouge_result = rouge.compute(predictions=predictions, references=references)

# print("\n--- ROUGE Scores ---")
# for key in ["rouge1", "rouge2", "rougeL"]:
#     score = rouge_result[key]
#     print(f"{key.upper()} - F1: {score:.4f}")

# # ========== BLEU ==========
# bleu_score = corpus_bleu([[ref.split()] for ref in references], [pred.split() for pred in predictions])
# print(f"\nBLEU Score: {bleu_score:.4f}")

# # ========== Print Results ==========
# print("\n--- Sample Results ---")
# for inp, pred, ref in zip(inputs, predictions, references):
#     print(f"\nInput     : {inp}")
#     print(f"Predicted : {pred}")
#     print(f"Expected  : {ref}")
#     if pred != ref:
#         print("⚠️  Mismatch Detected")



from transformers import ByT5Tokenizer, T5ForConditionalGeneration
import os
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
# from rouge_score import rouge_scorer
from evaluate import load
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from sklearn.metrics import precision_score, recall_score, f1_score


# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'byt5_sinhala', 'byt_base_old')
EVAL_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'evaluation_new.txt')
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'eval_results_new.txt')
PLOT_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'score_distribution_new.png')

def load_eval_data(file_path):
    """Read input-output pairs from evaluation.txt."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Match format: input: <sentence> → output: <sentence>
            match = re.match(r'input: (.*?) → output: (.*)', line)
            if match:
                input_sentence = match.group(1).strip()
                output_sentence = match.group(2).strip()
                data.append({"input": input_sentence, "output": output_sentence})
    # Split into validation and test (50-50)
    split_idx = len(data) // 2
    val_data = data[:split_idx]
    test_data = data[split_idx:]
    print(f"Loaded {len(data)} pairs: {len(val_data)} validation, {len(test_data)} test")
    return val_data, test_data

def load_byt5():
    try:
        from safetensors.torch import load_file
        print("Using safetensors for model loading")
    except ImportError:
        print("safetensors not installed; using default torch loading (ensure PyTorch >= 2.6)")
    tokenizer = ByT5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, use_safetensors=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    return model, tokenizer, device

def morph_complete(sentence, model, tokenizer, device):
    input_text = f"input: {sentence} → output:"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=256, truncation=True).to(device)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        max_length=256,
        num_beams=4,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        early_stopping=True
    )
    completed = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return completed

def evaluate_model(data, set_name):
    model, tokenizer, device = load_byt5()
    exact_matches = 0
    bleu_scores, meteor_scores, chrf_scores = [], [], []
    precision_scores, recall_scores, f1_scores = [], [], []
    # rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    chrf_metric = load("chrf")
    smoothie = SmoothingFunction().method4

    results = []
    for data_point in tqdm(data, desc=f"Evaluating {set_name}"):
        input_sentence = data_point["input"]
        expected_output = data_point["output"]
        predicted_output = morph_complete(input_sentence, model, tokenizer, device)

        # Exact match
        is_exact_match = predicted_output == expected_output
        if is_exact_match:
            exact_matches += 1

        # BLEU score
        reference = [expected_output.split()]
        candidate = predicted_output.split()
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        bleu_scores.append(bleu)

        # METEOR score
        meteor = single_meteor_score(expected_output.split(), predicted_output.split())
        meteor_scores.append(meteor)

        # chrF score
        chrf = chrf_metric.compute(predictions=[predicted_output], references=[expected_output])['score']
        chrf_scores.append(chrf)

        # Token-level precision, recall, F1
        ref_tokens = expected_output.split()
        pred_tokens = predicted_output.split()
        common = set(ref_tokens) & set(pred_tokens)

        tp = len(common)
        fp = len(set(pred_tokens) - set(ref_tokens))
        fn = len(set(ref_tokens) - set(pred_tokens))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)


        # ROUGE scores
        # rouge_result = scorer.score(expected_output, predicted_output)
        # for key in rouge_scores:
        #     rouge_scores[key].append(rouge_result[key].fmeasure)

        # Store result
        results.append({
            "input": input_sentence,
            "expected": expected_output,
            "predicted": predicted_output,
            "exact_match": is_exact_match,
            "bleu": bleu,
            "meteor": meteor,
            "chrf": chrf,
            # "rouge1": rouge_result['rouge1'].fmeasure,
            # "rouge2": rouge_result['rouge2'].fmeasure,
            # "rougeL": rouge_result['rougeL'].fmeasure
        })

    # Calculate average metrics
    total = len(data)
    accuracy = exact_matches / total if total > 0 else 0
    avg_bleu = sum(bleu_scores) / total if total > 0 else 0
    avg_meteor = sum(meteor_scores) / total if total > 0 else 0
    avg_chrf = sum(chrf_scores) / total if total > 0 else 0
    avg_precision = sum(precision_scores) / total if total > 0 else 0
    avg_recall = sum(recall_scores) / total if total > 0 else 0
    avg_f1 = sum(f1_scores) / total if total > 0 else 0

    # Plot score distributions
    plt.figure(figsize=(15, 10))
    metrics = ['BLEU', 'METEOR', 'chrF', 'Precision', 'Recall', 'F1']
    scores = [bleu_scores, meteor_scores, chrf_scores, precision_scores, recall_scores, f1_scores]
    for i, (metric, score_list) in enumerate(zip(metrics, scores)):
        plt.subplot(2, 3, i + 1)
        plt.hist(score_list, bins=20, edgecolor='black')
        plt.title(f'{set_name} {metric} Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(PLOT_FILE.replace('.png', f'_{set_name.lower()}.png'))
    plt.close()

    # Save results to file
    output_file = OUTPUT_FILE.replace('.txt', f'_{set_name.lower()}.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{set_name} Evaluation Results\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Exact Match Accuracy: {accuracy:.4f}\n")
        f.write(f"Average BLEU Score: {avg_bleu:.4f}\n")
        f.write(f"Average METEOR Score: {avg_meteor:.4f}\n")
        f.write(f"Average chrF Score: {avg_chrf:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n\n")
        # f.write(f"Average ROUGE-1 Score: {avg_rouge1:.4f}\n")
        # f.write(f"Average ROUGE-2 Score: {avg_rouge2:.4f}\n")
        # f.write(f"Average ROUGE-L Score: {avg_rougeL:.4f}\n\n")
        f.write(f"Detailed {set_name} Results:\n")
        for result in results:
            f.write(f"Input: {result['input']}\n")
            f.write(f"Expected: {result['expected']}\n")
            f.write(f"Predicted: {result['predicted']}\n")
            f.write(f"Exact Match: {result['exact_match']}\n")
            f.write(f"BLEU: {result['bleu']:.4f}\n")
            f.write(f"METEOR: {result['meteor']:.4f}\n")
            f.write(f"chrF: {result['chrf']:.4f}\n")
            f.write(f"Precision: {precision_scores[i]:.4f}\n")
            f.write(f"Recall: {recall_scores[i]:.4f}\n")
            f.write(f"F1: {f1_scores[i]:.4f}\n\n")
            # f.write(f"ROUGE-1: {result['rouge1']:.4f}\n")
            # f.write(f"ROUGE-2: {result['rouge2']:.4f}\n")
            # f.write(f"ROUGE-L: {result['rougeL']:.4f}\n\n")

    return results, bleu_scores, meteor_scores, chrf_scores, avg_precision, avg_recall, avg_f1


if __name__ == "__main__":
    # Load and split data
    val_data, test_data = load_eval_data(EVAL_FILE)
    
    # Evaluate validation set
    val_results, val_bleu, val_meteor, val_chrf, val_precision, val_recall, val_f1 = evaluate_model(val_data, "Validation")
    
    # Evaluate test set
    # test_results, test_bleu, test_meteor, test_chrf, test_rouge = evaluate_model(test_data, "Test")