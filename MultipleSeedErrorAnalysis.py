import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import processTestingData

# ===================== CONFIG =====================
MODEL_DIRS = [
    r"" # Include List of Model Directories
]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_PAIR = ("NOUN", "VERB") # SET THIS TO THE PAIR THAT YOU ARE EXAMININIG
PAIR_A, PAIR_B = TARGET_PAIR

# ===================== LOAD TEST DATA =====================
raw_sentences = processTestingData.sentences
raw_tags = processTestingData.pos_tags
print(f"Loaded {len(raw_sentences)} sentences.")

# ===================== EVALUATION FUNCTION =====================
def evaluate_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(DEVICE)

    label_list = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    true_flat = []
    pred_flat = []
    confusion_errors = Counter()

    for sent_idx, tokens in enumerate(raw_sentences):
        # Tokenize sentence individually to preserve word_ids
        encoding = tokenizer(tokens, is_split_into_words=True, truncation=True, padding=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [1, seq_len, num_labels]
            pred_ids = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

        word_ids = encoding.word_ids(batch_index=0)
        gold_tags = raw_tags[sent_idx]

        prev_word_idx = None
        for tok_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue  # skip special tokens
            if word_idx != prev_word_idx:
                # first subword of the word
                true_tag = gold_tags[word_idx]
                pred_tag = label_list[pred_ids[tok_idx]]

                true_flat.append(true_tag)
                pred_flat.append(pred_tag)

                if true_tag != pred_tag:
                    confusion_errors[(true_tag, pred_tag)] += 1
            prev_word_idx = word_idx

    acc = accuracy_score(true_flat, pred_flat)
    macro_f1 = f1_score(true_flat, pred_flat, average="macro")

    pair_confusions = (
        confusion_errors.get((PAIR_A, PAIR_B), 0)
        + confusion_errors.get((PAIR_B, PAIR_A), 0)
    )

    return acc, macro_f1, pair_confusions

# ===================== RUN EVALUATION =====================
results = []
pair_name = f"{PAIR_A}<->{PAIR_B}"

for i, model_dir in enumerate(MODEL_DIRS, start=1):
    print(f"Evaluating Model{i}: {model_dir}")
    acc, macro_f1, pair_conf = evaluate_model(model_dir)

    results.append({
        "Model": f"Model{i}",
        "Accuracy": acc,
        "Macro-F1": macro_f1,
        pair_name: pair_conf
    })

# ===================== PRINT SUMMARY =====================
df = pd.DataFrame(results)
print("\nSummary of Models:")
print(df.to_string(index=False))
