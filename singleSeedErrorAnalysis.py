import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from collections import defaultdict, Counter
# +Synthetic_Data
# === CONFIG ===
MODEL_DIR = (r"") # ENTER MODEL DIRECTORY HERE
TEST_FILE = "dataFiles/finalTestData.tsv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHORT_SENTENCE_THRESHOLD = 5
MAX_LENGTH = 128

# === LOAD MODEL & TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)

# === LOAD TEST DATA ===
import processTestingData
raw_sentences = processTestingData.sentences
raw_tags = processTestingData.pos_tags
print(f"Loaded {len(raw_sentences)} sentences from processTestingData.")

# === TOKENIZE & ALIGN LABELS ===
label_list = list(model.config.id2label.values())
label_to_id = {l: i for i, l in enumerate(label_list)}

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    aligned = []
    for i, lbls in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev:
                label_ids.append(label_to_id[lbls[wid]])
            else:
                label_ids.append(-100)
            prev = wid
        aligned.append(label_ids)
    tokenized["labels"] = aligned
    return tokenized

dataset = Dataset.from_dict({"tokens": raw_sentences, "labels": raw_tags})
tokenized_ds = dataset.map(tokenize_and_align, batched=True)

# === PREDICTION ===
data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
pred_out = trainer.predict(tokenized_ds)
pred_logits, true_labels = pred_out.predictions, pred_out.label_ids
pred_ids = np.argmax(pred_logits, axis=-1)

# === METRICS ===
true_flat, pred_flat = [], []
pos_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
len_stats = {'short': [0, 0], 'long': [0, 0]}
confusion_errors = Counter()
position_tag_errors = defaultdict(Counter)

predictions = []
for pred_seq, true_seq in zip(pred_ids, true_labels):
    pred_tags = []
    seq_len = sum(1 for t in true_seq if t != -100)
    correct_count = 0
    for pos, (p, t) in enumerate(zip(pred_seq, true_seq)):
        if t == -100:
            continue
        true_tag = label_list[t]
        pred_tag = label_list[p]
        true_flat.append(true_tag)
        pred_flat.append(pred_tag)
        pred_tags.append(pred_tag)

        pos_stats[pos]['total'] += 1
        if true_tag == pred_tag:
            pos_stats[pos]['correct'] += 1
            correct_count += 1
        else:
            confusion_errors[(true_tag, pred_tag)] += 1
            position_tag_errors[pos][true_tag] += 1

    predictions.append(pred_tags)
    cat = 'short' if seq_len <= SHORT_SENTENCE_THRESHOLD else 'long'
    len_stats[cat][0] += correct_count
    len_stats[cat][1] += seq_len

# === OVERALL METRICS ===
acc = accuracy_score(true_flat, pred_flat)
print(f"\nOverall accuracy: {acc:.4f}")
print("\nClassification Report (Precision/Recall/F1 per class):")
print(classification_report(true_flat, pred_flat, digits=3))

macro_f1 = f1_score(true_flat, pred_flat, average='macro')
print(f"Macro-F1 Score: {macro_f1:.4f}")

# === PER-TAG ACCURACY ===
print("\nPer-tag Accuracy:")
tag_totals = Counter(true_flat)
tag_correct = Counter([t for t, p in zip(true_flat, pred_flat) if t == p])
for tag in sorted(label_list):
    tot = tag_totals.get(tag, 0)
    corr = tag_correct.get(tag, 0)
    if tot > 0:
        print(f"{tag:6}: {corr / tot:.3f} ({corr}/{tot})")

# === SHORT VS LONG ===
print("\nShort vs Long Sentence Accuracy:")
for cat, (corr, tot) in len_stats.items():
    if tot > 0:
        print(f"{cat.title():5}: {corr / tot:.3f} ({corr}/{tot})")

# === POSITION-WISE ACCURACY ===
print("\nPosition-wise Accuracy:")
for pos in sorted(pos_stats.keys()):
    tot = pos_stats[pos]['total']
    corr = pos_stats[pos]['correct']
    print(f"Pos {pos + 1:2d}: {corr / tot:.3f} ({corr}/{tot})")

# === CONFUSION MATRIX (Console) ===
print("\nConfusion Matrix:")
cm = confusion_matrix(true_flat, pred_flat, labels=label_list)
cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
print(cm_df.to_string())

# === TOP CONFUSION ERRORS ===
print("\nTop 10 Most Common Confusion Errors:")
for (true_t, pred_t), count in confusion_errors.most_common(10):
    print(f"{true_t} → {pred_t}: {count}")

# === MOST CONFUSED POSITIONAL TAGS ===
print("\nMost Misclassified Tags by Position:")
for pos, err_counter in position_tag_errors.items():
    top_tag, count = err_counter.most_common(1)[0]
    print(f"Pos {pos+1:2d}: {top_tag} ({count} errors)")

# === DIAGONAL CONFUSIONS (Top 10 symmetric confusions) ===
print("\nDiagonal-like Confusions (Top 10):")
diag_like_confusions = [(t, p, c) for (t, p), c in confusion_errors.items() if t != p and (p, t) in confusion_errors]
diag_like_confusions.sort(key=lambda x: confusion_errors[(x[0], x[1])] + confusion_errors[(x[1], x[0])], reverse=True)
seen = set()
for t, p, _ in diag_like_confusions:
    if (t, p) in seen or (p, t) in seen:
        continue
    seen.add((t, p))
    total = confusion_errors[(t, p)] + confusion_errors[(p, t)]
    print(f"{t} ↔ {p}: {total} total confusions ({confusion_errors[(t, p)]}+{confusion_errors[(p, t)]})")
    if len(seen) >= 10:
        break
