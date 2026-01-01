import re
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from datasets import Dataset, DatasetDict

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# ----------------------------
# CONFIG
# ----------------------------
SEED = # ENTER SEED NUMBER HERE (WE USED 42-51 FOR OUR TESTING)
MODEL_ID_OR_PATH = "l3cube-pune/hing-roberta"
OUTPUT_DIR = rf"" # ENTER OUTPUT DIRECTORY HERE
TRAIN_FILE = r"dataFiles/finalTrainData.tsv"
VERBNOUN_FILE = r"dataFiles/synthetic_noun_verb.txt"

NUM_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 16
LABEL_SMOOTHING = 0.08
MAX_LENGTH = 128

EXCLUDE_LABELS_FROM_MACRO = {"X", "REST"}
# ----------------------------
# REPRODUCIBILITY
# ----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

# ----------------------------
# Utilities: cleaning / canonicalization
# ----------------------------
def clean_tag(tag: str) -> str:
    tag = "" if tag is None else str(tag)
    tag = re.sub(r"[\x00-\x1F\x7F]", "", tag)
    tag = tag.strip()
    return tag.upper()

def clean_token(tok: str) -> str:
    tok = "" if tok is None else str(tok)
    tok = tok.strip()
    return tok

def is_clean_word(token):
    if token.startswith("#") or token.startswith("@"):
        return False
    if re.match(r'https?://', token):
        return False
    if re.match(r'^\d+$', token):
        return False
    if all(char in r'.,!?;:()[]{}' for char in token):
        return False
    return True

# ----------------------------
# LOAD & CLEAN DATA
# ----------------------------
import preProcessData  # your module; must expose read_pos_file(path) -> (sentences, pos_tags)

orig_sents, orig_tags = preProcessData.read_pos_file(TRAIN_FILE)

orig_tags = [[clean_tag(t) for t in sent] for sent in orig_tags]
orig_sents = [[clean_token(w) for w in sent] for sent in orig_sents]

synth_sentences, synth_tags = preProcessData.read_pos_file(VERBNOUN_FILE)

synth_tags = [[clean_tag(t) for t in sent] for sent in synth_tags]
synth_sentences = [[clean_token(w) for w in sent] for sent in synth_sentences]

print(f"Loaded {len(orig_sents)} original sentences and {len(synth_sentences)} synthetic sentences.")

# Combine sentences and tags
sentences = orig_sents + synth_sentences
pos_tags = orig_tags + synth_tags
print(f"Loaded {len(sentences)} total sentences (orig {len(orig_sents)}, synth {len(synth_sentences)})")

# ----------------------------
# BUILD RAW DATASET & SPLIT
# ----------------------------
raw_dataset = Dataset.from_dict({"tokens": sentences, "tags": pos_tags})
split = raw_dataset.train_test_split(test_size=0.05, seed=SEED)
dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
print("Train size:", len(dataset["train"]), "Val size:", len(dataset["validation"]))

# ----------------------------
# NO UPSAMPLING: uniform weights on training split only
# ----------------------------
train_weights_list = [1.0] * len(dataset["train"])
print("Using uniform train weights for training samples only. Num train samples:", len(train_weights_list))

# ----------------------------
# CREATE LABEL MAPPINGS
# ----------------------------
unique_tags = sorted({tag for tag_list in pos_tags for tag in tag_list})
label2id = {tag: i for i, tag in enumerate(unique_tags)}
id2label = {i: tag for tag, i in label2id.items()}
num_labels = len(label2id)
print("Num labels:", num_labels)
print("Labels:", unique_tags)

# ----------------------------
# TOKENIZER + ALIGN LABELS
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_OR_PATH, use_fast=True)
data_collator = DataCollatorForTokenClassification(tokenizer, padding="longest")

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
        padding=False,
    )
    all_labels = []
    for i, labs in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev:
                lab = labs[wid]
                label_ids.append(label2id[lab])
            else:
                label_ids.append(-100)
            prev = wid
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "tags"])
tokenized_dataset.set_format(type="torch")
print("Tokenized dataset ready.")

# Consistency check
if len(train_weights_list) != len(tokenized_dataset["train"]):
    raise ValueError(f"Length mismatch: train_weights_list ({len(train_weights_list)}) != tokenized_dataset['train'] ({len(tokenized_dataset['train'])})")

# ----------------------------
# LOAD MODEL
# ----------------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_ID_OR_PATH,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# ----------------------------
# METRICS
# ----------------------------
def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_labels = []
    true_preds = []
    for pred_row, label_row in zip(preds, label_ids):
        for p_i, l_i in zip(pred_row, label_row):
            if l_i != -100:
                true_labels.append(int(l_i))
                true_preds.append(int(p_i))

    labels_list = list(range(num_labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, true_preds, labels=labels_list, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, lab in enumerate(labels_list):
        per_class_metrics[f"f1_{id2label[lab]}"] = float(f1[i])
        per_class_metrics[f"precision_{id2label[lab]}"] = float(precision[i])
        per_class_metrics[f"recall_{id2label[lab]}"] = float(recall[i])
        per_class_metrics[f"support_{id2label[lab]}"] = int(support[i])

    include_label_ids = [i for i, t in id2label.items() if t not in EXCLUDE_LABELS_FROM_MACRO]
    if len(include_label_ids) == 0:
        mp = mr = mf1 = 0.0
    else:
        mp, mr, mf1, _ = precision_recall_fscore_support(true_labels, true_preds, labels=include_label_ids, average="macro", zero_division=0)

    acc = accuracy_score(true_labels, true_preds)
    metrics = {"precision_macro_excl_X": float(mp), "recall_macro_excl_X": float(mr),
               "f1_macro_excl_X": float(mf1), "accuracy": float(acc)}
    metrics.update(per_class_metrics)
    return metrics

# ----------------------------
# LABEL-SMOOTHED LOSS + CUSTOM TRAINER
# ----------------------------
def label_smoothed_nll_loss(logits, labels, smoothing=LABEL_SMOOTHING):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -log_probs.mean(dim=-1)
    return (confidence * nll + smoothing * smooth_loss).mean()

class CustomTrainer(Trainer):
    def __init__(self, *args, train_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_weights = train_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        logits_flat = logits.view(-1, model.config.num_labels)
        labels_flat = labels.view(-1)
        active = labels_flat != -100
        if active.sum().item() == 0:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (loss, outputs) if return_outputs else loss

        logits_active = logits_flat[active]
        labels_active = labels_flat[active].long()
        loss = label_smoothed_nll_loss(logits_active, labels_active, smoothing=LABEL_SMOOTHING)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        batch_size = self.args.train_batch_size

        # Uniform sampling, shuffle normally
        return DataLoader(train_dataset, batch_size=batch_size, collate_fn=self.data_collator, shuffle=True)

# ----------------------------
# TRAINING ARGUMENTS
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro_excl_X",
    greater_is_better=True,
    report_to="none",
    seed=SEED,
)

# ----------------------------
# INSTANTIATE TRAINER
# ----------------------------
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_weights=train_weights_list
)

# ----------------------------
# RUN TRAINING
# ----------------------------
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Fine-tuning complete! Model saved to", OUTPUT_DIR)