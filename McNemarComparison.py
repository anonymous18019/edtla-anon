import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from statsmodels.stats.contingency_tables import mcnemar
import processTestingData

# ===================== CONFIG =====================
# Paired: BASELINE_DIRS[i] and EDTLA_DIRS[i] must be the same seed
BASELINE_DIRS = [
   # INSERT DIRS HERE
]

EDTLA_DIRS = [
    # INSERT DIRS HERE
]

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
TARGET_PAIR = ("NOUN", "VERB")
PAIR_A, PAIR_B = TARGET_PAIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== LOAD TEST DATA =====================
raw_sentences = processTestingData.sentences
raw_tags      = processTestingData.pos_tags
print(f"Loaded {len(raw_sentences)} test sentences.")

# ===================== EVALUATION FUNCTION =====================
def evaluate_model(model_dir):
    """Returns per-token (true_tag, pred_tag) pairs for the full test set."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model     = AutoModelForTokenClassification.from_pretrained(model_dir).to(DEVICE)
    label_list = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    true_flat, pred_flat = [], []
    confusion_errors = Counter()

    for sent_idx, tokens in enumerate(raw_sentences):
        encoding = tokenizer(
            tokens, is_split_into_words=True,
            truncation=True, padding=True, return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

        word_ids   = encoding.word_ids(batch_index=0)
        gold_tags  = raw_tags[sent_idx]
        prev_word  = None

        for tok_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word:
                true_tag = gold_tags[word_idx]
                pred_tag = label_list[pred_ids[tok_idx]]
                true_flat.append(true_tag)
                pred_flat.append(pred_tag)
                if true_tag != pred_tag:
                    confusion_errors[(true_tag, pred_tag)] += 1
            prev_word = word_idx

    acc      = accuracy_score(true_flat, pred_flat)
    macro_f1 = f1_score(true_flat, pred_flat, average="macro")
    pair_conf = (
        confusion_errors.get((PAIR_A, PAIR_B), 0)
        + confusion_errors.get((PAIR_B, PAIR_A), 0)
    )
    return acc, macro_f1, pair_conf, true_flat, pred_flat


# ===================== MCNEMAR HELPER =====================
def run_mcnemar(true_flat, base_pred, edtla_pred, restrict_to=None):
    """
    Compute exact McNemar test comparing baseline vs EDTLA predictions.
    restrict_to: if a set of gold tags is given, only those tokens are used.
    Returns (p_value, n_b_better, n_e_better) where
      n_b_better = tokens baseline got right, EDTLA got wrong
      n_e_better = tokens EDTLA got right, baseline got wrong
    """
    assert len(true_flat) == len(base_pred) == len(edtla_pred)
    b_correct = [t == p for t, p in zip(true_flat, base_pred)]
    e_correct = [t == p for t, p in zip(true_flat, edtla_pred)]

    b_right_e_wrong = 0
    b_wrong_e_right = 0

    for i, true_tag in enumerate(true_flat):
        if restrict_to and true_tag not in restrict_to:
            continue
        if b_correct[i] and not e_correct[i]:
            b_right_e_wrong += 1
        elif not b_correct[i] and e_correct[i]:
            b_wrong_e_right += 1

    # Exact McNemar (b=off-diagonal count where baseline better,
    #                c=off-diagonal count where EDTLA better)
    table = [[0, b_right_e_wrong],
             [b_wrong_e_right, 0]]
    # statsmodels exact McNemar uses the two discordant cells
    result = mcnemar([[b_right_e_wrong + b_wrong_e_right, b_right_e_wrong],
                       [b_wrong_e_right, 0]], exact=True)
    return result.pvalue, b_right_e_wrong, b_wrong_e_right


# ===================== MAIN LOOP =====================
rows = []

for i, (base_dir, edtla_dir, seed) in enumerate(
        zip(BASELINE_DIRS, EDTLA_DIRS, SEEDS), start=1):

    print(f"\n--- Seed {seed} (Model {i}) ---")
    print(f"  Evaluating baseline...")
    b_acc, b_f1, b_conf, b_true, b_pred = evaluate_model(base_dir)

    print(f"  Evaluating EDTLA 1%...")
    e_acc, e_f1, e_conf, e_true, e_pred = evaluate_model(edtla_dir)

    assert b_true == e_true, f"Token ordering mismatch at seed {seed}!"

    # McNemar — all tokens
    p_global, b_better_g, e_better_g = run_mcnemar(b_true, b_pred, e_pred)

    # McNemar — NOUN/VERB tokens only
    p_nv, b_better_nv, e_better_nv = run_mcnemar(
        b_true, b_pred, e_pred, restrict_to={"NOUN", "VERB"}
    )

    print(f"  Baseline:  acc={b_acc:.4f}, F1={b_f1:.4f}, {PAIR_A}<->{PAIR_B}={b_conf}")
    print(f"  EDTLA 1%:  acc={e_acc:.4f}, F1={e_f1:.4f}, {PAIR_A}<->{PAIR_B}={e_conf}")
    print(f"  McNemar (global):   p={p_global:.4f}  "
          f"(baseline better: {b_better_g}, EDTLA better: {e_better_g})")
    print(f"  McNemar (NOUN/VERB): p={p_nv:.4f}  "
          f"(baseline better: {b_better_nv}, EDTLA better: {e_better_nv})")

    rows.append({
        "Seed":           seed,
        "Base_Acc":       round(b_acc, 6),
        "EDTLA_Acc":      round(e_acc, 6),
        "Base_F1":        round(b_f1,  6),
        "EDTLA_F1":       round(e_f1,  6),
        f"Base_{PAIR_A}<->{PAIR_B}":  b_conf,
        f"EDTLA_{PAIR_A}<->{PAIR_B}": e_conf,
        "d_i":            b_conf - e_conf,
        "McNemar_p_global":  round(p_global, 4),
        "E_better_global":   e_better_g,
        "B_better_global":   b_better_g,
        "McNemar_p_NV":      round(p_nv, 4),
        "E_better_NV":       e_better_nv,
        "B_better_NV":       b_better_nv,
    })

# ===================== SUMMARY =====================
df = pd.DataFrame(rows)
print("\n\n========== FULL COMPARISON SUMMARY ==========")
print(df.to_string(index=False))

conf_col_base  = f"Base_{PAIR_A}<->{PAIR_B}"
conf_col_edtla = f"EDTLA_{PAIR_A}<->{PAIR_B}"

print(f"\nMean baseline errors: {df[conf_col_base].mean():.2f}")
print(f"Mean EDTLA 1% errors: {df[conf_col_edtla].mean():.2f}")
print(f"Mean d_i:             {df['d_i'].mean():.2f}")
print(f"Mean rel. change:     {(df['d_i'] / df[conf_col_base]).mean()*100:.2f}%")

print(f"\nMcNemar global — sig (p<0.05): "
      f"{(df['McNemar_p_global'] < 0.05).sum()} / 10 seeds")
print(f"McNemar NOUN/VERB — sig (p<0.05): "
      f"{(df['McNemar_p_NV'] < 0.05).sum()} / 10 seeds")

print("\nPer-seed McNemar global p-values:")
for _, row in df.iterrows():
    sig = "***" if row["McNemar_p_global"] < 0.001 else \
          "**"  if row["McNemar_p_global"] < 0.01  else \
          "*"   if row["McNemar_p_global"] < 0.05  else ""
    print(f"  Seed {int(row['Seed'])}: p={row['McNemar_p_global']:.4f} {sig}  "
          f"(EDTLA better: {int(row['E_better_global'])}, "
          f"baseline better: {int(row['B_better_global'])})")

print("\nPer-seed McNemar NOUN/VERB p-values:")
for _, row in df.iterrows():
    sig = "***" if row["McNemar_p_NV"] < 0.001 else \
          "**"  if row["McNemar_p_NV"] < 0.01  else \
          "*"   if row["McNemar_p_NV"] < 0.05  else ""
    print(f"  Seed {int(row['Seed'])}: p={row['McNemar_p_NV']:.4f} {sig}  "
          f"(EDTLA better: {int(row['E_better_NV'])}, "
          f"baseline better: {int(row['B_better_NV'])})")
