"""
Automated EDTLA pipeline:
1. Find top symmetric confusion pair from a baseline model + test set.
2. Generate synthetic sentences targeted at that pair using o4 (batched).
3. Number sentences and ask o4 to point out which sentence numbers to delete.
4. Remove flagged sentences and produce a final curated file.

Configure paths and API key via environment variables / constants below.
"""

import os
import re
import math
import sys
import time
import json
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Set

# HuggingFace / transformers
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer

# OpenAI client (the modern SDK style used earlier in examples)
from openai import OpenAI

import processTestingData

# ---------------------------
# CONFIG - EDIT THESE VALUES
# ---------------------------
MODEL_DIR = r""  # directory of your saved model (checkpoint) used for error analysis
# processTestingData.py must populate `sentences` and `pos_tags` lists as used in your prior code
# Change the below filenames
SYNTHETIC_RAW_OUT = "synthetic_raw_pos1_pos2.txt"  # file where raw generation batches are appended
SYNTHETIC_NUMBERED = "synthetic_numbered_pos1_pos2.txt"  # file with sentence numbers added (for validation)
SYNTHETIC_CURATED = "synthetic_curated_pos1_pos2.txt"  # final curated output (word<TAB>lang<TAB>POS, blank lines between sentences)

# Set to None to auto-detect from model; otherwise set as a tuple (POS1, POS2)
OVERRIDE_SYMMETRIC_PAIR = None  # example: force generation to target NOUN↔VERB

# OpenAI / generation parameters
OPENAI_MODEL = "gpt-4o"  # model to call
GEN_TEMPERATURE = 0.75
GEN_TOP_P = 0.95

VALIDATION_TEMPERATURE = 0.0
VALIDATION_TOP_P = 1.0

# How many synthetic sentences you want overall
TARGET_SENTENCES = 400

# How many sentences to request per generation call (batch). Recommended: 25-75
BATCH_SIZE = 50

# Conservative token estimator for output; used to compute max_output_tokens parameter
# (tokens per sentence: accounts for word + language token + POS token + separators)
EST_TOKENS_PER_SENT = 45

# Fallback max token per batch
MAX_TOKENS_PER_BATCH = max(2000, int(BATCH_SIZE * EST_TOKENS_PER_SENT + 800))

# OpenAI client will be created with OpenAI(api_key=...)
# ---------------------------


# instantiate OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ---------------------------
# Step 0: helper utilities
# ---------------------------
def normalize_sentence_block(block: str) -> str:
    """Normalize a sentence block (strip trailing whitespace, unify newlines)."""
    lines = [ln.rstrip() for ln in block.strip().splitlines() if ln.strip() != ""]
    return "\n".join(lines)


def split_into_sent_blocks(file_text: str) -> List[str]:
    """Split text into sentence blocks separated by one or more blank lines."""
    parts = re.split(r'\n\s*\n', file_text.strip())
    blocks = [normalize_sentence_block(p) for p in parts if p.strip() != ""]
    return blocks


def parse_numbers_csv(s: str) -> List[int]:
    """Parse a string of comma-separated numbers; robust to whitespace and stray chars."""
    # allow numbers separated by commas/spaces/newlines; extract ints
    found = re.findall(r'\d+', s)
    return [int(x) for x in found]


# ---------------------------
# Step 1: Compute confusion matrix and find top symmetric pair
# ---------------------------
def find_top_symmetric_confusion(model_dir: str) -> Tuple[str, str]:
    """
    Run the existing error analysis code to find the top symmetric confusion pair.
    Returns: (partOfSpeech1, partOfSpeech2)
    """
    # === CONFIG ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LENGTH = 128

    # === LOAD MODEL & TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(DEVICE)

    # === LOAD TEST DATA ===
    raw_sentences = processTestingData.sentences
    raw_tags = processTestingData.pos_tags
    print(f"Loaded {len(raw_sentences)} sentences for error analysis.")

    # === LABEL MAPPING (fixed) ===
    label_list = [model.config.id2label[i] for i in range(model.config.num_labels)]
    label_to_id = getattr(model.config, "label2id", {l: i for i, l in enumerate(label_list)})
    # === TOKENIZE & ALIGN LABELS ===
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
            # lbls is now a list of integer label ids
            word_ids = tokenized.word_ids(batch_index=i)
            prev = None
            label_ids = []
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev:
                    # safe: if lbls is shorter than word count, fall back to -100
                    if isinstance(wid, int) and wid < len(lbls):
                        label_ids.append(int(lbls[wid]))
                    else:
                        label_ids.append(-100)
                else:
                    label_ids.append(-100)
                prev = wid
            aligned.append(label_ids)
        tokenized["labels"] = aligned
        return tokenized

    # convert labels to integer IDs before creating Dataset
    int_tags = [[label_to_id.get(t, -100) for t in sent] for sent in raw_tags]
    dataset = Dataset.from_dict({"tokens": raw_sentences, "labels": int_tags})
    tokenized_ds = dataset.map(tokenize_and_align, batched=True)

    # === PREDICTION ===
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
    pred_out = trainer.predict(tokenized_ds)
    pred_logits, true_labels = pred_out.predictions, pred_out.label_ids
    pred_ids = np.argmax(pred_logits, axis=-1)

    # === CONFUSION COUNT ===
    confusion_errors = Counter()
    for pred_seq, true_seq in zip(pred_ids, true_labels):
        for p, t in zip(pred_seq, true_seq):
            if t == -100:
                continue
            true_tag = label_list[t]
            pred_tag = label_list[p]
            if true_tag != pred_tag:
                confusion_errors[(true_tag, pred_tag)] += 1

    # === DIAGONAL-LIKE SYMMETRIC CONFUSIONS ===
    diag_like_confusions = [(t, p, c) for (t, p), c in confusion_errors.items() if t != p and (p, t) in confusion_errors]
    diag_like_confusions.sort(key=lambda x: confusion_errors[(x[0], x[1])] + confusion_errors[(x[1], x[0])], reverse=True)

    if not diag_like_confusions:
        raise RuntimeError("No symmetric confusion pairs found in the test predictions.")

    top_pair = diag_like_confusions[0]
    a, b = top_pair[0], top_pair[1]
    print(f"Top symmetric confusion pair: {a} ↔ {b}")
    return a, b

# ---------------------------
# Step 2: Generate synthetic sentences (batched)
# ---------------------------


GEN_PROMPT_TEMPLATE = """[ROLE]
You are a data generator for a Hindi–English code-mixed (Hinglish) POS tagging dataset.

[OBJECTIVE]
Generate synthetic Hinglish sentences where contextual cues
clearly distinguish {partOfSpeech1} from {partOfSpeech2},
reducing symmetric confusion between the two tags.

[TARGETED ERROR MODE]
The targeted error is symmetric confusion between {partOfSpeech1} and {partOfSpeech2} tags
({partOfSpeech1}→{partOfSpeech2} and {partOfSpeech2}→{partOfSpeech1}).

[LANGUAGE AND STYLE]
- Language: Hindi–English code-mixed (romanized Hindi)
- Style: informal Twitter-style Hinglish
- Sentence length: approximately 5–20 tokens
- Sentences should resemble general social media commentary

[ANNOTATION REQUIREMENTS]
- Annotate every token with a POS tag using the dataset’s original tag set: NOUN, PROPN, VERB, ADJ, ADV, DET, ADP, PRON, PRON_WH, PART, PART_NEG, NUM, CONJ, X
- Each line should contain a word, a language (en or hi), and a POS, each separated by tabs.
- Sentences should be separated using blank lines
- A sentence may contain multiple {partOfSpeech1}s and {partOfSpeech2}s
- For each ambiguous word, generate multiple sentences keeping the surrounding context natural, preserving original token forms, and providing clear syntactic cues before and after the target token.

[EXAMPLE TRAINING DATA]
{INCLUDED FIRST 154 TOKENS OF TRAINING SET HERE FOR REFERENCE}
ACTUAL CONTENT NOT INCLUDED IN THIS RESPOSITORY

[DIVERSITY CONSTRAINTS]
- Vary sentence structure, word order, and verb tense
- Avoid repetitive phrasing and near-duplicate sentences

[QUANTITY]
Generate {batchSize} unique sentences.

[RESTRICTIONS]
- Do not include explanations or metadata
- Do not copy or paraphrase training examples
- Do not output unannotated text

[OUTPUT FORMAT]
Output only the annotated sentences. Do not output any other text.
"""

NOUN_VERB_TRAINING_DATA_RULES="""[NOUN-VERB Data Rules]
1. LIGHT-VERB CONSTRUCTIONS (MANDATORY)
If a content word is followed by a light verb:

• The content word MUST be tagged NOUN
• The light verb MUST be tagged VERB

2. POSTPOSITION ATTACHMENT (HIGH PRIORITY)
If a word is followed by a Hindi postposition

• That word MUST be tagged NOUN
• Do NOT convert it into a VERB

3. EVENT / OBJECT NOUN PROTECTION
Words denoting physical objects, food, abstract entities,
or events MUST remain NOUN unless they show tense or agreement.

Do NOT tag such tokens as VERB only because a verb follows them.

4. NOMINALIZED VERBS
If a verb stem does NOT show tense, agreement, or auxiliaries,
it MUST remain NOUN.

5. ENGLISH BARE IMPERATIVES
English verbs are VERB only if used as commands.

• VERB if sentence-initial command
• NOUN if preceded by determiner or used as entity

6. PARALLEL CLAUSE CONSISTENCY
If the same token appears in parallel or repeated clauses,
it MUST keep the same POS tag.

7. VERB TAGGING IS ALLOWED ONLY IF:
• The token shows tense or agreement
• OR it is a light verb
• OR it is an auxiliary

Otherwise, default to NOUN."""


def generate_synthetic_sentences(part1: str, part2: str,
                                 target: int,
                                 batch_size: int = BATCH_SIZE,
                                 out_file: str = SYNTHETIC_RAW_OUT) -> List[str]:
    """
    Generates synthetic sentences in batches from OpenAI o4 and writes them to out_file.
    Returns the list of unique generated sentence blocks.
    """
    extra_rules = ""
    pair = {part1, part2}
    if pair == {"NOUN", "VERB"}:
        extra_rules += NOUN_VERB_TRAINING_DATA_RULES
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    generated_blocks = []
    seen_norm = set()
    runs = math.ceil(target / batch_size)
    total_generated = 0
    print(f"Starting generation: target={target}, batch_size={batch_size}, runs={runs}")

    for run_idx in range(runs):
        to_gen = min(batch_size, target - total_generated)
        prompt = GEN_PROMPT_TEMPLATE.format(
            partOfSpeech1=part1,
            partOfSpeech2=part2,
            batchSize=to_gen,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P
        ) + extra_rules

        max_output_tokens = MAX_TOKENS_PER_BATCH  # conservative
        print(
            f"Generating batch {run_idx + 1}/{runs} (requesting {to_gen} sentences; max_output_tokens={max_output_tokens})...")
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                max_output_tokens=max_output_tokens

            )

        except Exception as e:
            print("OpenAI generation error:", e)
            # simple retry with backoff
            time.sleep(3)
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                max_output_tokens=max_output_tokens
            )

        text = resp.output_text.strip()

        # Split into blocks by blank lines
        blocks = split_into_sent_blocks(text)
        new_blocks = []
        for b in blocks:
            norm = re.sub(r'\s+', ' ', b.strip()).lower()
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            new_blocks.append(b)
            generated_blocks.append(b)
            total_generated += 1
            # stop if reached target
            if total_generated >= target:
                break

        # append new blocks to out_file with double newline separation
        if new_blocks:
            with open(out_file, "a", encoding="utf-8") as f:
                for blk in new_blocks:
                    f.write(blk.strip() + "\n\n")

        print(
            f"Batch {run_idx + 1} done. New unique blocks: {len(new_blocks)}. Total unique so far: {len(generated_blocks)}")

        if total_generated >= target:
            break

    print(f"Generation complete. Total unique generated sentences: {len(generated_blocks)} (written to {out_file})")
    return generated_blocks


# ---------------------------
# Step 3: Add sentence numbers
# ---------------------------
def write_numbered_file_from_blocks(blocks: List[str], numbered_out: str = SYNTHETIC_NUMBERED):
    """
    Write the list of sentence blocks to numbered file format:
    1
    token<TAB>lang<TAB>POS
    ...

    (blank line)
    2
    ...
    """
    with open(numbered_out, "w", encoding="utf-8") as f:
        for i, blk in enumerate(blocks, start=1):
            f.write(str(i) + "\n")
            # ensure each token line is present; write block as-is (already token-per-line)
            f.write(blk.strip() + "\n\n")
    print(f"Wrote numbered file with {len(blocks)} sentences to {numbered_out}")


# ---------------------------
# Step 4: Validation prompt (ask model which numbers to delete)
# ---------------------------
VALIDATION_PROMPT_TEMPLATE = """[ROLE]
You are a data quality checker for a Hindi–English code-mixed (Hinglish) POS tagging dataset.

[TASK]
Analyze the provided list of synthetic sentences (each sentence ends with a blank line, each word is on a separate line, annotated with word<TAB>language<TAB>POS). Before each sentence, there is a line with the sentence number (ex. 2) and nothing else. Identify numbers (sentences) that should be removed due to any of the following issues:
1. Exact duplicates of other sentences.
2. Near-duplicates with minor token changes.
3. Clearly incorrect or nonsensical content.
4. Violations of annotation format (missing POS or language, malformed tabs).

[REQUIREMENTS]
- Output **only** the numbers of sentences to be deleted.
- Sentence numbers start at 1.
- Separate sentence numbers with commas.
- Output sentence numbers in chronological order.
- Do **not** include any explanations, text, or metadata.
- Do **not** repeat line numbers.

[INPUT]
{chunk_text}
"""


def validate_numbered_file(numbered_file: str, chunk_size: int = 200) -> Set[int]:
    """
    Read a numbered file and ask the model (in chunks) which sentence numbers to delete.
    Returns a set of global sentence numbers to delete (1-based).
    """
    with open(numbered_file, "r", encoding="utf-8") as f:
        content = f.read()

    # split into "number + block" groups. We assume the numbered file uses the format:
    # <number>\n<token line>\n...\n\n<number>\n...
    groups = re.split(r'\n(?=\d+\n)', content.strip())  # each group begins with "N\n"
    # Normalize groups back to "N\n<block>" form
    sentence_blocks = []
    for g in groups:
        g = g.strip()
        if not g:
            continue
        # separate number and block
        m = re.match(r'^(\d+)\n(.*)$', g, flags=re.S)
        if not m:
            # skip malformed
            continue
        num = int(m.group(1))
        block = m.group(2).strip()
        sentence_blocks.append((num, block))

    total = len(sentence_blocks)
    print(f"Validating {total} numbered sentences in chunks of {chunk_size}...")

    to_delete_global = set()
    # process in chunks (preserve chronological order)
    for start in range(0, total, chunk_size):
        chunk = sentence_blocks[start:start + chunk_size]
        # build chunk_text in same numbered format
        chunk_text = "\n\n".join(f"{num}\n{blk}" for num, blk in chunk)
        prompt = VALIDATION_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
        print(f"Sending validation chunk for sentences {chunk[0][0]}..{chunk[-1][0]} (size={len(chunk)})")
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                temperature=VALIDATION_TEMPERATURE,
                top_p=VALIDATION_TOP_P,
                max_output_tokens=600  # should be small; output is just numbers
            )
        except Exception as e:
            print("Validation call error:", e)
            time.sleep(2)
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                temperature=VALIDATION_TEMPERATURE,
                top_p=VALIDATION_TOP_P,
                max_output_tokens=600
            )

        out = resp.output_text.strip()

        # Parse numbers; returned numbers are absolute sentence numbers (per our instructions).
        # If the model accidentally returns chunk-relative numbers, this simplistic parser still extracts ints.
        nums = parse_numbers_csv(out)
        # Ensure numbers are in the range of this chunk (if they are chunk-local, map)
        # Determine if majority of numbers are within chunk ranges; if not, assume they are global.
        chunk_nums = set(n for n, _ in chunk)
        mapped = []
        for n in nums:
            if n in chunk_nums:
                mapped.append(n)
            else:
                # if n <= len(chunk): probably chunk-relative numbering
                if 1 <= n <= len(chunk):
                    mapped.append(chunk[n - 1][0])  # map chunk-local to global
                else:
                    # out-of-range number: ignore with warning
                    print(f"Warning: validation returned out-of-range number {n}; ignoring.")
        # add mapped numbers to to_delete_global
        for v in mapped:
            to_delete_global.add(int(v))

        print(f"Chunk validation returned {len(mapped)} numbers to delete.")

    # sort
    to_delete_sorted = sorted(to_delete_global)
    print(f"Total sentences flagged for deletion: {len(to_delete_sorted)}")
    return set(to_delete_sorted)


# ---------------------------
# Step 5: Remove flagged sentences and write curated file
# ---------------------------
def write_curated_file(numbered_file: str, to_delete: Set[int], curated_out: str = SYNTHETIC_CURATED):
    """
    Read numbered file, remove sentence numbers in `to_delete`, remove the leading number lines,
    and write final curated file with token lines and blank lines between sentences.
    """
    with open(numbered_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    groups = re.split(r'\n(?=\d+\n)', content.strip())
    final_blocks = []
    for g in groups:
        g = g.strip()
        if not g:
            continue
        m = re.match(r'^(\d+)\n(.*)$', g, flags=re.S)
        if not m:
            continue
        num = int(m.group(1))
        block = m.group(2).strip()
        if num in to_delete:
            continue
        # Ensure each line in block is token<TAB>lang<TAB>POS
        lines = [ln.strip() for ln in block.splitlines() if ln.strip() != ""]
        # Quick format validation: ensure at least one tab in lines
        valid = all('\t' in ln for ln in lines)
        if not valid:
            # try to salvage: skip malformed sentence
            print(f"Skipping sentence {num} due to malformed lines (missing TAB).")
            continue
        final_blocks.append("\n".join(lines))

    # write curated_out
    with open(curated_out, "w", encoding="utf-8") as f:
        for blk in final_blocks:
            f.write(blk.strip() + "\n\n")

    print(f"Wrote curated file with {len(final_blocks)} sentences to {curated_out}")


# ---------------------------
# Main pipeline orchestration
# ---------------------------
def main():
    print("=== Automated EDTLA generation + curation pipeline ===")
    # 1. find top symmetric confusion pair
    if OVERRIDE_SYMMETRIC_PAIR is not None:
        part1, part2 = OVERRIDE_SYMMETRIC_PAIR
        print(f"Using OVERRIDE symmetric pair: {part1} ↔ {part2}")
    else:
        part1, part2 = find_top_symmetric_confusion(MODEL_DIR)
        print(f"Using TOP symmetric pair: {part1} ↔ {part2}")

    # 2. generate synthetic sentences targeted at that pair
    generated_blocks = generate_synthetic_sentences(part1, part2, TARGET_SENTENCES, batch_size=BATCH_SIZE,
                                                    out_file=SYNTHETIC_RAW_OUT)

    # if generation appended to an existing file, you may want to reload all blocks from that file:
    with open(SYNTHETIC_RAW_OUT, "r", encoding="utf-8") as f:
        all_text = f.read()
    blocks = split_into_sent_blocks(all_text)
    print(f"Total unique blocks read from {SYNTHETIC_RAW_OUT}: {len(blocks)}")

    # 3. numbering
    write_numbered_file_from_blocks(blocks, numbered_out=SYNTHETIC_NUMBERED)

    # 4. validate with o4 to get line numbers to delete
    to_delete = validate_numbered_file(SYNTHETIC_NUMBERED, chunk_size=200)

    # 5. remove flagged sentences and write curated output
    write_curated_file(SYNTHETIC_NUMBERED, to_delete, curated_out=SYNTHETIC_CURATED)

    print("=== Pipeline complete ===")
    print(f"Top symmetric pair: {part1} ↔ {part2}")
    print(f"Raw synthetic file: {SYNTHETIC_RAW_OUT}")
    print(f"Numbered synthetic file: {SYNTHETIC_NUMBERED}")
    print(f"Curated synthetic file: {SYNTHETIC_CURATED}")
    print(
        f"Total generated (unique) approx: {len(blocks)}; removed: {len(to_delete)}; final: check {SYNTHETIC_CURATED}")


if __name__ == "__main__":
    main()
