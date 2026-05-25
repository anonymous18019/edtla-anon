"""
Configure paths and API key via environment variables / constants below.
"""
# NOTE: Replace {First 154 tokens of training data--PASTE HERE} in the prompt below
# with the actual first 154 tokens of your training file (finalTrainData.tsv),
# formatted as word<TAB>language<TAB>POS, one token per line.

import os
import re
import math
import sys
import time
import json
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Set


# OpenAI client (the modern SDK style used earlier in examples)
from openai import OpenAI

import processTestingData

# ---------------------------
# CONFIG - EDIT THESE VALUES
# ---------------------------
# processTestingData.py must populate `sentences` and `pos_tags` lists as used in your prior code
# Change the below filenames
SYNTHETIC_RAW_OUT = ""  # file where raw generation batches are appended
SYNTHETIC_NUMBERED = ""  # file with sentence numbers added (for validation)
SYNTHETIC_CURATED = ""  # final curated output (word<TAB>lang<TAB>POS, blank lines between sentences)


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
# Generate synthetic sentences (batched)
# ---------------------------


GEN_PROMPT_TEMPLATE = """[ROLE]
You are a data generator for a Hindi–English code-mixed (Hinglish) POS tagging dataset.

[OBJECTIVE]
Generate diverse, random synthetic Hinglish sentences that mimic natural 
social media content (Twitter/X). Simply generate natural, representative 
text covering a wide variety of topics (daily life, sports, news, opinions, entertainment).

[LANGUAGE AND STYLE]
- Language: Hindi–English code-mixed (romanized Hindi)
- Style: informal Twitter-style Hinglish
- Sentence length: approximately 5–20 tokens
- Sentences should resemble general social media commentary

[ANNOTATION REQUIREMENTS]
- Annotate every token with a POS tag using the dataset’s original tag set: NOUN, PROPN, VERB, ADJ, ADV, DET, ADP, PRON, PRON_WH, PART, PART_NEG, NUM, CONJ, X
- Each line should contain a word, a language (en or hi), and a POS, each separated by tabs.
- Sentences should be separated using blank lines

[EXAMPLE TRAINING DATA]
{First 154 tokens of training data--PASTE HERE}
"""


def generate_synthetic_sentences(
                                 target: int,
                                 batch_size: int = BATCH_SIZE,
                                 out_file: str = SYNTHETIC_RAW_OUT) -> List[str]:
    """
    Generates synthetic sentences in batches from OpenAI o4 and writes them to out_file.
    Returns the list of unique generated sentence blocks.
    """

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    generated_blocks = []
    seen_norm = set()
    runs = math.ceil(target / batch_size)
    total_generated = 0
    print(f"Starting generation: target={target}, batch_size={batch_size}, runs={runs}")

    for run_idx in range(runs):
        to_gen = min(batch_size, target - total_generated)
        prompt = GEN_PROMPT_TEMPLATE.format(
            batchSize=to_gen,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P
        )

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

    # 2. generate synthetic sentences targeted at that pair
    generated_blocks = generate_synthetic_sentences(TARGET_SENTENCES, batch_size=BATCH_SIZE,
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
    print(f"Raw synthetic file: {SYNTHETIC_RAW_OUT}")
    print(f"Numbered synthetic file: {SYNTHETIC_NUMBERED}")
    print(f"Curated synthetic file: {SYNTHETIC_CURATED}")
    print(
        f"Total generated (unique) approx: {len(blocks)}; removed: {len(to_delete)}; final: check {SYNTHETIC_CURATED}")


if __name__ == "__main__":
    main()
