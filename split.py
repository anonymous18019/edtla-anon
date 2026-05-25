import random

def read_sentences(filepath):
    sentences = []
    current = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                if current:
                    sentences.append(current)
                    current = []
            else:
                current.append(line)
    if current:
        sentences.append(current)
    return sentences

def write_sentences(sentences, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for sent in sentences:
            for line in sent:
                f.write(line)
            f.write('\n')

# Load full 301-sentence synthetic dataset
all_sents = read_sentences('') # ENTER FILE NAME
print(f"Total sentences available: {len(all_sents)}")

# Fix a single sampling seed — separate from your training seeds 42-51
random.seed(99)

# Shuffle once
shuffled = all_sents.copy()
random.shuffle(shuffled)

# Sample sizes (sentences, not tokens — proportional to your 301 at 5.5%)
# 1%  → 55 sentences
# 3%  → 165 sentences
# 5.5% → 301 sentences (already exists, no new file needed)

budget_1pct  = shuffled[:55]
budget_3pct  = shuffled[:165]

write_sentences(budget_1pct,  '') # 1 percent file name
write_sentences(budget_3pct,  '') # 3 percent file name

print("Written: 1% (55 sentences), 3% (165 sentences)")
print("5.5% already exists as []") # original file name