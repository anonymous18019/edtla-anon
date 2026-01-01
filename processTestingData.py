import re

filenames = [r""] # ENTER TEST DATA FILE NAME

sentences = []      # list of lists of words
pos_tags = []       # list of lists of POS tags
languages = []      # list of lists of language codes

def parse_line_by_columns(line):
    line_strip = line.strip()
    if not line_strip or line_strip.startswith("#"):
        return None

    tokens = line_strip.split()
    if len(tokens) != 3:
        return None

    word, lang, tag = tokens
    if not word or not lang or not tag:
        return None

    return word, lang, tag

# Read and parse files
for filename in filenames:
    print(f"Reading file: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            curr_sent = []
            curr_tags = []
            curr_langs = []
            for line in f:
                if not line.strip():
                    if curr_sent:
                        sentences.append(curr_sent)
                        pos_tags.append(curr_tags)
                        languages.append(curr_langs)
                        curr_sent, curr_tags, curr_langs = [], [], []
                    continue

                result = parse_line_by_columns(line)
                if result:
                    word, lang, tag = result
                    curr_sent.append(word)
                    curr_tags.append(tag)
                    curr_langs.append(lang)

            # Add last sentence if file doesn't end in a newline
            if curr_sent:
                sentences.append(curr_sent)
                pos_tags.append(curr_tags)
                languages.append(curr_langs)
    except FileNotFoundError:
        print(f"File not found: {filename}")
