def read_pos_file(filepath):
    sentences = []
    tags = []

    current_sentence = []
    current_tags = []
    skip_sentence = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current_sentence and not skip_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                current_sentence = []
                current_tags = []
                skip_sentence = False
            else:
                parts = line.split()  # split on whitespace, not just tabs
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[-1]
                    current_sentence.append(token)
                    current_tags.append(tag)
                else:
                    print(f"Skipping malformed line: '{line}'")
                    skip_sentence = True

        if current_sentence and not skip_sentence:
            sentences.append(current_sentence)
            tags.append(current_tags)

    print(f"Loaded {len(sentences)} sentences and {len(tags)} tag sequences from {filepath}")
    return sentences, tags


