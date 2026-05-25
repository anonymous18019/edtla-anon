
# EDTLA: Error-Driven Targeted LLM Augmentation for Hinglish POS Tagging

Code for our ARR submission on error-driven targeted LLM augmentation for code-mixed POS tagging.

## Dataset
`dataFiles/` contains placeholder TSV files (no tokens, Twitter/X policy). Download the Hinglish POS corpus from [Singh et al., COLING 2018](https://aclanthology.org/C18-1271/) and place in `dataFiles/`. Training split: first 29,683 tokens; test: remaining 3,327 tokens.

## Reproduction
1. Train baseline (seeds 42–51): `trainOriginalModel.py`
2. Train EDTLA (set `VERBNOUN_FILE` to desired budget file): `trainNounVerbSyntheticModel.py`
3. Evaluate across seeds: `MultipleSeedErrorAnalysis.py`
4. McNemar tests: `McNemarComparison.py`

Synthetic data subsets: `synthetic_noun_verb_1pct.txt` (55 sent, main result), `synthetic_noun_verb_3pct.txt` (165 sent), `synthetic_noun_verb.txt` (301 sent, full). To regenerate subsets: `split.py`.

## License
MIT
