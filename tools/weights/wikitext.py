# wikitext.py OUT: wikitext-2-raw-v1 test split joined as llama.cpp's wiki.test.raw is.
import sys
from datasets import load_dataset
d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
open(sys.argv[1], "w", encoding="utf-8").write("".join(d["text"]))
