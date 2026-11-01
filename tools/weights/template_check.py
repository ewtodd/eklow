#!/usr/bin/env python3
"""template_check.py GGUF JINJA: exit 1 unless the GGUF's tokenizer.chat_template is byte for byte JINJA."""
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
got = r.fields["tokenizer.chat_template"].contents()
want = open(sys.argv[2], encoding="utf-8").read()
ok = got == want
print(f"chat template of {sys.argv[1]} {'==' if ok else '!='} {sys.argv[2]} ({len(got)} vs {len(want)} chars)")
sys.exit(0 if ok else 1)
