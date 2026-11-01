# Writes a small GGUF with only the target's token_embd.weight and output.weight (what the draft reads of its target), so
# the 1 GB travels to the card instead of the 37 GB file.
#   python3 -I extract_head.py TARGET.gguf OUT.gguf
import sys

import gguf

r = gguf.GGUFReader(sys.argv[1])
w = gguf.GGUFWriter(sys.argv[2], "qwen35moe")
for t in r.tensors:
    if t.name in ("token_embd.weight", "output.weight"):
        w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
