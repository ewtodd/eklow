# eklow's image preparation (eklow-image-prep) against Hugging Face's processors, the models' reference: the size each
# picks and the resized pixels, image by image. Needs Python with transformers, torch, torchvision and Pillow (no model
# files: the processors are built from the projectors' settings).
#
#   python3 prep_compare.py PREP_BIN IMAGE...
#
# Environment: GEMMA_TOKENS (budgets, default "70 280 1120"), QWEN ("MIN:MAX" pairs, default "8:1024 8:256").
# Prints per image and setting: both sizes, then the mean and largest absolute byte difference and the share of bytes
# more than 2 apart. Pillow and stb_image decode JPEGs slightly differently, so a JPEG's bytes differ a little even
# where the resize is the same; the original-size line shows that part alone.
import os, subprocess, sys, tempfile

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from transformers.image_utils import SizeDict
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

prep, images = sys.argv[1], sys.argv[2:]
gemma_tokens = [int(t) for t in os.environ.get("GEMMA_TOKENS", "70 280 1120").split()]
qwen_bounds = [tuple(int(x) for x in b.split(":")) for b in os.environ.get("QWEN", "8:1024 8:256").split()]
gemma = Gemma4ImageProcessor()
qwen = Qwen2VLImageProcessor(patch_size=16, merge_size=2)


def eklow(args, path):
    with tempfile.NamedTemporaryFile(suffix=".rgb") as f:
        out = subprocess.run([prep, *map(str, args), path, f.name], check=True, capture_output=True, text=True).stdout
        w, h = map(int, out.split())
        return np.fromfile(f.name, dtype=np.uint8).reshape(h, w, 3)


def report(label, ours, ref):
    if ours.shape != ref.shape:
        print(f"  {label:18s} SIZE eklow {ours.shape[1]}x{ours.shape[0]} hf {ref.shape[1]}x{ref.shape[0]}")
        return
    d = np.abs(ours.astype(np.int16) - ref.astype(np.int16))
    print(f"  {label:18s} {ours.shape[1]}x{ours.shape[0]}  mean |d| {d.mean():.3f}  max {d.max()}  >2: {100 * (d > 2).mean():.2f}%")


for path in images:
    pil = np.array(Image.open(path).convert("RGB"))
    chw = torch.from_numpy(pil).permute(2, 0, 1).contiguous()
    print(os.path.basename(path), f"{pil.shape[1]}x{pil.shape[0]}")
    report("decoded", eklow(["decode"], path), pil)
    for t in gemma_tokens:
        ref = gemma.aspect_ratio_preserving_resize(image=chw, patch_size=16, max_patches=t * 9, pooling_kernel_size=3,
                                                   resample=tvF.InterpolationMode.BICUBIC)
        report(f"gemma {t}", eklow(["gemma", t], path), ref.permute(1, 2, 0).numpy())
    for lo, hi in qwen_bounds:
        ref = qwen.resize(chw[None], size=SizeDict(shortest_edge=lo * 1024, longest_edge=hi * 1024),
                          resample=tvF.InterpolationMode.BICUBIC, factor=32)[0]
        report(f"qwen {lo}:{hi}", eklow(["qwen", lo, hi], path), ref.permute(1, 2, 0).numpy())
