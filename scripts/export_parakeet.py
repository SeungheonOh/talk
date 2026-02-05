#!/usr/bin/env python3
"""Export Parakeet-TDT-0.6B-v2 from NeMo to ONNX (FP16) + vocab.txt.

Uses FP16 so the encoder fits in a single ONNX file (<2GB) which is
required for CoreML acceleration (ort's CoreML EP breaks with external data).

Usage:
    pip install nemo_toolkit[asr] onnx onnxscript onnxconverter-common
    python scripts/export_parakeet.py

Outputs to models/parakeet-tdt/:
    encoder-model.onnx   (FP16, single file ~1.2GB)
    decoder_joint-model.onnx
    vocab.txt
"""

import functools
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr

OUT_DIR = Path("models/parakeet-tdt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PyTorch 2.10+ uses dynamo-based ONNX export by default, which is
# incompatible with NeMo's dynamic_axes format. Monkey-patch to force
# the legacy exporter.
_orig_export = torch.onnx.export

@functools.wraps(_orig_export)
def _legacy_export(*args, **kwargs):
    kwargs["dynamo"] = False
    return _orig_export(*args, **kwargs)

torch.onnx.export = _legacy_export

print("Downloading/loading Parakeet-TDT-0.6B-v2...")
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

print("Exporting encoder + decoder_joint to ONNX...")
model.export(str(OUT_DIR / "model.onnx"))

# Convert encoder to FP16 so it fits in a single protobuf (<2GB)
# This eliminates external data files which break CoreML EP.
print("Converting encoder to FP16...")
import onnx
from onnxconverter_common import float16

enc_path = OUT_DIR / "encoder-model.onnx"
enc_model = onnx.load(str(enc_path), load_external_data=True)

# Remove scattered external data files
for f in OUT_DIR.iterdir():
    if f.name not in ("encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt"):
        f.unlink()

enc_fp16 = float16.convert_float_to_float16(enc_model, keep_io_types=True)
onnx.save_model(enc_fp16, str(enc_path))
print(f"  encoder-model.onnx: {enc_path.stat().st_size / 1024**2:.0f} MB (FP16, single file)")

print("Writing vocab.txt...")
vocab = model.decoding.tokenizer.vocab
# vocab is a list of tokens; blank is appended as last entry
with open(OUT_DIR / "vocab.txt", "w") as f:
    for token in vocab:
        f.write(token + "\n")
    f.write("<blank>\n")

print(f"\nDone! Models exported to {OUT_DIR}/")
print(f"  encoder-model.onnx (FP16)")
print(f"  decoder_joint-model.onnx")
print(f"  vocab.txt ({len(vocab)+1} entries)")
