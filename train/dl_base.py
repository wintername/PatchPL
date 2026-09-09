#!/usr/bin/env python3
"""Download Qwen2.5-Coder-3B base model (ASCII only)."""
import os, sys
from huggingface_hub import snapshot_download

DEST = "/home/wcx/swe/models/Qwen2.5-Coder-3B"
os.makedirs(os.path.dirname(DEST), exist_ok=True)
path = snapshot_download(
    "Qwen/Qwen2.5-Coder-3B", local_dir=DEST,
    ignore_patterns=["*.gguf"], resume_download=True)
print("BASE-DL-DONE", path)
