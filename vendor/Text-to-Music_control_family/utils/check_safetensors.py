#!/usr/bin/env python3
# list_modules_from_safetensors.py
import argparse
from safetensors import safe_open

def main():
    ap = argparse.ArgumentParser(description="List module names in a .safetensors file")
    ap.add_argument("path", help="Path to .safetensors file")
    ap.add_argument(
        "--level", type=int, default=None,
        help="Optional: keep only this many dot-components (e.g., 3 => 'model.layers.0')"
    )
    args = ap.parse_args()

    with safe_open(args.path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    if args.level:
        def head(s, n): 
            parts = s.split(".")
            return ".".join(parts[:min(n, len(parts))])
        modules = {head(k, args.level) for k in keys}
    else:
        # default: everything before the last token (e.g., drop 'weight', 'bias', etc.)
        def prefix(k):
            return k.rsplit(".", 1)[0] if "." in k else "<root>"
        modules = {prefix(k) for k in keys}

    for m in sorted(modules):
        print(m)

if __name__ == "__main__":
    main()
