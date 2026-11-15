"""Optional CPU worker; models are read from the local HF cache only."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for name in ("image", "output", "model", "revision"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = CLIPVisionModelWithProjection.from_pretrained(
        args.model, revision=args.revision, local_files_only=True, use_safetensors=True).eval().to("cpu")
    processor = CLIPImageProcessor.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    with Image.open(args.image) as im, torch.inference_mode():
        vector = model(**processor(images=im.convert("RGB"), return_tensors="pt")).image_embeds[0]
        vector = vector / vector.norm()
    result = {"model": args.model, "revision": args.revision, "dimensions": len(vector),
              "vector": vector.tolist(), "normalization": "L2", "device": "cpu"}
    Path(args.output).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
