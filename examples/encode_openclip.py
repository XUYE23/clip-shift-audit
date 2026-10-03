"""Optional real-image adapter. Explicit invocation may download model weights.

CSV columns: path,split,label. split is calibration or test.
Calibration uses only known classes. Test label -1 denotes unknown.
Paths resolve relative to the CSV. labels.json is an ordered array of class names.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("classes")
    parser.add_argument("--output", default="features.npz")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch
    import open_clip
    from PIL import Image
    labels = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    if not labels or not all(isinstance(x,str) and x for x in labels):
        raise ValueError("classes must contain nonempty strings")
    root = Path(args.manifest).resolve().parent
    with open(args.manifest, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        path = (root / row["path"]).resolve()
        if path in seen:
            raise ValueError("duplicate image path across manifest; check split leakage")
        seen.add(path)
        label = int(row["label"])
        if row["split"] not in {"calibration","test"} or not -1 <= label < len(labels):
            raise ValueError("invalid split or label")
        if row["split"] == "calibration" and label < 0:
            raise ValueError("calibration must contain known classes only")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    model = model.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    splits = {"calibration": [], "test": []}
    test_y = []
    with torch.inference_mode():
        texts = tokenizer([f"a photo of a {name}" for name in labels]).to(args.device)
        text = model.encode_text(texts).float().cpu().numpy()
        for row in rows:
            with Image.open(root / row["path"]) as im:
                tensor = preprocess(im.convert("RGB")).unsqueeze(0).to(args.device)
            vector = model.encode_image(tensor).float().cpu().numpy()[0]
            splits[row["split"]].append(vector)
            if row["split"] == "test":
                test_y.append(int(row["label"]))
    if not splits["calibration"] or not splits["test"]:
        raise ValueError("both calibration and test splits are required")
    np.savez_compressed(args.output, text=text,
                        calibration=np.stack(splits["calibration"]),
                        test=np.stack(splits["test"]),
                        labels=np.array(test_y, dtype=np.int64))
    Path(args.output + ".metadata.json").write_text(json.dumps({
        "model": args.model, "pretrained": args.pretrained,
        "classes": labels, "prompt": "a photo of a {name}",
        "open_clip_version": getattr(open_clip, "__version__", "unknown")
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
