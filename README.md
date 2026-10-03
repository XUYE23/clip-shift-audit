# CLIP Shift Audit

[![Tests](https://github.com/XUYE23/clip-shift-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/XUYE23/clip-shift-audit/actions/workflows/ci.yml)

**Does your CLIP classifier know when to abstain?**

Audit cached image/text embeddings with known-only threshold calibration,
tie-correct OOD metrics, and a standalone HTML report. Core evaluation runs on CPU.

[简体中文](README.zh-CN.md) · [Protocol](docs/protocol.md) · [Contributing](CONTRIBUTING.md)

## Try it in a minute

Python 3.10+:

```sh
git clone https://github.com/XUYE23/clip-shift-audit.git
cd clip-shift-audit
python -m pip install .
clip-shift-audit demo --output outputs/demo.json
python -m unittest discover -s tests -v
```

Open outputs/demo.html. The bundled demo uses synthetic feature vectors, seed 23.
It checks the pipeline and metric behavior; it supplies no evidence of real CLIP accuracy.

## What you get

- Compare maximum cosine, maximum softmax probability, and top-two margin.
- Calibrate an abstention threshold using known-class calibration data.
- Report AUROC, FPR@95 TPR, known/unknown acceptance, selective accuracy, and
  tie-grouped risk/coverage curves.
- Fingerprint input arrays; reject zero vectors, NaN values, malformed labels,
  and test sets missing either class group.
- Export readable JSON and an offline HTML report.

## Bring real embeddings

Create an NPZ containing text [C,D], calibration [N,D], test [M,D],
and integer labels [M]. Labels 0..C-1 index text rows; -1 marks unknown classes.

```sh
clip-shift-audit audit features.npz --alpha 0.05 --temperature 0.01
```

Calibration images must be disjoint from test images and contain known classes only.
Keep model, prompts, and temperature fixed before calibration.

Optional OpenCLIP image extraction:

```sh
python -m pip install "open_clip_torch>=3,<4" pillow
python examples/encode_openclip.py images.csv labels.json --device cpu
clip-shift-audit audit features.npz
```

CSV header: path,split,label. Split values: calibration,test. Class names in
labels.json: ["cat","dog"]. Relative image paths resolve beside the CSV.
The adapter checks duplicate resolved paths. Copies of the same image at different
paths require your own content-based leakage audit. Weights download only when
you explicitly run the adapter. The optional model path requires separate validation
with your chosen checkpoint.

## Scope and prior art

[OpenCLIP](https://github.com/mlfoundations/open_clip) supplies pretrained encoders.
[OpenOOD](https://github.com/Jingkang50/OpenOOD) supplies a broad OOD research benchmark.
This package focuses on a small deployment audit for already cached CLIP features.
Scoring rules and order-statistic calibration are established methods; this alpha
claims an engineering workflow contribution. It has no SOTA claim.

Threshold validity depends on exchangeability of known calibration and future
known samples. Distribution shift can invalidate coverage. Unknown recall has no
calibration guarantee. Recalibrate after changing class vocabulary, prompts, model,
temperature, or score. See docs/protocol.md.

## Roadmap

- Real-image evaluation with published split hashes and checkpoint provenance.
- Class-conditional diagnostics and per-session class-incremental reports.
- Bootstrap intervals and repeated-seed comparisons.
- Community adapters for SigLIP and OpenOOD exports.

MIT licensed. Initial implementation developed with AI assistance.

## Recorded local validation

9 tests passed after installation on Windows / Python 3.12.14.
[Validation record](docs/validation.json) · [Demo result](docs/demo-result.json)

On the seeded synthetic fixture, cosine AUROC = 0.9935, FPR@95 = 0.0125, and known acceptance = 0.80. The shifted known-test distribution illustrates why nominal calibration coverage can fail after shift. These values do not describe a real image benchmark.
