import argparse
import html
import json
from pathlib import Path
import numpy as np
from .core import audit


def demo_data(seed=23):
    rng = np.random.default_rng(seed)
    classes, dim = 6, 32
    text = rng.normal(size=(classes, dim))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    cal_y = rng.integers(classes, size=240)
    known_y = rng.integers(classes, size=240)
    cal = text[cal_y] + rng.normal(0, .18, (240, dim))
    known = text[known_y] + rng.normal(0, .22, (240, dim))
    unknown = rng.normal(size=(160, dim))
    return text, cal, np.vstack([known, unknown]), np.r_[known_y, np.full(160, -1)]


def write_report(report, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(payload + "\n", encoding="utf-8")
    rows = []
    for name, m in report["methods"].items():
        rows.append(f"<tr><td>{name}</td><td>{m['auroc']:.4f}</td>"
                    f"<td>{m['fpr_at_95_tpr']:.4f}</td>"
                    f"<td>{m['known_acceptance']:.4f}</td>"
                    f"<td>{m['unknown_acceptance']:.4f}</td></tr>")
    page = """<!doctype html><meta charset="utf-8"><title>CLIP Shift Audit</title>
<style>body{font:16px system-ui;max-width:1100px;margin:50px auto;padding:24px;background:#101827;color:#e5ecfa}
h1{color:#68e0c3}table{border-collapse:collapse;width:100%}td,th{padding:14px;text-align:left;border-bottom:1px solid #455}
pre{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#abb}</style><h1>CLIP Shift Audit</h1>"""
    page += f"<p>{html.escape(report.get('data_note', report['protocol']))}</p>"
    page += "<table><tr><th>Score</th><th>AUROC ↑</th><th>FPR@95 ↓</th><th>Known accept</th><th>Unknown accept ↓</th></tr>"
    page += "".join(rows) + "</table><details><summary>Full reproducible report</summary><pre>"
    page += html.escape(payload) + "</pre></details>"
    path.with_suffix(".html").write_text(page, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit CLIP embeddings on held-out data.")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=23)
    demo.add_argument("--output", default="outputs/demo.json")
    run = sub.add_parser("audit")
    run.add_argument("input", help="NPZ: text, calibration, test, labels")
    run.add_argument("--alpha", type=float, default=.05)
    run.add_argument("--temperature", type=float, default=.01)
    run.add_argument("--output", default="outputs/audit.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            report = audit(*demo_data(args.seed))
            report.update(seed=args.seed, data_note="Synthetic normalized feature demo. No real CLIP inference or benchmark claim.")
        else:
            with np.load(args.input, allow_pickle=False) as data:
                report = audit(data["text"], data["calibration"], data["test"],
                               data["labels"], alpha=args.alpha, temperature=args.temperature)
        write_report(report, args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({k: {m: v for m,v in row.items() if m != "risk_coverage"}
                      for k,row in report["methods"].items()}, indent=2))
    return 0
