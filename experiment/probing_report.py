"""Create standalone HTML comparisons of completed and pending attentive probes."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment.utils import PROJECT_ROOT


VARIANTS = ("raw", "layer-3", "layer-6", "layer-9", "layer-12")
LABELS = {"raw": "Raw motion", **{f"layer-{n}": f"JEPA layer {n}" for n in (3, 6, 9, 12)}}
COLORS = {"raw": "#475569", "layer-3": "#2563eb", "layer-6": "#7c3aed",
          "layer-9": "#0d9488", "layer-12": "#d97706"}


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def numeric(value):
    number = finite(value)
    return "—" if number is None else f"{number:.9f}"


def read_run(task, directory, name):
    summary = read_json(directory / "summary.json")
    config = read_json(directory / "config.json") or summary.get("signature", {})
    optimization = config.get("optimization", {})
    budget = int(optimization.get("epochs", 100))
    points = []
    if task == "segmentation":
        history = read_json(directory / "metrics.json")
        for row in history if isinstance(history, list) else []:
            points.append({
                "epoch": int(row["epoch"]), "train": finite(row["train"].get("loss")),
                "val": finite(row["val"].get("babel-120", {}).get("frame_map")),
                "val_loss": finite(row["val"].get("loss")),
            })
        best_score = finite(summary.get("best_val_frame_map"))
        test = summary.get("test")
        test_score = finite((test or {}).get("babel-120", {}).get("frame_map"))
        epochs = int(summary.get("actual_epochs", max((p["epoch"] for p in points), default=0)))
        feature_metadata = config.get("features", {})
    else:
        metrics_path = directory / "metrics.csv"
        if metrics_path.is_file():
            with metrics_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if finite(row.get("epoch")) is None:
                        continue
                    points.append({
                        "epoch": int(row["epoch"]), "train": finite(row.get("train_loss")),
                        "val": finite(row.get("val_mean_r1")),
                        "t2m": finite(row.get("val_t2m_r1")), "m2t": finite(row.get("val_m2t_r1")),
                    })
        best_score = finite(summary.get("best_val", {}).get("mean_r1"))
        test = summary.get("test")
        test_score = finite((test or {}).get("mean_r1"))
        epochs = int(summary.get("epochs_completed", max((p["epoch"] for p in points), default=0)))
        feature_metadata = read_json(directory / "provenance.json").get("probe", {}).get("extractor", {})
    best_epoch = summary.get("best_epoch")
    observed = [p for p in points if p["val"] is not None]
    if best_score is None and observed:
        peak = max(observed, key=lambda point: point["val"])
        best_score, best_epoch = peak["val"], peak["epoch"]
    final_score = points[-1]["val"] if points else None
    stop = read_json(directory / "early_stop.json")
    failure = read_json(directory / "failure.json")
    if failure:
        status, complete = "Failure recorded", False
    elif stop:
        status, complete = "Intentional early stop", False
    elif summary:
        status, complete = ("Complete", True) if epochs >= budget else ("Finished below budget", False)
    elif points or config:
        status, complete = "Progress recorded", False
    else:
        status, complete = "Pending", False
    return {
        "name": name, "label": LABELS.get(name, name), "directory": directory,
        "summary": summary, "config": config, "optimization": optimization,
        "budget": budget, "epochs": epochs, "points": points, "status": status,
        "complete": complete, "best_epoch": best_epoch, "best": best_score,
        "final": final_score, "test_score": test_score, "test": test,
        "feature_metadata": feature_metadata, "stop": stop, "failure": failure,
    }


def matched(left, right):
    """Require completed matching recipes before presenting raw-probe deltas."""
    if not left["complete"] or not right["complete"] or left["best"] is None or right["best"] is None:
        return False
    fields = ("epochs", "seed", "batch_size", "warmup_epochs", "lr", "weight_decay")
    return all(left["optimization"].get(key) == right["optimization"].get(key) for key in fields)


def link(path, report_dir, label):
    relative = os.path.relpath(path, report_dir)
    return f'<a href="{html.escape(relative, quote=True)}">{html.escape(label)}</a>'


def plot_curves(task, runs):
    shown = [run for run in runs if run["points"]]
    if not shown:
        return '<p class="muted">Learning curves will appear once epoch metrics are available.</p>'
    panels = (("val", "Validation BABEL-120 frame mAP"), ("train", "Training masked BCE"),
              ("val_loss", "Validation masked BCE")) if task == "segmentation" else (
              ("val", "Validation mean bidirectional R@1"), ("train", "Training contrastive loss"),
              ("t2m", "Validation text-to-motion R@1"), ("m2t", "Validation motion-to-text R@1"))
    columns = 2 if task == "tmr" else 3
    rows = 2 if task == "tmr" else 1
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#334155", "text.color": "#0f172a",
                         "axes.edgecolor": "#cbd5e1", "grid.color": "#e2e8f0"}):
        figure, axes = plt.subplots(rows, columns, figsize=(13.8, 7.8 if rows == 2 else 4.0), squeeze=False)
        for axis, (metric, title) in zip(axes.flat, panels):
            for run in shown:
                values = [p for p in run["points"] if p.get(metric) is not None]
                if values:
                    axis.plot([p["epoch"] for p in values], [p[metric] for p in values],
                              color=COLORS[run["name"]], label=run["label"], linewidth=1.7)
            axis.set(title=title, xlabel="Epoch")
            axis.grid(alpha=.7, linewidth=.6)
            axis.set_xlim(left=0)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .005))
        figure.tight_layout(rect=(0, .075 if rows == 1 else .055, 1, 1))
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=150, facecolor="white")
        plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img class="curves" src="data:image/png;base64,{encoded}" alt="Complete logged training and validation learning curves for the raw probe and each JEPA layer">'


def conclusions(runs):
    raw = runs[0]
    completed = [run for run in runs if run["complete"]]
    statements = [f"{len(completed)} of {len(runs)} planned runs have completed their recorded training budgets."]
    compared = [run for run in runs[1:] if matched(run, raw)]
    if compared:
        winner = max([raw, *compared], key=lambda run: run["best"])
        statements.append(f"Among completed runs with the same recipe, {winner['label']} has the highest selected validation score ({winner['best']:.9f}).")
        for run in compared:
            delta = 100 * (run["best"] - raw["best"])
            direction = "above" if delta > 0 else "below" if delta < 0 else "equal to"
            statements.append(f"{run['label']}: {abs(delta):.4f} percentage points {direction} the matched raw-motion probe.")
    else:
        statements.append("A completed matched raw-versus-JEPA comparison is not available yet.")
    if len(completed) < len(runs):
        statements.append("Layer ordering is provisional until the remaining runs finish; intermediate peaks are shown for progress only.")
    for run in completed:
        points = run["points"]
        selected = next((point for point in points if point["epoch"] == run["best_epoch"]), None)
        final = points[-1] if points else None
        if (selected and final and run["best"] is not None and final["val"] is not None
                and selected["train"] is not None and final["train"] is not None
                and final["val"] < run["best"] and final["train"] < selected["train"]):
            delta = 100 * (run["best"] - final["val"])
            statements.append(
                f"{run['label']}: final validation is {delta:.4f} percentage points below its selected peak, "
                f"while training loss decreased from {selected['train']:.6f} at the selected epoch to "
                f"{final['train']:.6f}. This train/validation divergence is consistent with overfitting after the peak; "
                "the best checkpoint remains the evaluation model."
            )
    statements.append("These are single-seed downstream results. Differences do not establish uncertainty across seeds, and the full supervised Transformer is a separate architecture reference.")
    return "".join(f"<li>{html.escape(item)}</li>" for item in statements)


def generate(task, run_root, report_dir, checkpoint, comparison_root=None):
    run_root, report_dir, checkpoint = (Path(path).expanduser().resolve() for path in (run_root, report_dir, checkpoint))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen checkpoint does not exist: {checkpoint}")
    runs = [read_run(task, run_root / variant / "seed-42", variant) for variant in VARIANTS]
    reference_dir = PROJECT_ROOT / f"output/{task}/base-supervised-20261017/seed-42"
    reference = read_run(task, reference_dir, "Supervised base Transformer")
    raw = runs[0]
    title = "Segmentation attentive probing" if task == "segmentation" else "Text-motion retrieval attentive probing"
    metric = "BABEL-120 frame mAP" if task == "segmentation" else "Mean bidirectional R@1"
    completed = sum(run["complete"] for run in runs)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    source = next((run["feature_metadata"] for run in runs[1:] if run["feature_metadata"]), {})
    source_step = source.get("checkpoint_step", "200000 (planned; source metadata pending)")
    rows, details, configs = [], [], []
    for run in runs:
        change = (f"{100 * (run['best'] - raw['best']):+.4f}" if run["name"] != "raw" and matched(run, raw) else "—")
        badge = "complete" if run["complete"] else "pending"
        best = numeric(run["best"])
        if run["best"] is not None and not run["complete"]:
            best += '<span class="small">observed so far</span>'
        rows.append(f"<tr><th>{html.escape(run['label'])}</th><td><span class='badge {badge}'>{run['status']}</span></td>"
                    f"<td>{run['epochs']} / {run['budget']}</td><td>{run['best_epoch'] or '—'}</td><td>{best}</td>"
                    f"<td>{numeric(run['final'])}</td><td>{change}</td><td>{numeric(run['test_score'])}</td></tr>")
        optimization = run["optimization"]
        actual = bool(optimization)
        planned = {"seed": 42, "epochs": 100, "warmup_epochs": 5, "batch_size": 256,
                   "lr": 3e-4 if task == "segmentation" else 1e-4, "weight_decay": .01}
        recipe = {key: optimization.get(key, value) for key, value in planned.items()}
        configs.append(f"<tr><th>{html.escape(run['label'])}</th><td>{'Recorded' if actual else 'Planned'}</td>"
                       + "".join(f"<td>{recipe[key]:g}</td>" for key in planned) + "</tr>")
        artifacts = []
        for filename in ("best.pth.tar", "latest.pth.tar", "summary.json", "config.json",
                         "metrics.json" if task == "segmentation" else "metrics.csv", "provenance.json",
                         "early_stop.json", "failure.json"):
            path = run["directory"] / filename
            if path.is_file():
                artifacts.append(link(path, report_dir, filename))
        exact = {"status": run["status"], "actual_epochs": run["epochs"], "budget_epochs": run["budget"],
                 "best_epoch": run["best_epoch"], "best_validation": run["best"], "final_validation": run["final"],
                 "test": run["test"], "feature_metadata": run["feature_metadata"],
                 "parameters": run["summary"].get("parameters", run["summary"].get("parameter_counts")),
                 "stop_record": run["stop"] or None, "failure_record": run["failure"] or None}
        details.append(f"<details><summary>{html.escape(run['label'])}: exact values and artifacts</summary>"
                       f"<p>{' · '.join(artifacts) if artifacts else 'No saved artifacts yet.'}</p>"
                       f"<p class='path'>{html.escape(str(run['directory']))}</p>"
                       f"<pre>{html.escape(json.dumps(exact, indent=2))}</pre></details>")
    if task == "segmentation":
        architecture = ("Each standardized frame queries all valid frames in one six-head self-attention layer. "
                        "A residual preserves frame identity, followed by one linear 120-class projection. "
                        "There is no feed-forward block or trainable JEPA backbone.")
        evaluation = ("Masked frame BCE trains annotated frames only. Validation BABEL-120 frame mAP selects the checkpoint. "
                      "All valid training frames, including unannotated context, contribute to feature statistics. "
                      "The prepared segmentation test split is empty, so test metrics are unavailable.")
    else:
        architecture = ("One learned query pools all valid motion frames through one six-head attention layer, followed by "
                        "one linear projection to the 256D retrieval space and L2 normalization. This replaces the entire "
                        "motion Transformer. The trainable base text encoder remains 12 layers, width 384, six heads, "
                        "receiving frozen 4096D LLM2Vec sentence features.")
        evaluation = ("Symmetric multi-positive contrastive loss trains the motion probe and text encoder. "
                      "Full-gallery validation mean bidirectional R@1 selects the checkpoint; selected-model test recall "
                      "is reported when available. Text-to-motion and motion-to-text curves remain separate in the figure.")
    reference_text = (f"The earlier supervised base Transformer completed {reference['epochs']} epochs, selecting epoch "
                      f"{reference['best_epoch'] or '—'} with validation {metric} {numeric(reference['best'])}; "
                      f"final validation {numeric(reference['final'])}; selected-checkpoint test {numeric(reference['test_score'])}.")
    reference_artifacts = " · ".join(link(reference_dir / file, report_dir, file) for file in
                                    ("summary.json", "best.pth.tar", "metrics.json" if task == "segmentation" else "metrics.csv")
                                    if (reference_dir / file).is_file())
    parameter_rows = []
    for run in [*runs, reference]:
        counts = run["summary"].get("parameters", run["summary"].get("parameter_counts", {}))
        total = counts.get("total", run["summary"].get("trainable_parameters"))
        total_m = "—" if total is None else f"{total / 1_000_000:.3f}"
        components = "; ".join(f"{key}: {value / 1_000_000:.3f} M"
                               for key, value in counts.items() if key != "total")
        parameter_rows.append(f"<tr><th>{html.escape(run['label'])}</th><td>{total_m}</td>"
                              f"<td>{html.escape(components) or '—'}</td></tr>")
    comparison_section = ""
    if comparison_root is not None:
        comparison_root = Path(comparison_root).expanduser().resolve()
        comparison_rows = []
        for run in runs[1:]:
            prior = read_run(task, comparison_root / run["name"] / "seed-42", run["name"])
            delta = (f"{100 * (run['best'] - prior['best']):+.4f}" if matched(run, prior) else "—")
            comparison_rows.append(
                f"<tr><th>{html.escape(run['label'])}</th><td>{numeric(prior['best'])}</td>"
                f"<td>{numeric(run['best'])}</td><td>{delta}</td>"
                f"<td>{numeric(prior['test_score'])}</td><td>{numeric(run['test_score'])}</td></tr>")
        comparison_section = (
            "<h2>Prior JEPA checkpoint comparison</h2>"
            "<p>The current checkpoint uses centering and sharpening; the prior checkpoint uses the baseline objective. "
            "Both were trained for 200,000 updates. Raw-motion controls are reused from the prior suite. "
            "Differences are reported only after both probes finish matching recorded training recipes.</p>"
            "<div class='scroll'><table><thead><tr><th>Layer</th><th>Prior best val</th>"
            "<th>Current best val</th><th>Δ prior (pp)</th><th>Prior test</th><th>Current test</th>"
            "</tr></thead><tbody>" + "".join(comparison_rows) + "</tbody></table></div>"
            f"<p class='path'>Prior suite: {html.escape(str(comparison_root))}</p>")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--blue:#2563eb}}
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.65 system-ui,sans-serif}}
main{{max-width:1340px;margin:36px auto;padding:40px 46px;background:white;border:1px solid var(--line);border-radius:18px}}
h1{{font-size:36px;line-height:1.2;margin:8px 0 16px;letter-spacing:-1px}}h2{{font-size:22px;margin:34px 0 12px}}
p{{max-width:1100px}}.eyebrow{{color:var(--blue);font-weight:750;letter-spacing:1.5px;text-transform:uppercase;font-size:12px}}
.meta,.muted{{color:var(--muted)}}.notice{{border-left:4px solid var(--blue);background:#eff6ff;padding:14px 18px;margin:24px 0}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}.card{{border:1px solid var(--line);padding:18px;border-radius:10px}}
.card strong{{display:block;font-size:24px}}.card span{{color:var(--muted);font-size:13px}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}th,td{{padding:12px 10px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead th{{font-weight:650;color:#475569;background:#f8fafc}}tbody th{{font-weight:650}}
.badge{{border-radius:6px;padding:4px 7px;font-size:11px;background:#f1f5f9;color:#475569}}.badge.complete{{background:#ecfdf5;color:#047857}}
.small{{display:block;font-size:10px;color:var(--muted)}}.curves{{width:100%;height:auto;margin-top:10px}}
a{{color:#1d4ed8;text-decoration:none}}a:hover{{text-decoration:underline}}details{{border:1px solid var(--line);border-radius:9px;margin:10px 0;padding:13px 16px}}
summary{{font-weight:650;cursor:pointer}}pre{{font-size:12px;overflow:auto;background:#f8fafc;padding:16px;border-radius:8px;line-height:1.55}}
.path{{font:12px/1.6 ui-monospace,monospace;overflow-wrap:anywhere;color:var(--muted)}}li{{margin:8px 0}}
footer{{border-top:1px solid var(--line);padding-top:18px;margin-top:30px;color:var(--muted);font-size:12px}}
@media(max-width:800px){{main{{margin:0;padding:24px 18px;border-radius:0}}h1{{font-size:29px}}.cards{{grid-template-columns:1fr}}}}
@media print{{body{{background:white}}main{{border:0;margin:0;padding:10px;max-width:none}}details{{break-inside:avoid}}.scroll{{overflow:visible}}}}
</style></head><body><main>
<div class="eyebrow">Motion-JEPA · downstream evaluation</div><h1>{title}</h1>
<p class="meta">Report generated {html.escape(timestamp)} · Single seed: 42 · Base encoder</p>
<p>Measure which frozen JEPA layer supports this task with a shallow attentive probe, using raw motion with the same probe as the matched baseline.</p>
<div class="cards"><div class="card"><strong>{completed} / 5</strong><span>Runs with completed recorded budgets</span></div>
<div class="card"><strong>3 · 6 · 9 · 12</strong><span>One-based frozen encoder layers</span></div>
<div class="card"><strong>{'Frame mAP' if task == 'segmentation' else 'Mean R@1'}</strong><span>Validation checkpoint selection</span></div></div>
<div class="notice">{'All five recorded training budgets are complete.' if completed == 5 else 'This report includes pending or incomplete runs. Observed peaks are progress indicators, not final layer comparisons.'}
“Progress recorded” describes saved metrics only; this report does not inspect process liveness.</div>
<h2>Layer comparison</h2><p class="muted">Scores are fractions on [0, 1]. Δ raw uses percentage points and is shown only for completed runs with matching recorded recipes.</p>
<div class="scroll"><table><thead><tr><th>Features</th><th>Status</th><th>Epochs</th><th>Best epoch</th><th>Best validation</th><th>Final validation</th><th>Δ raw (pp)</th><th>Test</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{comparison_section}
<h2>Trainable parameters</h2>
<p class="muted">M = 1,000,000 parameters; values are rounded to three decimals. Counts come from saved run summaries and exclude frozen JEPA and LLM2Vec encoders. TMR totals include both the motion probe and trainable text encoder. Exact integer counts remain in the saved artifacts.</p>
<div class="scroll"><table><thead><tr><th>Model / features</th><th>Total trainable (M)</th><th>Components (M)</th></tr></thead><tbody>{''.join(parameter_rows)}</tbody></table></div>
<h2>Assessment</h2><ul>{conclusions(runs)}</ul>
<h2>Full learning curves</h2><p class="muted">Every available epoch is plotted without smoothing or truncation. Test metrics do not select checkpoints.</p>{plot_curves(task, runs)}
<h2>Experimental protocol</h2><p>{architecture}</p><p>{evaluation}</p>
<p>Frozen JEPA features come from the EMA target encoder at update {html.escape(str(source_step))}. Layers are read immediately after their one-based Transformer block, before the encoder's final LayerNorm. Raw motion has 366 channels; base JEPA features have 384.</p>
<p>JEPA inputs use the checkpoint's pretraining mean/std. Probe inputs use per-channel population mean/std computed only from valid frames in the task's training dataset. Padding, validation and test frames never contribute. Channels with standard deviation below 10⁻⁶ use a divisor of one.</p>
<p>Per-sample motion features remain in anonymous CPU RAM throughout probing. The frozen network is released from GPU memory before constructing the probe. No per-sample motion feature cache is serialized. Saved probe checkpoints retain normalization buffers, parameters and provenance; TMR continues to use its existing frozen sentence-feature cache.</p>
<p class="path">Frozen checkpoint: {html.escape(str(checkpoint))}<br>Checkpoint SHA-256: {html.escape(str(source.get('checkpoint_sha256', 'pending extraction metadata')))}</p>
<div class="scroll"><table><thead><tr><th>Features</th><th>Recipe</th><th>Seed</th><th>Epoch budget</th><th>Warmup</th><th>Batch</th><th>Learning rate</th><th>Weight decay</th></tr></thead><tbody>{''.join(configs)}</tbody></table></div>
<p class="muted">AdamW, gradient clipping at 1.0, warmup followed by cosine decay. Planned values are explicitly labeled until a run writes its actual configuration. Each layer uses its own best validation checkpoint.</p>
<h2>Historical supervised reference</h2><p>{reference_text}</p><p>This reference trains the full raw-motion Transformer and has a different trainable capacity. It is shown separately and is not used for matched raw-probe deltas.</p><p>{reference_artifacts}</p>
<h2>Exact results and saved artifacts</h2>{''.join(details)}
<footer>Source run directory: {html.escape(str(run_root))}. All figures are embedded; this HTML report can be opened offline. Artifact links are relative to the report location.</footer>
</main></body></html>"""
    report_dir.mkdir(parents=True, exist_ok=True)
    temporary = report_dir / "index.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(report_dir / "index.html")
    return report_dir / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("segmentation", "tmr"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path)
    args = parser.parse_args()
    print(generate(args.task, args.run_root, args.report_dir, args.checkpoint, args.comparison_root))


if __name__ == "__main__":
    main()
