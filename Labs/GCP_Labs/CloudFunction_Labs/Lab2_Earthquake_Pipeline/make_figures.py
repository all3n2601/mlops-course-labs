"""Draw the README figures from the evidence the cloud run saved (evidence/*.json, evidence/*.txt)
and from data/usgs_earthquakes_2025.csv. Nothing here is typed in by hand."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})

metrics = json.load(open("evidence/metrics.json"))
quality = json.load(open("evidence/quality_report.json"))


def label_bars(ax, bars, fmt="{:.3f}", pad=0.01, horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        x, y = (v + pad, b.get_y() + b.get_height() / 2) if horizontal else (b.get_x() + b.get_width() / 2, v + pad)
        ax.text(x, y, fmt.format(v), ha="left" if horizontal else "center",
                va="center" if horizontal else "bottom", color=INK, fontsize=10)


# 1. model results -------------------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
runs = metrics["validation_runs"]
names = list(runs)
vals = [runs[n]["f1"] for n in names]
colors = [BLUE if n == metrics["selected"] else GRAY for n in names]
bars = a.barh(names, vals, color=colors, height=0.5)
a.invert_yaxis(); a.set_xlim(0, 1); a.set_xlabel("F1 on the validation month (June 2025)")
a.set_title("Three configurations compared"); label_bars(a, bars, horizontal=True)
a.text(0.0, -0.32, f"blue = selected ({metrics['selected']})", transform=a.transAxes, color=MUTED, fontsize=9.5)
a.grid(axis="x", color=GRID, linewidth=0.8); a.set_axisbelow(True)

kinds = ["f1", "precision", "recall"]
x = range(len(kinds)); w = 0.36
m1 = b.bar([i - w / 2 for i in x], [metrics["test"][k] for k in kinds], w - 0.04, color=BLUE, label=f"selected: {metrics['selected']}")
m2 = b.bar([i + w / 2 for i in x], [metrics["test_logreg_baseline"][k] for k in kinds], w - 0.04, color=ORANGE, label="baseline: logistic regression")
b.set_xticks(list(x), ["F1", "precision", "recall"]); b.set_ylim(0, 1.0)
b.set_title(f"Unseen test months: Jul to Dec 2025 ({metrics['rows']['test']:,} events)")
label_bars(b, m1, "{:.2f}"); label_bars(b, m2, "{:.2f}")
b.legend(frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.12), fontsize=9.5)
b.grid(axis="y", color=GRID, linewidth=0.8); b.set_axisbelow(True)
fig.tight_layout(); fig.savefig("images/model_results.png", dpi=150); plt.close(fig)

# 2. the data -------------------------------------------------------------------------------
df = pd.read_csv("data/usgs_earthquakes_2025.csv", parse_dates=["time"])
df["month"] = df["time"].dt.month
counts = df.groupby("month").size()
period = lambda m: "train (Jan to May)" if m <= 5 else ("validation (Jun)" if m == 6 else "test (Jul to Dec)")
colour = {"train (Jan to May)": BLUE, "validation (Jun)": ORANGE, "test (Jul to Dec)": AQUA}
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
bars = a.bar(counts.index, counts.values, color=[colour[period(m)] for m in counts.index], width=0.7)
for bar, v in zip(bars, counts.values):
    a.text(bar.get_x() + bar.get_width() / 2, v + 40, f"{v:,}", ha="center", va="bottom", fontsize=8.5)
a.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
a.set_ylim(0, counts.max() * 1.12); a.set_title("Events per month, and how the time split uses them")
a.set_ylabel("earthquakes M2.5+")
a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colour.values()], labels=list(colour), frameon=False,
         loc="upper left", fontsize=9.5)
a.grid(axis="y", color=GRID, linewidth=0.8); a.set_axisbelow(True)
miss = quality["missing_by_feature"]
bars = b.bar(list(miss), list(miss.values()), color=BLUE, width=0.6)
for bar, v in zip(bars, miss.values()):
    b.text(bar.get_x() + bar.get_width() / 2, v + 25, f"{v:,}" if v else "0", ha="center", va="bottom", fontsize=9.5)
b.set_ylim(0, max(miss.values()) * 1.2)
b.set_title(f"Missing readings (of {quality['rows_out']:,} rows)"); b.tick_params(axis="x", labelrotation=30)
b.grid(axis="y", color=GRID, linewidth=0.8); b.set_axisbelow(True)
fig.tight_layout(); fig.savefig("images/data_overview.png", dpi=150); plt.close(fig)

# 3. the saved command output, drawn as an image -------------------------------------------
def terminal(path, title, text, out):
    import textwrap
    lines = [w for l in text.rstrip().splitlines() for w in (textwrap.wrap(l, 108, subsequent_indent="    ") or [""])]
    step = 0.25
    fig = plt.figure(figsize=(11, step * len(lines) + 0.9))
    fig.patch.set_facecolor("#1b1b1a")
    fig.text(0.015, 1 - 0.28 / fig.get_figheight(), title, color="#9d9c95", fontsize=9.5, family="monospace", va="top")
    for i, line in enumerate(lines):
        c = "#8fd18f" if line.startswith("##") else ("#f0efec")
        fig.text(0.015, 1 - (0.75 + step * i) / fig.get_figheight(), line, color=c, fontsize=9.5, family="monospace", va="top")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)


wf = open("evidence/workflow_run.yaml").read()
result = json.loads(next(l for l in wf.splitlines() if l.startswith("result:")).split("result: ", 1)[1].strip("'"))
state = next(l for l in wf.splitlines() if l.startswith("state:")).split(": ")[1]
start = next(l for l in wf.splitlines() if l.startswith("startTime:")).split(": ")[1].strip("'")
end = next(l for l in wf.splitlines() if l.startswith("endTime:")).split(": ")[1].strip("'")
secs = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
txt = [f"## workflow lab6-earthquake-pipeline: state {state}, {secs:.0f} s",
       f"selected model : {result['selected_model']}",
       f"test scores    : F1 {result['test']['f1']}  precision {result['test']['precision']}  recall {result['test']['recall']}",
       "sample event 1 : Kamchatka region, full readings   -> "
       f"P(M4.5+) = {result['predictions'][0]['probability_m45_or_higher']}  flag = {result['predictions'][0]['predicted_m45_or_higher']}",
       "sample event 2 : southern California, few readings -> "
       f"P(M4.5+) = {result['predictions'][1]['probability_m45_or_higher']}  flag = {result['predictions'][1]['predicted_m45_or_higher']}"]
terminal("evidence/workflow_run.yaml", "gcloud workflows run   (from evidence/workflow_run.yaml)", "\n".join(txt), "images/workflow_run.png")
ev = open("evidence/cloud_evidence.txt").read()
terminal("evidence/cloud_evidence.txt", "deployed resources and endpoint checks   (from evidence/cloud_evidence.txt)", ev, "images/cloud_evidence.png")
print("ok")
