"""Presentation only: read hash-checked frozen CSVs; never import model code."""
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent
STAGES = {
    "eeg": ("phase2_eeg", "3ba65f49586b8c13e4e497ba0aa2e160bfed9d1cdd64434bb72a131f72a47b21"),
    "ecg": ("phase3_ecg", "6813d261b922f82721d27bccdfa452135a56c4d47c3afe5f7bc72506a0c4737e"),
    "fusion": ("phase4_fusion", "ae451eb4142626cd35daa0e4309a64ece2f56fd43b242c0a6b5b1e55cfb9ebea"),
}
PATIENTS = ("PN00", "PN06", "PN10", "PN12", "PN14")
COLORS = {"EEG RF": "#26547C", "ECG RF": "#B8860B", "Adaptive fusion": "#527A47"}
SOURCES = {}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(stage, name):
    directory, identity = STAGES[stage]
    folder = ROOT / "results" / directory / identity[:12]
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = json.loads(manifest_path.with_suffix(".json.provenance.json").read_text(encoding="utf-8"))
    assert manifest["experiment_identity"] == identity
    assert sha256(manifest_path) == sidecar["artifact_sha256"]
    path = folder / name
    relative = path.relative_to(ROOT).as_posix()
    expected = next(row["sha256"] for row in manifest["artifacts"] if row["path"] == relative)
    assert sha256(path) == expected, "Frozen input hash mismatch: " + relative
    SOURCES[relative] = expected
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric(stage, model):
    return next(row for row in read(stage, "pooled_metrics.csv") if row["model"] == model)


def result_table():
    systems = [("EEG RF", "eeg", "E1_RF"), ("ECG RF", "ecg", "C1_RF"),
               ("Fixed late fusion (F1)", "fusion", "F1"), ("Adaptive late fusion (F2; primary)", "fusion", "F2"),
               ("Early fusion (F3)", "fusion", "F3")]
    expected = [(13, 69), (4, 7), (11, 89), (13, 68), (11, 95)]
    lines = ["| System | Detected | Sensitivity | False declarations | FAR/h | Median delay |",
             "|---|---:|---:|---:|---:|---:|"]
    for (title, stage, model), counts in zip(systems, expected):
        row = metric(stage, model)
        assert (int(row["detected"]), int(row["false_declarations"])) == counts
        assert int(row["events"]) == 27
        delay = f"{float(row['delay_median_s']):g} s" + ("*" if model == "F3" else "")
        lines.append(f"| {title} | {row['detected']}/27 | {float(row['sensitivity'])*100:.2f}% | {row['false_declarations']} | {float(row['far_per_h']):.3f} | {delay} |")
    fates = collections.Counter(row["classification"] for row in read("fusion", "event_comparison.csv"))
    assert dict(fates) == dict(preserved=12, rescued=1, lost=1, still_missed=13)
    alpha = collections.Counter(float(row["alpha"]) for row in read("fusion", "selected_parameters.csv"))
    assert [alpha[a] for a in (0, .1, .25, .5)] == [15, 0, 5, 2]
    assert len(read("fusion", "selected_parameters.csv")) == 22
    return "\n".join(lines)


def check_docs():
    expected = result_table()
    for name in ("README.md", "docs/FINAL_PROJECT_SUMMARY.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        actual = text.split("<!-- frozen-results:start -->", 1)[1].split("<!-- frozen-results:end -->", 1)[0].strip()
        assert actual == expected, "Result table mismatch: " + name
    print("DOCUMENT_TABLES_PASS 2/2; model counts, 27 events, 22 parents, event fates and alpha distribution")


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 15,
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "text.color": "#222222", "axes.labelcolor": "#222222",
        "svg.hashsalt": "siena-frozen-closeout", "savefig.facecolor": "white"})
    destination = ROOT / "docs/figures"
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(fig, name):
        for extension in ("png", "svg"):
            path = destination / (name + "." + extension)
            metadata = {"Date": None} if extension == "svg" else {"Software": "Matplotlib; frozen Siena results"}
            fig.savefig(path, dpi=180, metadata=metadata)
            outputs.append(path)
        plt.close(fig)

    systems = [("EEG RF", "eeg", "E1_RF"), ("ECG RF", "ecg", "C1_RF"), ("Adaptive fusion", "fusion", "F2")]
    data = {title: {row["patient"]: row for row in read(stage, "patient_metrics.csv") if row["model"] == model}
            for title, stage, model in systems}
    assert all(set(rows) == set(PATIENTS) for rows in data.values())
    for field, ylabel, name, title in [
        ("sensitivity", "Strict event sensitivity (%)", "02_patient_sensitivity", "Seizure detection by patient"),
        ("far_per_h", "False / unmatched declarations per hour", "03_patient_far", "False-alarm rate by patient")]:
        fig, ax = plt.subplots(figsize=(10, 5.6))
        fig.subplots_adjust(left=.10, right=.98, top=.76, bottom=.20)
        positions = np.arange(len(PATIENTS)); width = .24
        for index, (label, _, _) in enumerate(systems):
            rows = [data[label][p] for p in PATIENTS]
            values = [float(row[field]) * (100 if field == "sensitivity" else 1) for row in rows]
            bars = ax.bar(positions + (index-1)*width, values, width, label=label,
                color=COLORS[label], hatch=("", "///", "..")[index], edgecolor="white", linewidth=.5)
            labels = [f"{row['detected']}/{row['events']}" if field == "sensitivity" else f"{float(row[field]):.2f}" for row in rows]
            ax.bar_label(bars, labels=labels, padding=4, fontsize=10)
        ax.set_xticks(positions, PATIENTS); ax.set_ylabel(ylabel)
        ax.set_ylim(0, 112 if field == "sensitivity" else 3.4)
        if field == "sensitivity": ax.set_yticks([0,25,50,75,100])
        ax.yaxis.grid(True, color="#E5E5E5", linewidth=.7); ax.set_axisbelow(True)
        fig.suptitle(title, x=.1, ha="left", y=.96, fontsize=17)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.09,.90), ncol=3, frameon=False)
        note = "Labels show detected / eligible events; 5 patients, 22 whole-parent held-out recordings, 27 events."
        if field == "far_per_h": note = "FAR uses each modality's evaluable time: EEG/F2 59.81 h; ECG 56.17 h. Different denominators."
        fig.text(.1,.06,note,fontsize=10,color="#555555")
        save(fig,name)

    fate = collections.Counter(row["classification"] for row in read("fusion", "event_comparison.csv"))
    fig,ax=plt.subplots(figsize=(9,4.9));fig.subplots_adjust(left=.28,right=.94,top=.79,bottom=.18)
    labels=["Preserved EEG detection","New rescue","Lost EEG detection","Still missed"]
    values=[fate[k] for k in ("preserved","rescued","lost","still_missed")]
    bars=ax.barh(labels,values,color=["#26547C","#B8860B","#C3623A","#8A8A8A"],height=.62)
    ax.bar_label(bars,padding=6,fontsize=12);ax.invert_yaxis();ax.set_xlim(0,15);ax.set_xticks(range(0,16,3))
    ax.set_xlabel("Eligible seizures (n = 27)");ax.xaxis.grid(True,color="#E5E5E5");ax.set_axisbelow(True)
    fig.suptitle("Adaptive fusion: event-level fate",x=.08,ha="left",y=.96,fontsize=17)
    fig.text(.08,.86,"13 detections before and after conceal one rescue and one lost EEG detection.",fontsize=11)
    fig.text(.08,.05,"F2 is the prespecified primary multimodal analysis; counts are descriptive.",fontsize=10,color="#555555")
    save(fig,"04_event_fate")

    scores=[r for r in read("fusion","scores/PN06__PN06-1.edf.csv") if r["model"]=="F2"]
    matches=read("fusion","event_matches.csv")
    event=next(r for r in matches if r["event_id"]=="PN06-1" and r["model"]=="F0")
    onset,offset=float(event["onset_s"]),float(event["offset_s"])
    rows=[r for r in scores if onset-90<=float(r["anchor_s"])<=offset+90]
    times=np.array([float(r["anchor_s"])-onset for r in rows])
    cardiac=next(r for r in read("ecg","fold_metrics.csv") if r["test_file"]=="PN06-1.edf" and r["model"]=="C1_RF")
    fig,axes=plt.subplots(3,1,figsize=(10,8.2),sharex=True)
    fig.subplots_adjust(left=.12,right=.97,top=.85,bottom=.16,hspace=.32)
    for ax,column,label,color,threshold in zip(axes,["score_e","score_c","score"],
        ["EEG RF","ECG RF","Adaptive fusion"],COLORS.values(),
        [[float(r['eeg_threshold']) for r in rows],[float(cardiac['threshold'])]*len(rows),[float(r['effective_threshold']) for r in rows]]):
        values=[float(r[column]) if r[column] else np.nan for r in rows]
        ax.axvspan(0,offset-onset,color="#E6C66A",alpha=.25,label="Seizure interval")
        ax.step(times,values,where="post",color=color,linewidth=1.8,label=label+" score")
        ax.step(times,threshold,where="post",color="#444444",linestyle="--",linewidth=1.2,label="Applied threshold")
        ax.set_ylim(0,1.05);ax.set_yticks([0,.5,1]);ax.set_ylabel("Score")
        ax.set_title(label,loc="left",fontsize=12);ax.yaxis.grid(True,color="#E5E5E5")
    axes[0].axvline(float(event['delay_s']),color=COLORS['EEG RF'],linestyle=":")
    axes[0].annotate("EEG declaration: +42 s",xy=(42,.97),xytext=(65,.58),fontsize=10,
        arrowprops=dict(arrowstyle="->",color="#555555"))
    axes[-1].set_xlim(-90,offset-onset+90);axes[-1].set_xlabel("Seconds relative to canonical seizure onset")
    fig.suptitle("PN06-1: an EEG detection lost under adaptive fusion",x=.08,ha="left",y=.97,fontsize=16)
    fig.text(.08,.91,"Frozen outer-held scores | alpha = 0.25 | ECG available throughout the seizure",fontsize=11)
    axes[1].legend(loc="upper left",bbox_to_anchor=(0,.75),fontsize=9,frameon=False,ncol=3)
    fig.text(.08,.055,"Low ECG scores lower the convex fused score. F2 does not declare within the seizure interval.\nScores are not calibrated probabilities; lines hold each decision for its 5-second cell.",fontsize=10,color="#555555")
    save(fig,"05_pn06_1_failure")

    manifest=dict(purpose="Presentation figures only; no fitting, model imports or frozen-file writes",
        experiment_identities={k:v[1] for k,v in STAGES.items()},generator_sha256=sha256(Path(__file__)),
        python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in
            ("matplotlib","numpy","pillow","contourpy","cycler","fonttools","kiwisolver","pyparsing")},
        sources=SOURCES,outputs={p.relative_to(ROOT).as_posix():sha256(p) for p in outputs})
    (destination/"figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("FIGURES_COMPLETE 4 figures, PNG+SVG; source hashes and renderer versions recorded")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-docs",action="store_true")
    parser.add_argument("--print-table",action="store_true")
    args=parser.parse_args()
    if args.check_docs:check_docs()
    elif args.print_table:print(result_table())
    else:
        result_table()
        figures()
