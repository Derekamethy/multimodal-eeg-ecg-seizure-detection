"""Phase 3 ECG only; accepted EEG code, scores and evaluator remain immutable."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import pyedflib

from src.multimodal_seizure.data import PROJECT_ROOT, RAW_ROOT, read_event_table, get_official_channels
from src.multimodal_seizure.windows import build_window_index
from src.multimodal_seizure.cardiac import RULES, FEATURES, analyze_pair, relative_features
from src.multimodal_seizure.cardiac_analysis import coverage, responses, patient_responses, comparison
from src.multimodal_seizure.ecg_baseline import MODEL_CONFIG, COLUMNS, THRESHOLDS, SEED, fit_outer, predict, evaluate_ecg, ecg_window_metrics
from src.multimodal_seizure.metrics import summarize
from src.multimodal_seizure.provenance import artifact_spec, stamp_artifact, require_current, sha256

EEG = PROJECT_ROOT / "results/phase2_eeg/3ba65f49586b"
CODE = [Path(__file__).resolve()] + [PROJECT_ROOT / f"src/multimodal_seizure/{name}.py" for name in
    ("cardiac", "cardiac_analysis", "ecg_baseline", "alarms", "metrics", "baseline", "windows", "data", "provenance")]
CONFIG = dict(phase=3, seed=SEED, rules=RULES, features=FEATURES, models=MODEL_CONFIG, threshold_grid=THRESHOLDS,
    stride_s=5, current_window="[t-60,t)", reference_window="[t-360,t-60), five disjoint past 60-s blocks",
    baseline_ready="at least two good historical blocks (120 s); relative NaNs otherwise; absolute model remains available",
    quality_policy="engineering validity heuristics, not clinical normality; fixed without seizure-performance tuning",
    selection="one good lead or both agreeing; lexicographic valid-RR fraction, coverage, contrast, then lead 1",
    timing="forward filter reset per past window; decisions causal, window-end beat estimates may revise; no clinical online accuracy claim",
    physiology="baseline qualified windows fully inside [-360,-60); >=120s support; pre-event latest available decision within 5s; ictal anchors [onset,offset); postictal anchors [offset,offset+60)",
    physiology_interpretation="HR summaries of trailing 60-s windows, not isolated instantaneous ictal HR; no responder cutoff or latency threshold",
    false_alarm_diagnostics="coincidence within +/-5s, same EDF; stable-HR proxy requires good ECG, ready baseline and abs(HR-baseline)<=10 bpm; not clinical normality",
    legacy_audit="old ecg.py uses window sosfiltfilt, removes invalid RR then differences across gaps, compresses RR time, and pools two leads; legacy QC is stale and unused",
    frozen_eeg_identity="3ba65f49586b8c13e4e497ba0aa2e160bfed9d1cdd64434bb72a131f72a47b21",
    calibration_failure="preserve unsupported fold with NaN scores and explicit status; no invented window split or external threshold")


def save(path, frame, spec):
    if path.exists():
        require_current(path, expected_identity=spec["cache_identity"])
    frame.to_csv(path, index=False, float_format="%.17g")
    stamp_artifact(path, spec)


def experiment():
    frozen = json.loads((EEG / "config.json").read_text())
    assert frozen["experiment_identity"] == CONFIG["frozen_eeg_identity"]
    config = dict(CONFIG, python=platform.python_version(), packages={n:importlib.metadata.version(n)
        for n in ("numpy", "scipy", "pandas", "scikit-learn", "pyedflib")},
        frozen_eeg_config_sha256=sha256(EEG / "config.json"), frozen_eeg_manifest_sha256=sha256(EEG / "manifest.json"))
    spec = artifact_spec("phase3_config", CODE, config)
    out = PROJECT_ROOT / "results/phase3_ecg" / spec["cache_identity"][:12]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    if path.exists():
        require_current(path, expected_identity=spec["cache_identity"])
    else:
        path.write_text(json.dumps(dict(config, experiment_identity=spec["cache_identity"], metadata_identity=spec["metadata_identity"], code_sha256=spec["dependencies"]), indent=2) + "\n", encoding="utf-8")
        stamp_artifact(path, spec)
    return out, path


def extract_record(patient, filename, out, config):
    out, config = Path(out), Path(config)
    spec = artifact_spec("phase3_ecg_features", CODE + [config], dict(patient=patient, file_name=filename))
    if out.exists():
        require_current(out, expected_identity=spec["cache_identity"])
        return patient, filename, "verified_cache"
    timeline = build_window_index(patient, filename).drop(columns=["eeg_available"])
    mapping = get_official_channels(patient, "ECG", filename)
    indices = mapping.edf_index_0based.to_numpy(int)
    reader = pyedflib.EdfReader(str(RAW_ROOT / patient / filename))
    history, records = {}, []
    try:
        if len(indices) != 2 or any(reader.getSampleFrequency(int(i)) != 512 for i in indices):
            raise ValueError("Expected the two official 512-Hz ECG leads")
        if any(reader.getPhysicalDimension(int(i)).strip().lower() not in {"uv", "µv"} for i in indices):
            raise ValueError("Unexpected ECG units")
        rails = [(reader.getPhysicalMinimum(int(i)), reader.getPhysicalMaximum(int(i))) for i in indices]
        last_chunk, signals = None, None
        for t in timeline.anchor_s.astype(int):
            row = dict(ecg_available=False, selected_lead=0, selection_reason="startup", beat_agreement=np.nan,
                baseline_ready=False, baseline_good_s=0, baseline_hr=np.nan, baseline_rr=np.nan,
                ecg_start_s=t-60 if t>=60 else np.nan, ecg_end_s=t, sampling_rate_hz=512,
                reference_start_s=max(0,t-360), reference_end_s=max(0,t-60))
            row.update({"ecg__"+name:np.nan for name in FEATURES})
            for lead in (1,2):
                row.update({f"lead{lead}_good":False, f"lead{lead}_n_peaks":0, f"lead{lead}_reason":"startup",
                            f"lead{lead}_peaks_samples_json":"[]", f"lead{lead}_edf_index":int(indices[lead-1])})
            if t >= 60:
                chunk = int((t-60)//300)*300
                if chunk != last_chunk:
                    end = min(chunk+360, reader.file_duration)
                    signals = np.vstack([reader.readSignal(int(i), start=chunk*512, n=int((end-chunk)*512)) for i in indices])
                    last_chunk = chunk
                start = (t-60-chunk)*512
                pair = analyze_pair(signals[:,start:start+60*512], 512, t-60, rails)
                relative, baseline = relative_features(pair, history, t)
                row.update(baseline, ecg_available=pair["selected"] is not None, selected_lead=pair["selected"]+1 if pair["selected"] is not None else 0,
                           selection_reason=pair["reason"], beat_agreement=pair["agreement"])
                row.update({"ecg__"+k:v for k,v in dict(pair["features"], **relative).items()})
                for lead, value in enumerate(pair["leads"],1):
                    row.update({f"lead{lead}_{k}":v for k,v in value["evidence"].items()})
                    row[f"lead{lead}_peaks_samples_json"] = json.dumps(value["peaks"].tolist(), separators=(",",":"))
                history[t] = pair
                history = {u:p for u,p in history.items() if u >= t-360}
            records.append(row)
    finally:
        reader.close()
    derived = pd.DataFrame(records)
    timeline = timeline.drop(columns=[c for c in derived if c in timeline])
    frame = pd.concat([timeline,derived],axis=1)
    frame["training_eligible"] &= frame.ecg_available
    save(out,frame,spec)
    return patient,filename,len(frame),int(frame.ecg_available.sum())


def load_frames(out, events):
    frames = {}
    for patient, name in events[["patient","canonical_file_name"]].drop_duplicates().itertuples(index=False):
        path = out / "features" / f"{patient}__{name}.csv"
        require_current(path)
        frames[name] = pd.read_csv(path,float_precision="round_trip")
    return frames


def describe(out, config, events, frames):
    rows, response_rows = [], []
    for name, frame in frames.items():
        refs = events.loc[events.canonical_file_name==name]
        rows.append(dict(patient=frame.patient.iloc[0],file_name=name,**coverage(frame,refs)))
        response_rows.extend(responses(frame,refs))
    cov = pd.DataFrame(rows)
    totals = cov.groupby("patient")[["raw_s","startup_s","uncertain_s","poor_quality_s","acquisition_missing_s","baseline_not_ready_good_s","evaluable_s"]].sum().reset_index()
    totals["coverage_pct"] = totals.evaluable_s/totals.raw_s*100
    response_table = pd.DataFrame(response_rows)
    summary = patient_responses(response_table).merge(totals,on="patient")
    sanity = []
    for patient in sorted(events.patient.unique()):
        name = sorted(events.loc[events.patient==patient,"canonical_file_name"].unique())[0]
        frame=frames[name]
        sanity.append(frame.loc[frame.anchor_s==120].drop(columns=COLUMNS).iloc[0].to_dict())
    spec=artifact_spec("phase3_physiology",CODE+[config]+sorted((out/"features").glob("*.csv")))
    for name, frame in {"coverage_recordings.csv":cov,"coverage_patients.csv":totals,"cardiac_responses.csv":response_table,
                        "cardiac_patient_summary.csv":summary,"beat_sanity.csv":pd.DataFrame(sanity)}.items():
        save(out/name,frame,spec)
    print("PHYSIOLOGY_SAVED",len(response_table),"events; no model outcomes used",flush=True)
    return response_table


def models(out,config,events,frames,response_table):
    (out/"scores").mkdir(exist_ok=True)
    folds, tables, alarms_all, matches_all, exposures, patients, failed = [],[],[],[],[],[],[]
    score_paths=[]
    for patient in sorted(events.patient.unique()):
        refs={n:g for n,g in events.loc[events.patient==patient].groupby("canonical_file_name")}
        data={n:frames[n] for n in refs}
        cache={}
        for kind in MODEL_CONFIG:
            streams=[]
            for held in sorted(data):
                print("FIT",patient,kind,held,flush=True)
                status="nested"
                try:
                    model,threshold,table=fit_outer(data,refs,held,kind,cache)
                except ValueError as error:
                    if "Calibration-limited" not in str(error): raise
                    model,threshold,status=None,1.01,"calibration_limited"
                    failed.append(dict(patient=patient,model=kind,test_file=held,reason=str(error)))
                    table=pd.DataFrame([dict(threshold=x,selected=False,feasible=False,reason=str(error)) for x in THRESHOLDS])
                stream=predict(model,data[held])
                stream["threshold"],stream["model"],stream["calibration_status"]=threshold,kind,status
                stream["train_files_json"]=json.dumps(sorted(n for n in data if n!=held))
                stream["seed"]=SEED
                # Features/lead evidence remain in the score artifact for direct reconstruction.
                streams.append(stream)
                tables.append(table.assign(patient=patient,model=kind,test_file=held,calibration_status=status))
            combined=pd.concat(streams,ignore_index=True)
            path=out/"scores"/f"{patient}_{kind}_continuous_oof.csv"
            spec=artifact_spec("phase3_scores",CODE+[config]+[out/"features"/f"{patient}__{n}.csv" for n in data])
            save(path,combined,spec) # Saved before outer alarm evaluation.
            score_paths.append(path)
            pm,pe=[],[]
            for name,stream in combined.groupby("file_name",sort=True):
                metric,alarms,matches,exposure=evaluate_ecg(stream,refs[name],float(stream.threshold.iloc[0]))
                tags=dict(patient=patient,model=kind,test_file=name)
                metric.update(ecg_window_metrics(stream),**tags,threshold=float(stream.threshold.iloc[0]),calibration_status=stream.calibration_status.iloc[0])
                folds.append(metric);pm.append(metric);pe.extend(matches)
                alarms_all.extend(dict(**tags,**a) for a in alarms)
                matches_all.extend(dict(**tags,**e) for e in matches)
                exposures.extend(dict(**tags,start_s=a,end_s=b) for a,b in exposure)
            metric=summarize(pm,pe);metric.update(ecg_window_metrics(combined),patient=patient,model=kind)
            patients.append(metric)
            print("COMPLETE",patient,kind,flush=True)
    pooled=[]
    for kind in MODEL_CONFIG:
        row=summarize([r for r in folds if r["model"]==kind],[r for r in matches_all if r["model"]==kind])
        scores=pd.concat([pd.read_csv(p,float_precision="round_trip") for p in score_paths if kind in p.name])
        row.update(ecg_window_metrics(scores),model=kind);pooled.append(row)
    alarm_table=pd.DataFrame(alarms_all,columns=["patient","model","test_file","declaration_s","end_s","termination_reason","match_status","event_id"])
    match_table=pd.DataFrame(matches_all)
    spec=artifact_spec("phase3_evaluation",CODE+[config]+score_paths)
    for name,frame in {"threshold_selection.csv":pd.concat(tables,ignore_index=True),"fold_metrics.csv":pd.DataFrame(folds),
        "patient_metrics.csv":pd.DataFrame(patients),"pooled_metrics.csv":pd.DataFrame(pooled),"alarms.csv":alarm_table,
        "event_matches.csv":match_table,"evaluation_exposure.csv":pd.DataFrame(exposures,columns=["patient","model","test_file","start_s","end_s"])}.items():
        save(out/name,frame,spec)
    compare(out,config,frames,response_table,match_table,alarm_table,score_paths)
    artifacts=sorted(p for p in out.rglob("*") if p.is_file() and not p.name.endswith(".provenance.json") and p.name not in {"manifest.json","failure.json"})
    manifest=dict(status="complete" if not failed else "completed_with_calibration_limits",model_folds=len(folds),calibration_limited_folds=failed,
        experiment_identity=json.loads(config.read_text())["experiment_identity"],frozen_eeg_identity=CONFIG["frozen_eeg_identity"],
        artifacts=[dict(path=p.relative_to(PROJECT_ROOT).as_posix(),sha256=sha256(p),sidecar=p.with_suffix(p.suffix+".provenance.json").relative_to(PROJECT_ROOT).as_posix()) for p in artifacts])
    path=out/"manifest.json";path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    stamp_artifact(path,artifact_spec("phase3_manifest",CODE+[config]))
    print("PHASE3_COMPLETE",out,flush=True)


def compare(out,config,frames,response_table,match_table,alarm_table,score_paths):
    comparison_table=comparison(match_table,response_table,None,EEG)
    eeg_alarm=pd.read_csv(EEG/"alarms.csv")
    eeg_false=eeg_alarm.loc[(eeg_alarm.model=="E1_RF") & (eeg_alarm.match_status!="matched")]
    ecg_false=alarm_table.loc[(alarm_table.model=="C1_RF") & (alarm_table.match_status!="matched")]
    diagnostic=[]
    for a in eeg_false.itertuples():
        candidates=frames[a.test_file].loc[frames[a.test_file].anchor_s<=a.declaration_s]
        state="unavailable";delta=np.nan
        if len(candidates):
            last=candidates.iloc[-1]
            if a.declaration_s-last.anchor_s<=5 and last.ecg_available:
                state="baseline_not_ready"
                if last.baseline_ready:
                    delta=last.ecg__relative_hr_delta
                    state="within_10bpm_recent_baseline" if abs(delta)<=10 else "larger_hr_deviation"
        coincidence=bool(((ecg_false.test_file==a.test_file) & (abs(ecg_false.declaration_s-a.declaration_s)<=5)).any())
        diagnostic.append(dict(patient=a.patient,file_name=a.test_file,source="EEG",declaration_s=a.declaration_s,
            ecg_state=state,hr_delta=delta,other_false_within_5s=coincidence))
    for a in ecg_false.itertuples():
        coincidence=bool(((eeg_false.test_file==a.test_file) & (abs(eeg_false.declaration_s-a.declaration_s)<=5)).any())
        diagnostic.append(dict(patient=a.patient,file_name=a.test_file,source="ECG",declaration_s=a.declaration_s,
            ecg_state="not_classified",hr_delta=np.nan,other_false_within_5s=coincidence))
    shift=[]
    for path in score_paths:
        scored=pd.read_csv(path,float_precision="round_trip")
        for name,g in scored.groupby("file_name"):
            bg=g.loc[g.ecg_available & (g.evaluation_label==0)]
            shift.append(dict(patient=g.patient.iloc[0],model=g.model.iloc[0],file_name=name,
                ecg_good_fraction=float(g.loc[g.anchor_s>=60,"ecg_available"].mean()),
                background_hr_median=bg.ecg__median_hr.median(),background_hr_q25=bg.ecg__median_hr.quantile(.25),
                background_hr_q75=bg.ecg__median_hr.quantile(.75),background_rr_median=bg.ecg__median_rr.median(),
                reference_hr_median=bg.baseline_hr.median(),background_score_median=bg.score.median(),
                background_score_p95=bg.score.quantile(.95),background_score_p99=bg.score.quantile(.99)))
    # Explicit immutable comparison inputs; no score combination occurs.
    spec=artifact_spec("phase3_descriptive_comparison",CODE+[config,EEG/"event_matches.csv",EEG/"alarms.csv"]+score_paths)
    for name,frame in {"event_complementarity.csv":comparison_table,"false_alarm_comparison.csv":pd.DataFrame(diagnostic),"recording_shift.csv":pd.DataFrame(shift)}.items():
        save(out/name,frame,spec)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args();out,config=experiment();events=read_event_table()
    print("EXPERIMENT",out,flush=True)
    (out/"features").mkdir(exist_ok=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(extract_record,p,n,str(out/"features"/f"{p}__{n}.csv"),str(config))
                  for p,n in events[["patient","canonical_file_name"]].drop_duplicates().itertuples(index=False)]
            for future in as_completed(jobs): print("FEATURES",*future.result(),flush=True)
        frames=load_frames(out,events)
        response_table=describe(out,config,events,frames)
        models(out,config,events,frames,response_table)
    except Exception as error:
        path=out/"failure.json";path.write_text(json.dumps(dict(error=repr(error)),indent=2),encoding="utf-8")
        stamp_artifact(path,artifact_spec("phase3_failure",CODE+[config]));raise


if __name__=="__main__": main()
