"""Preregistered Phase 4 only; fit exact frozen configurations on parent splits."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd

from src.multimodal_seizure import baseline, ecg_baseline, fusion
from src.multimodal_seizure.fusion_analysis import build_tables
from src.multimodal_seizure.data import PROJECT_ROOT, read_event_table
from src.multimodal_seizure.provenance import artifact_spec, stamp_artifact, sha256, require_current, metadata_identity
from src.multimodal_seizure.metrics import evaluate_record

EEG=PROJECT_ROOT/"results/phase2_eeg/3ba65f49586b"
ECG=PROJECT_ROOT/"results/phase3_ecg/6813d261b922"
CANONICAL="91b7c18e669fd92ef785c9692b15635c766db42d584bb1a24300dd34d64160b3"
IDENTITIES=("3ba65f49586b8c13e4e497ba0aa2e160bfed9d1cdd64434bb72a131f72a47b21","6813d261b922f82721d27bccdfa452135a56c4d47c3afe5f7bc72506a0c4737e")
CODE=[Path(__file__).resolve(),PROJECT_ROOT/"verify_phase4_fusion.py"]+[PROJECT_ROOT/f"src/multimodal_seizure/{n}.py" for n in
    ("fusion","fusion_analysis","baseline","ecg_baseline","cardiac","alarms","metrics","windows","data","provenance")]
CONFIG=dict(phase=4,primary="F2",comparators=["F0","F1","F3"],seed=baseline.SEED,
    canonical_identity=CANONICAL,frozen_eeg_identity=IDENTITIES[0],frozen_ecg_identity=IDENTITIES[1],
    alpha_grid=fusion.ALPHAS,threshold_grid=baseline.THRESHOLDS,
    formula="sF=(1-alpha*q)*sE+alpha*q*sC; branch assignment guarantees exact fallback, including NaN ECG",
    q="binary frozen Phase 3 ecg_available AND finite frozen-style ECG score; never inferred from outcomes",
    availability="EEG defines availability; missing ECG never removes EEG decisions",
    threshold_amendment="User explicitly chose actual alarm fallback: q=0 uses inner EEG threshold; q=1 uses selected fusion threshold; alpha=0 always uses EEG threshold. This applies to F1 and F2. No state reset on gate transitions.",
    evaluator_adapter="raw fused score retained; finite score>=effective threshold converted to 0/1, NaN preserved; frozen evaluator called at 0.5; identical high/low semantics",
    models=dict(EEG=baseline.MODEL_CONFIG["E1_RF"],ECG=ecg_baseline.MODEL_CONFIG["C1_RF"],early=baseline.MODEL_CONFIG["E1_RF"]),
    rf_max_features="sqrt",rf_bootstrap=True,early_features="46 frozen EEG +12 frozen ECG; no quality predictors; unavailable ECG set NaN; train-only median imputer",
    split="patient-specific outer whole-parent holdout; inner leave-one-outer-training-parent-out; no random windows; exact-model cache keyed by training parents",
    selection="inner EEG frozen FAR<=1/h rule first; F2 nonzero requires sensitivity>=EEG, FAR<=EEG, one strict improvement, common-event median paired delay increase<=5s and equal exposure; else alpha=0 and EEG threshold",
    common_delay="median of per-event (fusion delay - EEG delay) for inner events detected by both; if no common events, condition vacuous, count and NaN penalty reported",
    adaptive_ties="higher sensitivity, lower FAR, lower median detected delay, smaller alpha, higher threshold",
    fixed_and_early_selection="F1 alpha=.5 and F3 RF: standard inner FAR<=1/h, sensitivity/FAR/delay/higher-threshold ordering; F1 also uses user-approved q=0 EEG threshold",
    alarm="unchanged Phase 2: first high opens, two consecutive lows close, no smoothing/refractory/backdating; uncertainty/gaps reset; strict onset<=declaration<offset; duplicates unmatched",
    false_fate="descriptive one-to-one match: exact declaration first, then +/-5s nearest, then overlapping alarm intervals nearest; otherwise suppressed/new. Same parent only; no causal attribution",
    contributions="intersection of frozen EEG evaluable intervals with q=1 decision cells; selected alpha exposure includes q=0; ECG-active exposure requires alpha>0 and q=1",
    optional_controls="NC4 quality-disabled and temporal-shift deferred; invalid ECG has no model score; avoid changing frozen quality/features",
    statistics="5 patients,22 parents,27 events; paired descriptive comparisons; no independent-window significance or post-hoc primary-model selection")


def frozen_snapshot():
    assert metadata_identity()==CANONICAL
    snapshot={}
    for out,identity in zip((EEG,ECG),IDENTITIES):
        config=json.loads((out/"config.json").read_text());manifest=json.loads((out/"manifest.json").read_text())
        assert config["experiment_identity"]==manifest["experiment_identity"]==identity
        assert config["metadata_identity"]==CANONICAL
        for entry in manifest["artifacts"]:
            assert sha256(PROJECT_ROOT/entry["path"])==entry["sha256"]
        for p in sorted(out.rglob("*")):
            if p.is_file():snapshot[p.relative_to(PROJECT_ROOT).as_posix()]=sha256(p)
    assert json.loads((EEG/"config.json").read_text())["models"]["E1_RF"]==baseline.MODEL_CONFIG["E1_RF"]
    assert json.loads((ECG/"config.json").read_text())["models"]["C1_RF"]==ecg_baseline.MODEL_CONFIG["C1_RF"]
    return snapshot


def save(path,frame,config,dependencies=()):
    spec=artifact_spec("phase4_"+path.stem,CODE+[config]+list(dependencies))
    if path.exists():raise ValueError("Refusing to overwrite Phase 4 artifact: "+str(path))
    frame.to_csv(path,index=False,float_format="%.17g")
    stamp_artifact(path,spec)


def equal_evaluation(a,b):
    for key,value in a[0].items():np.testing.assert_allclose(value,b[0][key],rtol=0,atol=0,equal_nan=True)
    for i in (1,2):pd.testing.assert_frame_equal(pd.DataFrame(a[i]),pd.DataFrame(b[i]))
    assert a[3]==b[3]


def controls(stream,choice,refs,frozen_eeg,frozen_ecg):
    joined=stream.merge(frozen_eeg[fusion.KEYS+["score","threshold"]],on=fusion.KEYS,validate="one_to_one",suffixes=("","_frozen"))
    assert len(joined)==len(stream)==len(frozen_eeg)
    np.testing.assert_allclose(joined.score_e,joined.score,rtol=0,atol=1e-14,equal_nan=True)
    assert (joined.threshold==choice["eeg_threshold"]).all()
    c=stream.merge(frozen_ecg[fusion.KEYS+["score"]],on=fusion.KEYS,validate="one_to_one")
    np.testing.assert_allclose(c.score_c,c.score,rtol=0,atol=1e-14,equal_nan=True)
    reference=evaluate_record(frozen_eeg,refs,float(stream.recording_duration_s.iloc[0]),choice["eeg_threshold"])
    rows=[]
    for name,alpha,outage in (("NC1_alpha_zero",0,False),("NC2_entire_ECG_outage",choice["alpha"],True),("NC2_max_alpha_outage",.5,True)):
        frame=fusion.late_frame(stream,alpha,choice["threshold"],choice["eeg_threshold"],outage)
        np.testing.assert_array_equal(frame.score,stream.score_e)
        equal_evaluation(fusion.evaluate(frame,refs),reference)
        rows.append(dict(control=name,passed=True,decision_count=len(frame),detected=reference[0]["detected"],false_declarations=reference[0]["false_declarations"],evaluable_s=reference[0]["evaluable_s"]))
    f2=fusion.late_frame(stream,choice["alpha"],choice["threshold"],choice["eeg_threshold"])
    np.testing.assert_array_equal(f2.loc[~f2.q,"score"],f2.loc[~f2.q,"score_e"])
    assert (f2.loc[~f2.q,"effective_threshold"]==choice["eeg_threshold"]).all()
    return rows


def main():
    before=frozen_snapshot()
    resolved=dict(CONFIG,frozen_artifact_sha256=before,python=platform.python_version(),
        packages={n:importlib.metadata.version(n) for n in ("numpy","pandas","scipy","scikit-learn")})
    spec=artifact_spec("phase4_config",CODE,resolved)
    out=PROJECT_ROOT/"results/phase4_fusion"/spec["cache_identity"][:12]
    out.mkdir(parents=True,exist_ok=False)
    for sub in ("inner","scores"): (out/sub).mkdir()
    config=out/"config.json"
    config.write_text(json.dumps(dict(resolved,experiment_identity=spec["cache_identity"],metadata_identity=spec["metadata_identity"],code_sha256=spec["dependencies"]),indent=2)+"\n",encoding="utf-8")
    stamp_artifact(config,spec)
    print("PHASE4_EXPERIMENT",spec["cache_identity"],flush=True)
    events=read_event_table();all_choices=[];all_candidates=[];early_candidates=[];negative=[];outer_paths=[];inner_paths=[]
    for patient in sorted(events.patient.unique()):
        refs={n:g for n,g in events.loc[events.patient==patient].groupby("canonical_file_name")}
        eeg,ecg,joined={},{},{}
        dependencies=[]
        for name in refs:
            ep=EEG/"features"/f"{patient}__{name}.csv";cp=ECG/"features"/f"{patient}__{name}.csv"
            require_current(ep);require_current(cp);dependencies.extend([ep,cp])
            eeg[name]=pd.read_csv(ep,float_precision="round_trip")
            ecg[name]=pd.read_csv(cp,float_precision="round_trip")
            joined[name]=fusion.align(eeg[name],ecg[name])
            assert len([c for c in joined[name] if c.startswith("eeg__")])==46
        frozen_e=pd.read_csv(EEG/"scores"/f"{patient}_E1_RF_continuous_oof.csv",float_precision="round_trip")
        frozen_c=pd.read_csv(ECG/"scores"/f"{patient}_C1_RF_continuous_oof.csv",float_precision="round_trip")
        cache={}
        for held in sorted(refs):
            print("INNER_FIT",patient,held,flush=True)
            inner=fusion.inner_scores(eeg,ecg,joined,held,cache)
            path=out/"inner"/f"{patient}__{held}.csv"
            save(path,pd.concat(inner.values(),ignore_index=True).assign(outer_held=held),config,dependencies)
            inner_paths.append(path)
            choice,candidates,early=fusion.select(inner,{n:refs[n] for n in inner})
            tags=dict(patient=patient,test_file=held)
            support=[dict(validation_file=n,train_files=json.loads(f.train_files_json.iloc[0])) for n,f in inner.items()]
            choice.update(**tags,inner_support_json=json.dumps(support),train_files_json=json.dumps(sorted(inner)))
            all_choices.append(choice);all_candidates.append(candidates.assign(**tags));early_candidates.append(early.assign(**tags))
            print("SELECTED",patient,held,"alpha",choice["alpha"],"threshold",choice["threshold"],"EEG",choice["eeg_threshold"],flush=True)
            # Only now access the outer-held frame for predictions or outcomes.
            stream=fusion.fit_scores(eeg,ecg,joined,sorted(inner),held,[],cache)
            variants=fusion.outer_variants(stream,choice)
            combined=pd.concat([f.assign(model=k) for k,f in variants.items()],ignore_index=True)
            path=out/"scores"/f"{patient}__{held}.csv"
            save(path,combined,config,dependencies+[inner_paths[-1]])
            outer_paths.append(path)
            negative.extend(dict(**tags,**row) for row in controls(stream,choice,refs[held],frozen_e.loc[frozen_e.file_name==held],frozen_c.loc[frozen_c.file_name==held]))
        print("PATIENT_SCORES_SAVED",patient,flush=True)
    for name,frame in (("selected_parameters",pd.DataFrame(all_choices)),("alpha_threshold_candidates",pd.concat(all_candidates,ignore_index=True)),
        ("early_threshold_candidates",pd.concat(early_candidates,ignore_index=True)),("negative_controls",pd.DataFrame(negative))):
        save(out/(name+".csv"),frame,config,inner_paths if "candidates" in name or name=="selected_parameters" else outer_paths)
    streams=pd.concat([pd.read_csv(p,float_precision="round_trip") for p in outer_paths],ignore_index=True)
    tables=build_tables(streams,events)
    for name,frame in tables.items():save(out/(name+".csv"),frame,config,outer_paths)
    assert frozen_snapshot()==before
    files=sorted(p for p in out.rglob("*") if p.is_file() and not p.name.endswith(".provenance.json"))
    manifest=dict(status="complete",experiment_identity=spec["cache_identity"],outer_parents=22,model_folds=len(tables["fold_metrics"]),
        frozen_artifacts_unchanged=True,frozen_file_count=len(before),
        artifacts=[dict(path=p.relative_to(PROJECT_ROOT).as_posix(),sha256=sha256(p),sidecar=p.with_suffix(p.suffix+".provenance.json").relative_to(PROJECT_ROOT).as_posix()) for p in files])
    path=out/"manifest.json";path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    stamp_artifact(path,artifact_spec("phase4_manifest",CODE+[config]))
    print("PHASE4_COMPLETE",out,flush=True)


if __name__=="__main__":main()
