"""Read-only Phase 3 replay; no feature extraction, fitting or parameter changes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.multimodal_seizure.data import PROJECT_ROOT, read_event_table
from src.multimodal_seizure.provenance import metadata_identity, require_current, sha256
from src.multimodal_seizure.ecg_baseline import evaluate_ecg, ecg_window_metrics
from src.multimodal_seizure.cardiac_analysis import coverage, responses, comparison
from src.multimodal_seizure.metrics import summarize


def equal_table(rows, path, keys):
    actual=pd.DataFrame(rows).sort_values(keys,kind="stable").reset_index(drop=True).replace("",np.nan)
    expected=pd.read_csv(path,float_precision="round_trip").sort_values(keys,kind="stable").reset_index(drop=True)
    if len(actual)==0 and len(expected)==0: return
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)


def verify(out):
    require_current(out/"manifest.json")
    manifest=json.loads((out/"manifest.json").read_text())
    config=json.loads((out/"config.json").read_text())
    assert config["metadata_identity"]==metadata_identity()
    eeg=PROJECT_ROOT/"results/phase2_eeg/3ba65f49586b"
    assert sha256(eeg/"manifest.json")==config["frozen_eeg_manifest_sha256"]
    assert sha256(eeg/"config.json")==config["frozen_eeg_config_sha256"]
    frozen=json.loads((eeg/"manifest.json").read_text())
    assert frozen["experiment_identity"]==config["frozen_eeg_identity"]
    for item in frozen["artifacts"]:
        assert sha256(PROJECT_ROOT/item["path"])==item["sha256"]
    for item in manifest["artifacts"]:
        path=PROJECT_ROOT/item["path"]
        assert sha256(path)==item["sha256"]
        require_current(path)
    events=read_event_table()
    assert len(events)==27 and "PN00-3.edf" not in set(events.canonical_file_name)
    coverage_rows,response_rows=[],[]
    for path in sorted((out/"features").glob("*.csv")):
        frame=pd.read_csv(path,float_precision="round_trip")
        name=frame.file_name.iloc[0];patient=frame.patient.iloc[0]
        assert frame.anchor_s.iloc[0]==10
        np.testing.assert_array_equal(np.diff(frame.anchor_s),5)
        assert not frame.loc[frame.anchor_s<60,"ecg_available"].any()
        assert not frame.loc[frame.annotation_uncertain,"training_eligible"].any()
        refs=events.loc[events.canonical_file_name==name]
        coverage_rows.append(dict(patient=patient,file_name=name,**coverage(frame,refs)))
        response_rows.extend(responses(frame,refs))
    equal_table(coverage_rows,out/"coverage_recordings.csv",["patient","file_name"])
    equal_table(response_rows,out/"cardiac_responses.csv",["event_id"])
    assert len(response_rows)==27
    folds=pd.read_csv(out/"fold_metrics.csv",float_precision="round_trip")
    thresholds=pd.read_csv(out/"threshold_selection.csv",float_precision="round_trip")
    metrics_all,alarms_all,matches_all,exposure_all=[],[],[],[]
    for path in sorted((out/"scores").glob("*.csv")):
        combined=pd.read_csv(path,float_precision="round_trip")
        for (patient,kind,name),stream in combined.groupby(["patient","model","file_name"],sort=True):
            assert name not in json.loads(stream.train_files_json.iloc[0])
            choices=thresholds.loc[(thresholds.patient==patient)&(thresholds.model==kind)&(thresholds.test_file==name)]
            assert len(choices)==20
            if stream.calibration_status.iloc[0]=="nested":
                assert choices.selected.sum()==1
                best=choices.loc[choices.feasible].sort_values(["sensitivity","far_per_h","delay_median_s","threshold"],ascending=[False,True,True,False],na_position="last").iloc[0]
                assert float(stream.threshold.iloc[0])==best.threshold
                for inner in json.loads(best.inner_support_json):
                    assert name not in inner["train_files"] and name!=inner["validation_file"]
                    assert inner["validation_file"] not in inner["train_files"]
            else:
                assert not stream.model_available.any() and not choices.selected.any()
            metric,alarms,matches,exposure=evaluate_ecg(stream,events.loc[events.canonical_file_name==name],float(stream.threshold.iloc[0]))
            metric.update(ecg_window_metrics(stream))
            saved=folds.loc[(folds.patient==patient)&(folds.model==kind)&(folds.test_file==name)].iloc[0]
            for key,value in metric.items():
                np.testing.assert_allclose(value,saved[key],rtol=1e-10,atol=1e-10,equal_nan=True,err_msg=f"{name}:{key}")
            tags=dict(patient=patient,model=kind,test_file=name)
            metrics_all.append(dict(**tags,**metric))
            alarms_all.extend(dict(**tags,**a) for a in alarms)
            matches_all.extend(dict(**tags,**m) for m in matches)
            exposure_all.extend(dict(**tags,start_s=a,end_s=b) for a,b in exposure)
    for filename,rows in (("alarms.csv",alarms_all),("event_matches.csv",matches_all),("evaluation_exposure.csv",exposure_all)):
        equal_table(rows,out/filename,["patient","model","test_file"])
    for filename,keys in (("patient_metrics.csv",["patient","model"]),("pooled_metrics.csv",["model"])):
        saved=pd.read_csv(out/filename,float_precision="round_trip")
        for _,row in saved.iterrows():
            relevant=lambda item:all(item[k]==row[k] for k in keys)
            metric=summarize([m for m in metrics_all if relevant(m)],[m for m in matches_all if relevant(m)])
            for key,value in metric.items():
                np.testing.assert_allclose(value,row[key],rtol=1e-10,atol=1e-10,equal_nan=True)
    joined=comparison(pd.DataFrame(matches_all),pd.DataFrame(response_rows),None,eeg)
    equal_table(joined.to_dict("records"),out/"event_complementarity.csv",["event_id"])
    assert len(metrics_all)==44
    print("PHASE3_REPLAY_PASS 44/44 model-folds; 27/27 response rows; coverage, thresholds, alarm/event tables, metrics, complementarity and provenance")
    print("PHASE2_ARTIFACTS_UNCHANGED",len(frozen["artifacts"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("output",type=Path)
    verify(parser.parse_args().output.resolve())
