"""Read-only Phase 4 reconstruction from saved inner and outer scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.multimodal_seizure import fusion
from src.multimodal_seizure.data import PROJECT_ROOT,read_event_table
from src.multimodal_seizure.provenance import sha256,metadata_identity,cache_identity,sidecar
from src.multimodal_seizure.fusion_analysis import build_tables


def equal_table(actual,path,keys):
    expected=pd.read_csv(path,float_precision="round_trip",dtype={"match_status":str})
    actual=actual.replace("",np.nan)
    if len(actual)==len(expected)==0:return
    actual=actual.sort_values(keys,kind="stable").reset_index(drop=True)
    expected=expected.sort_values(keys,kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)


def verify(out):
    from run_phase4_fusion import frozen_snapshot,controls,EEG,ECG
    identity=metadata_identity()
    hashes,verified,visiting={},{},set()
    def digest(path):
        path=path.resolve()
        if path not in hashes:hashes[path]=sha256(path)
        return hashes[path]
    def provenance(path):
        path=path.resolve()
        if path in verified:return
        assert path not in visiting,"Cyclic provenance"
        visiting.add(path)
        info=json.loads(sidecar(path).read_text())
        assert info["metadata_identity"]==identity and info["artifact_sha256"]==digest(path)
        assert info["cache_identity"]==cache_identity(identity,info["kind"],info["config"],info["dependencies"])
        for name,expected in info["dependencies"].items():
            dependency=PROJECT_ROOT/name
            assert digest(dependency)==expected
            if sidecar(dependency).exists():provenance(dependency)
        visiting.remove(path);verified[path]=True
    provenance(out/"manifest.json")
    manifest=json.loads((out/"manifest.json").read_text());config=json.loads((out/"config.json").read_text())
    assert frozen_snapshot()==config["frozen_artifact_sha256"]
    for entry in manifest["artifacts"]:
        path=PROJECT_ROOT/entry["path"]
        assert digest(path)==entry["sha256"]
        provenance(path)
    events=read_event_table();choices=[];candidates=[];early=[];negatives=[];outer=[]
    selected=pd.read_csv(out/"selected_parameters.csv",float_precision="round_trip")
    for row in selected.to_dict("records"):
        patient,held=row["patient"],row["test_file"]
        frame=pd.read_csv(out/"inner"/f"{patient}__{held}.csv",float_precision="round_trip")
        assert (frame.outer_held==held).all() and not (frame.file_name==held).any()
        inner={n:g.drop(columns="outer_held") for n,g in frame.groupby("file_name",sort=True)}
        training=json.loads(row["train_files_json"])
        assert set(training)==set(inner) and held not in training
        for name,f in inner.items():
            assert not f.duplicated(fusion.KEYS).any()
            parents=json.loads(f.train_files_json.iloc[0])
            assert set(parents)==set(training)-{name} and held not in parents
        refs={n:events.loc[events.canonical_file_name==n] for n in inner}
        choice,table,et=fusion.select(inner,refs)
        tags=dict(patient=patient,test_file=held)
        for key,value in choice.items():
            if isinstance(value,str):assert value==row[key]
            else:np.testing.assert_allclose(value,row[key],rtol=1e-12,atol=1e-12,equal_nan=True)
        candidates.append(table.assign(**tags));early.append(et.assign(**tags));choices.append(choice)
        scored=pd.read_csv(out/"scores"/f"{patient}__{held}.csv",float_precision="round_trip")
        base=scored.loc[scored.model=="F0"].drop(columns=["model","q","score","effective_threshold","alpha","threshold","eeg_threshold"])
        variants=fusion.outer_variants(base,choice)
        for kind,reconstructed in variants.items():
            actual=scored.loc[scored.model==kind].drop(columns="model").reset_index(drop=True)
            pd.testing.assert_frame_equal(reconstructed.reset_index(drop=True),actual,check_dtype=False,check_exact=True,check_like=True)
            assert actual.eeg_available.equals(base.eeg_available.reset_index(drop=True))
        fe=pd.read_csv(EEG/"scores"/f"{patient}_E1_RF_continuous_oof.csv",float_precision="round_trip")
        fc=pd.read_csv(ECG/"scores"/f"{patient}_C1_RF_continuous_oof.csv",float_precision="round_trip")
        negatives.extend(dict(**tags,**v) for v in controls(base,choice,events.loc[events.canonical_file_name==held],fe.loc[fe.file_name==held],fc.loc[fc.file_name==held]))
        outer.append(scored)
        print("REPLAY_PARENT",patient,held,flush=True)
    equal_table(pd.concat(candidates,ignore_index=True),out/"alpha_threshold_candidates.csv",["patient","test_file","alpha","threshold"])
    equal_table(pd.concat(early,ignore_index=True),out/"early_threshold_candidates.csv",["patient","test_file","threshold"])
    equal_table(pd.DataFrame(negatives),out/"negative_controls.csv",["patient","test_file","control"])
    tables=build_tables(pd.concat(outer,ignore_index=True),events)
    for name,frame in tables.items():
        keys=[k for k in ("level","patient","model","test_file","file_name","event_id","alpha","declaration_s","eeg_declaration_s","start_s") if k in frame]
        equal_table(frame,out/(name+".csv"),keys or [frame.columns[0]])
    f=tables["fold_metrics"]
    original=pd.read_csv(EEG/"fold_metrics.csv",float_precision="round_trip")
    for row in f.loc[f.model=="F0"].to_dict("records"):
        expected=original.loc[(original.model=="E1_RF")&(original.test_file==row["test_file"])].iloc[0]
        for key,value in row.items():
            if key not in ("patient","model","test_file","alpha"):
                np.testing.assert_allclose(value,expected[key],rtol=1e-12,atol=1e-12,equal_nan=True)
    assert len(choices)==22 and len(f)==88 and len(tables["event_comparison"])==27
    assert len(tables["false_alarm_fate"])==69
    assert frozen_snapshot()==config["frozen_artifact_sha256"]
    print("PHASE4_REPLAY_PASS 88/88 model-folds; 22/22 inner selections; 27/27 event comparisons; 69/69 EEG false fates; 66/66 negative controls; frozen files unchanged",len(config["frozen_artifact_sha256"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("output",type=Path)
    verify(parser.parse_args().output.resolve())
