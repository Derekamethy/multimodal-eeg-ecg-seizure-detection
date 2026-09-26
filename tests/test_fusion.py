import copy
import json
import numpy as np
import pandas as pd
import pytest

from multimodal_seizure import baseline, ecg_baseline, fusion
from multimodal_seizure.metrics import evaluate_record


def sample(name="a.edf"):
    t=np.arange(10,100,5)
    y=((t>=30)&(t<45)).astype(int)
    frame=pd.DataFrame(dict(patient="P",file_name=name,anchor_s=t,recording_duration_s=100,
        eeg_available=True,ecg_available=t>=60,annotation_uncertain=False,
        evaluation_label=y,label=y,training_eligible=True,score_e=y*.8,score_c=np.nan,score_early=y*.8))
    for c in ecg_baseline.COLUMNS: frame[c]=y.astype(float)
    frame["eeg__one"]=y.astype(float)
    frame["eeg__two"]=t.astype(float)
    event=pd.DataFrame([dict(event_id=name,onset_relative_s=30,offset_relative_s_canonical=45,uncertain_intervals_json="[]")])
    return frame,event


def assert_eval(a,b):
    pd.testing.assert_frame_equal(pd.DataFrame([a[0]]),pd.DataFrame([b[0]]))
    for i in (1,2): pd.testing.assert_frame_equal(pd.DataFrame(a[i]),pd.DataFrame(b[i]))
    assert a[3]==b[3]


def test_alignment_uses_keys_and_rejects_duplicates_missing_stride_and_metadata():
    f,_=sample()
    e=f.drop(columns=["ecg_available"]+ecg_baseline.COLUMNS)
    c=f.drop(columns=["eeg_available","eeg__one","eeg__two"])
    aligned=fusion.align(e,c.sample(frac=1,random_state=4).reset_index(drop=True))
    np.testing.assert_array_equal(aligned.ecg_available,f.ecg_available)
    for bad in (pd.concat([c,c.iloc[:1]]),c.iloc[1:],c.assign(recording_duration_s=101),c.assign(patient="Q")):
        with pytest.raises(ValueError): fusion.align(e,bad)
    with pytest.raises(ValueError,match="stride"): fusion.align(e,c.drop(index=4))


def test_zero_alpha_and_full_outage_reproduce_complete_eeg_evaluation():
    f,e=sample()
    f["score_c"]=.9
    expected=evaluate_record(f.assign(score=f.score_e),e,100,.7)
    for alpha,outage in ((0,False),(.1,True),(.25,True),(.5,True)):
        result=fusion.late_frame(f,alpha,.2,.7,outage)
        np.testing.assert_array_equal(result.score,f.score_e)
        assert_eval(fusion.evaluate(result,e),expected)


def test_q_zero_ignores_bad_ecg_contents_startup_and_uses_eeg_threshold():
    f,e=sample()
    f["ecg_available"]=False
    for contents in (np.nan,1.,-999.,np.inf):
        result=fusion.late_frame(f.assign(score_c=contents),.5,.05,.9)
        np.testing.assert_array_equal(result.score,f.score_e)
        assert (result.effective_threshold==.9).all()
        assert fusion.evaluate(result,e)[0]["total_declarations"]==0
    result=fusion.late_frame(f,.5,.05,.7)
    assert result.anchor_s.iloc[0]==10 and fusion.evaluate(result,e)[0]["detected"]==1


def test_gate_transition_preserves_frozen_state_and_no_ecg_substitution():
    f,e=sample()
    f["score_e"]=.7;f["score_c"]=.1;f["ecg_available"]=True
    f.loc[f.anchor_s<20,"ecg_available"]=False
    f.loc[f.anchor_s==40,"eeg_available"]=False
    result=fusion.late_frame(f,.5,.6,.6)
    metric,alarms,_,_=fusion.evaluate(result,e)
    assert alarms[0]["declaration_s"]==10 and alarms[0]["end_s"]==25
    assert metric["evaluable_s"]==85


def test_frozen_grids_models_and_predictors():
    assert fusion.ALPHAS==(0,.1,.25,.5)
    assert baseline.THRESHOLDS==tuple(round(i/20,2) for i in range(1,20))+(1.01,)
    assert ecg_baseline.MODEL_CONFIG["C1_RF"]==baseline.MODEL_CONFIG["E1_RF"]
    model=baseline.make_model("E1_RF")["model"]
    assert (model.n_estimators,model.max_depth,model.min_samples_leaf,model.max_features,model.random_state)==(300,6,5,"sqrt",20260922)
    f,_=sample();f["lead1_good"]=True
    result=fusion.early_frame(f)
    columns=[c for c in result if c.startswith("eeg__cardiac__")]
    assert len(columns)==12 and result.loc[~f.ecg_available,columns].isna().all().all()
    assert not any("quality" in c or "good" in c for c in columns)


@pytest.mark.parametrize("detected,far,delay,allowed",[(1,.5,0,False),(3,1.1,0,False),(2,1,0,False),(2,.5,6,False),(2,.5,5,True),(3,1,0,True)])
def test_protection_rejects_losses_and_excess_delay(detected,far,delay,allowed):
    base=dict(detected=2,far_per_h=1,evaluable_s=100)
    candidate=dict(base,detected=detected,far_per_h=far)
    assert fusion.protection(candidate,base,{("a","1"):10+delay},{("a","1"):10})[0]==allowed


def test_ties_prefer_smaller_alpha_before_higher_threshold():
    row=dict(sensitivity=.5,far_per_h=.5,delay_median_s=10,alpha=.1,threshold=.5)
    assert min([dict(row,alpha=.25,threshold=.9),row],key=lambda r:fusion.ranking(r,True)) is row


def test_inner_selection_zero_reproduction():
    f,e=sample()
    choice,table,early=fusion.select({"a.edf":f},{"a.edf":e})
    assert len(table)==80 and len(early)==20
    assert choice["alpha"]==0 and choice["threshold"]==choice["eeg_threshold"]
    assert choice["reason"]=="no_eligible_nonzero_candidate"


def test_outer_mutation_cannot_affect_inner_training_selection_and_imputation():
    names=["a.edf","b.edf","multi.4.5.6.edf"]
    joined={n:sample(n)[0] for n in names}
    # Ensure ECG has both classes without changing the separation policy.
    for f in joined.values(): f["ecg_available"]=True
    events={n:sample(n)[1] for n in names}
    held=names[-1]
    first=fusion.inner_scores(joined,joined,joined,held,{})
    model,cols=baseline.fit_training({n:fusion.early_frame(f) for n,f in joined.items()},names[:2],[held],"E1_RF")
    expected=pd.concat([fusion.early_frame(joined[n]) for n in names[:2]])[cols].median().to_numpy()
    np.testing.assert_allclose(model["imputer"].statistics_,expected)
    changed=copy.deepcopy(joined)
    changed[held]["label"]=1-changed[held].label
    changed[held][ecg_baseline.COLUMNS]=1e9
    changed[held]["eeg__one"]=-1e9
    changed[held]["ecg_available"]=False
    events[held]["onset_relative_s"]=80
    second=fusion.inner_scores(changed,changed,changed,held,{})
    for n in first:
        pd.testing.assert_frame_equal(first[n],second[n],check_exact=False,atol=1e-14,rtol=0)
        assert held not in json.loads(first[n].train_files_json.iloc[0])
        assert n not in json.loads(first[n].train_files_json.iloc[0])
    refs={n:events[n] for n in names[:2]}
    pd.testing.assert_frame_equal(fusion.select(first,refs)[1],fusion.select(second,refs)[1])
    with pytest.raises(ValueError,match="leakage"):
        fusion.fit_scores(joined,joined,joined,names,held,[],{})
