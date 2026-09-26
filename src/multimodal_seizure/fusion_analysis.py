"""Descriptive outer comparisons; never used to select fusion parameters."""
import numpy as np
import pandas as pd
from .alarms import interval_seconds, subtract_intervals
from .fusion import evaluate
from .metrics import summarize, window_metrics


def intersection(a,b):
    return [[max(x,u),min(y,v)] for x,y in a for u,v in b if max(x,u)<min(y,v)]


def build_tables(streams,events):
    folds,alarms,matches,exposures=[],[],[],[]
    results={}
    for (patient,model,name),frame in streams.groupby(["patient","model","file_name"],sort=True):
        metric,aa,mm,ex=evaluate(frame,events.loc[events.canonical_file_name==name])
        results[(model,name)]=(metric,aa,mm,ex)
        tags=dict(patient=patient,model=model,test_file=name)
        folds.append(dict(**metric,**window_metrics(frame),**tags,alpha=frame.alpha.iloc[0],threshold=frame.threshold.iloc[0]))
        alarms.extend(dict(**tags,**a) for a in aa)
        matches.extend(dict(**tags,**m) for m in mm)
        exposures.extend(dict(**tags,start_s=a,end_s=b) for a,b in ex)
    fold=pd.DataFrame(folds);match=pd.DataFrame(matches)
    alarm=pd.DataFrame(alarms,columns=["patient","model","test_file","declaration_s","end_s","termination_reason","match_status","event_id"])
    patients,pooled=[],[]
    for keys,target in ((["patient","model"],patients),(["model"],pooled)):
        for values,group in fold.groupby(keys):
            values=values if isinstance(values,tuple) else (values,)
            tags=dict(zip(keys,values)); mm=match;ss=streams
            for key,value in tags.items(): mm=mm.loc[mm[key]==value];ss=ss.loc[ss[key]==value]
            target.append(dict(**summarize(group.to_dict("records"),mm.to_dict("records")),**window_metrics(ss),**tags))
    tables=dict(fold_metrics=fold,patient_metrics=pd.DataFrame(patients),pooled_metrics=pd.DataFrame(pooled),
        alarms=alarm,event_matches=match,evaluation_exposure=pd.DataFrame(exposures))
    paired=[]
    fields=["detected","missed","false_declarations","far_per_h","delay_median_s","warning_fraction","longest_alarm_s","evaluable_s"]
    for level,source,keys in (("parent",fold,["patient","test_file"]),("patient",tables["patient_metrics"],["patient"]),("pooled",tables["pooled_metrics"],[])):
        base=source.loc[source.model=="F0"]
        fused=source.loc[source.model=="F2"]
        for _,r in fused.iterrows():
            b=base
            for k in keys:b=b.loc[b[k]==r[k]]
            b=b.iloc[0];row=dict(level=level,**{k:r[k] for k in keys})
            for field in fields:
                row.update({"eeg_"+field:b[field],"fusion_"+field:r[field],"delta_"+field:r[field]-b[field]})
            row["far_change_pct"]=(r.far_per_h/b.far_per_h-1)*100 if b.far_per_h>0 else np.nan
            row["coverage_delta_pct"]=(r.evaluable_s-b.evaluable_s)/b.raw_s*100
            paired.append(row)
    tables["paired_comparison"]=pd.DataFrame(paired)
    contributions,event_rows,fates,additions=[],[],[],[]
    for (patient,name),frame in streams.loc[streams.model=="F2"].groupby(["patient","file_name"]):
        ex=results[("F2",name)][3]
        duration=float(frame.recording_duration_s.iloc[0]);alpha=float(frame.alpha.iloc[0])
        qcells=[[t,min(t+5,duration)] for t in frame.loc[frame.q,"anchor_s"]]
        qex=intersection(ex,qcells);q1=interval_seconds(qex);total=interval_seconds(ex)
        contributions.append(dict(patient=patient,file_name=name,alpha=alpha,evaluable_s=total,q1_s=q1,q0_s=total-q1,
            q1_fraction=q1/total,ecg_active_s=q1 if alpha>0 else 0))
        eeg_events={m["event_id"]:m for m in results[("F0",name)][2]}
        for m in results[("F2",name)][2]:
            b=eeg_events[m["event_id"]]
            state="preserved" if b["detected"] and m["detected"] else "rescued" if m["detected"] else "lost" if b["detected"] else "still_missed"
            ictal=[[m["onset_s"],m["offset_s"]]]
            ictal_q=interval_seconds(intersection(ictal,qcells))
            available=frame.loc[(frame.anchor_s<=m["onset_s"])&(frame.anchor_s+5>m["onset_s"]),"q"]
            event_rows.append(dict(patient=patient,file_name=name,event_id=m["event_id"],eeg_detected=b["detected"],fusion_detected=m["detected"],
                eeg_delay_s=b["delay_s"],fusion_delay_s=m["delay_s"],delay_delta_s=m["delay_s"]-b["delay_s"],
                ecg_available_at_onset=bool(available.iloc[0]) if len(available) else False,
                ecg_ictal_available_fraction=ictal_q/(m["offset_s"]-m["onset_s"]),alpha=alpha,classification=state,
                changed_delay_only=state=="preserved" and b["delay_s"]!=m["delay_s"]))
        eeg_false=[a for a in results[("F0",name)][1] if a["match_status"]!="matched"]
        fusion_false=[a for a in results[("F2",name)][1] if a["match_status"]!="matched"]
        assignment={};used=set()
        # Prespecified one-to-one diagnostic: exact, then +/-5 s, then overlap.
        for stage in ("preserved","shifted","replaced"):
            pairs=[]
            for i,a in enumerate(eeg_false):
                if i in assignment:continue
                for j,b in enumerate(fusion_false):
                    if j in used:continue
                    delta=abs(a["declaration_s"]-b["declaration_s"])
                    ok=delta==0 if stage=="preserved" else delta<=5 if stage=="shifted" else max(a["declaration_s"],b["declaration_s"])<min(a["end_s"],b["end_s"])
                    if ok:pairs.append((delta,i,j))
            for _,i,j in sorted(pairs):
                if i not in assignment and j not in used:assignment[i]=(j,stage);used.add(j)
        for i,a in enumerate(eeg_false):
            state=frame.loc[frame.anchor_s==a["declaration_s"]].iloc[0]
            j,status=assignment.get(i,(None,"suppressed"))
            fates.append(dict(patient=patient,file_name=name,eeg_declaration_s=a["declaration_s"],fate=status,
                fusion_declaration_s=fusion_false[j]["declaration_s"] if j is not None else np.nan,
                ecg_available=state.q,alpha=alpha,ecg_score=state.score_c,fused_score=state.score,
                effective_threshold=state.effective_threshold,eeg_threshold=state.eeg_threshold,
                relative_hr_delta=state.ecg__relative_hr_delta if state.q else np.nan))
        for j,a in enumerate(fusion_false):
            if j not in used:additions.append(dict(patient=patient,file_name=name,**a))
    tables.update(event_comparison=pd.DataFrame(event_rows),false_alarm_fate=pd.DataFrame(fates),
        added_false_alarms=pd.DataFrame(additions,columns=["patient","file_name","declaration_s","end_s","termination_reason","match_status","event_id"]),
        ecg_contribution_recordings=pd.DataFrame(contributions))
    cov=tables["ecg_contribution_recordings"]
    patients=cov.groupby("patient")[["evaluable_s","q1_s","q0_s","ecg_active_s"]].sum().reset_index()
    patients["q1_fraction"]=patients.q1_s/patients.evaluable_s
    tables["ecg_contribution_patients"]=patients
    alpha=cov.groupby(["patient","alpha"]).agg(folds=("file_name","size"),evaluable_s=("evaluable_s","sum"),q1_s=("q1_s","sum"),ecg_active_s=("ecg_active_s","sum")).reset_index()
    tables["alpha_distribution"]=alpha
    return tables
