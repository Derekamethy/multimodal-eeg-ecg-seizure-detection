"""Training-side fusion selection with frozen models and alarm semantics."""
import json
import numpy as np
import pandas as pd

from . import baseline, ecg_baseline
from .metrics import evaluate_record, summarize

ALPHAS = (0., .10, .25, .50)
KEYS = ["patient", "file_name", "anchor_s"]


def align(eeg, ecg):
    """Keep the EEG timeline; reject missing, duplicated or conflicting keys."""
    for frame in (eeg, ecg):
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate decision keys")
        for _, group in frame.groupby(KEYS[:2]):
            times = np.sort(group.anchor_s.to_numpy(float))
            if len(times) > 1 and not np.all(np.diff(times) == 5):
                raise ValueError("Decision stride mismatch")
    common = [c for c in ("recording_duration_s", "evaluation_label", "label", "annotation_uncertain") if c in eeg and c in ecg]
    right = [c for c in ecg if c not in eeg or c in KEYS + common]
    joined = eeg.merge(ecg[right], on=KEYS, how="outer", validate="one_to_one", suffixes=("", "_ecg"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("EEG/ECG timeline or parent identity mismatch")
    for column in common:
        if not joined[column].equals(joined[column + "_ecg"]):
            raise ValueError("Canonical metadata mismatch: " + column)
    return joined.drop(columns=["_merge"] + [c + "_ecg" for c in common]).sort_values(KEYS).reset_index(drop=True)


def early_frame(frame):
    result = frame.copy()
    # Frozen EEG fit uses this prefix. Only the frozen 12 cardiac predictors enter.
    for column in ecg_baseline.COLUMNS:
        result["eeg__cardiac__" + column[5:]] = result[column].where(result.ecg_available)
    return result


def late_frame(frame, alpha, threshold, eeg_threshold, outage=False):
    if alpha not in ALPHAS:
        raise ValueError("Unregistered alpha")
    out = frame.copy()
    q = out.ecg_available.to_numpy(bool) & np.isfinite(out.score_c.to_numpy(float))
    if outage:
        q[:] = False
    score = out.score_e.to_numpy(float).copy()
    active = q & (alpha != 0)
    # Do not multiply missing scores by zero: exact fallback also covers NaNs.
    score[active] = (1-alpha)*score[active] + alpha*out.score_c.to_numpy(float)[active]
    out["q"], out["score"] = q, score
    out["effective_threshold"] = np.where(active, threshold, eeg_threshold)
    out["alpha"], out["threshold"], out["eeg_threshold"] = alpha, threshold, eeg_threshold
    return out


def evaluate(frame, events):
    """Adapt only the high/low decision; the accepted evaluator stays untouched."""
    score = frame.score.to_numpy(float)
    binary = np.where(np.isfinite(score), (score >= frame.effective_threshold.to_numpy(float)).astype(float), np.nan)
    return evaluate_record(frame.assign(score=binary), events, float(frame.recording_duration_s.iloc[0]), .5)


def aggregate(streams, events):
    metrics, matches, event_delays = [], [], {}
    for name, frame in streams.items():
        metric, _, matched, _ = evaluate(frame, events[name])
        metrics.append(metric)
        matches.extend(matched)
        event_delays.update({(name,m["event_id"]):m["delay_s"] for m in matched if m["detected"]})
    return summarize(metrics, matches), event_delays


def protection(candidate, comparator, delays, reference_delays):
    common = sorted(delays.keys() & reference_delays.keys())
    penalty = float(np.median([delays[k]-reference_delays[k] for k in common])) if common else np.nan
    reasons = []
    if candidate["detected"] < comparator["detected"]: reasons.append("sensitivity_loss")
    if candidate["far_per_h"] > comparator["far_per_h"]: reasons.append("FAR_increase")
    if not (candidate["detected"] > comparator["detected"] or candidate["far_per_h"] < comparator["far_per_h"]): reasons.append("no_strict_improvement")
    if common and penalty > 5: reasons.append("common_event_delay_penalty")
    if candidate["evaluable_s"] != comparator["evaluable_s"]: reasons.append("coverage_loss")
    return not reasons, penalty, len(common), ";".join(reasons) or "protected_improvement"


def ranking(row, adaptive=False):
    delay = row["delay_median_s"]
    return (-row["sensitivity"], row["far_per_h"], delay if np.isfinite(delay) else np.inf,
            row["alpha"] if adaptive else 0, -row["threshold"])


def select(inner, events):
    eeg = {n:f.assign(score=f.score_e) for n,f in inner.items()}
    eeg_threshold, eeg_table = baseline.choose_threshold(eeg, events)
    comparator, reference_delays = aggregate({n:late_frame(f,0,eeg_threshold,eeg_threshold) for n,f in inner.items()},events)
    rows = []
    for alpha in ALPHAS:
        for threshold in baseline.THRESHOLDS:
            streams = {n:late_frame(f,alpha,threshold,threshold if alpha == 0 else eeg_threshold) for n,f in inner.items()}
            metric, delays = aggregate(streams,events)
            allowed, penalty, n_common, reason = protection(metric,comparator,delays,reference_delays)
            rows.append(dict(**metric,alpha=alpha,threshold=threshold,feasible=metric["far_per_h"]<=1,
                eligible=alpha>0 and allowed,common_delay_delta_s=penalty,common_events=n_common,reason=reason))
    zero = [r for r in rows if r["alpha"]==0 and r["feasible"]]
    assert min(zero,key=ranking)["threshold"]==eeg_threshold
    for row, saved in zip([r for r in rows if r["alpha"]==0],eeg_table.to_dict("records")):
        for key in comparator:
            np.testing.assert_allclose(row[key],saved[key],rtol=0,atol=0,equal_nan=True)
    eligible = [r for r in rows if r["eligible"]]
    f2 = min(eligible,key=lambda r:ranking(r,True)) if eligible else next(r for r in zero if r["threshold"]==eeg_threshold)
    f1 = min([r for r in rows if r["alpha"]==.5 and r["feasible"]],key=ranking)
    early_threshold, early_table = baseline.choose_threshold({n:f.assign(score=f.score_early) for n,f in inner.items()},events)
    for row in rows:
        row["selected_f2"] = row is f2
        row["selected_f1"] = row is f1
    return dict(eeg_threshold=eeg_threshold,alpha=f2["alpha"],threshold=f2["threshold"],f1_threshold=f1["threshold"],
        f3_threshold=early_threshold,reason="protected_improvement" if eligible else "no_eligible_nonzero_candidate",
        inner_eeg_detected=comparator["detected"],inner_eeg_events=comparator["events"],inner_eeg_far=comparator["far_per_h"],
        inner_eeg_delay=comparator["delay_median_s"],inner_fusion_detected=f2["detected"],inner_fusion_far=f2["far_per_h"],
        inner_fusion_delay=f2["delay_median_s"],common_events=f2["common_events"],common_delay_delta_s=f2["common_delay_delta_s"]),pd.DataFrame(rows),early_table


def fit_scores(eeg, ecg, joined, names, validation, forbidden, cache):
    if set(names) & (set(forbidden) | {validation}):
        raise ValueError("Parent leakage")
    key=tuple(names)
    if key not in cache:
        em,columns=baseline.fit_training(eeg,names,forbidden+[validation],"E1_RF")
        cm=ecg_baseline.fit_training(ecg,names,forbidden+[validation],"C1_RF")
        fm,fcolumns=baseline.fit_training({n:early_frame(joined[n]) for n in names},names,forbidden+[validation],"E1_RF")
        cache[key]=(em,columns,cm,fm,fcolumns)
    em,columns,cm,fm,fcolumns=cache[key]
    es=baseline.predict_frame(em,columns,eeg[validation])[KEYS+["score"]].rename(columns={"score":"score_e"})
    cs=ecg_baseline.predict(cm,ecg[validation])[KEYS+["score"]].rename(columns={"score":"score_c"})
    fs=baseline.predict_frame(fm,fcolumns,early_frame(joined[validation]))[KEYS+["score"]].rename(columns={"score":"score_early"})
    columns=KEYS+["recording_duration_s","eeg_available","ecg_available","evaluation_label","annotation_uncertain","ecg__relative_hr_delta"]
    out=joined[validation][columns].copy()
    for values in (es,cs,fs): out=out.merge(values,on=KEYS,validate="one_to_one")
    out["train_files_json"]=json.dumps(names)
    return out


def inner_scores(eeg,ecg,joined,held,cache):
    names=sorted(n for n in eeg if n!=held)
    if len(names)<2: raise ValueError("Insufficient independent inner parents")
    return {v:fit_scores(eeg,ecg,joined,[n for n in names if n!=v],v,[held],cache) for v in names}


def outer_variants(stream, choice):
    output={"F0":late_frame(stream,0,choice["eeg_threshold"],choice["eeg_threshold"]),
        "F1":late_frame(stream,.5,choice["f1_threshold"],choice["eeg_threshold"]),
        "F2":late_frame(stream,choice["alpha"],choice["threshold"],choice["eeg_threshold"])}
    output["F3"]=stream.assign(score=stream.score_early,effective_threshold=choice["f3_threshold"],
        threshold=choice["f3_threshold"],eeg_threshold=choice["eeg_threshold"],alpha=np.nan,
        q=stream.ecg_available & np.isfinite(stream.score_c))
    return output
