from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

from src.multimodal_seizure.data import get_official_channels, read_edf_segment, read_event_table

OUT = Path(__file__).resolve().parent / 'ecg_signal_qc.csv'
events = read_event_table()
rows=[]

def qrs_summary(x, fs):
    x=np.asarray(x,float)
    x=x-np.median(x)
    sos=butter(3,[5,25],btype="bandpass",fs=fs,output="sos")
    y=sosfiltfilt(sos,x)
    scale=np.median(np.abs(y-np.median(y))) * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        return {"n_peaks":0,"median_hr_bpm":np.nan,"rr_cv":np.nan,"snr_proxy":0.0}
    z=np.abs(y)/scale
    peaks,_=find_peaks(z,distance=int(0.30*fs),prominence=2.5)
    rr=np.diff(peaks)/fs
    rr=rr[(rr>=0.30)&(rr<=2.0)]
    hr=60/rr if len(rr) else np.array([])
    return {
        "n_peaks":len(peaks),
        "median_hr_bpm":float(np.median(hr)) if len(hr) else np.nan,
        "rr_cv":float(np.std(rr)/np.mean(rr)) if len(rr)>1 else np.nan,
        "snr_proxy":float(np.percentile(z,95)),
    }

for patient in ["PN00","PN06","PN10","PN12","PN14"]:
    ev=events[events.patient==patient].iloc[0]
    ecg=get_official_channels(patient,"ECG")
    start=max(0.0,float(ev.onset_relative_s)-120.0)
    indices=list(ecg.edf_index_0based.astype(int))
    data,fs=read_edf_segment(patient,ev.canonical_file_name,indices,start_s=start,duration_s=60.0)
    corr=float(np.corrcoef(data[0],data[1])[0,1])
    for j,row in ecg.iterrows():
        s=qrs_summary(data[j],fs)
        rows.append({
            "patient":patient,
            "file":ev.canonical_file_name,
            "official_ecg":row.official_name,
            "raw_label":row.edf_raw_label,
            "fs_hz":fs,
            "segment_start_s":start,
            "channel_pair_corr":corr,
            **s,
        })
df=pd.DataFrame(rows)
df.to_csv(OUT,index=False)
print(df.to_string(index=False))
print("OUTPUT",OUT)
