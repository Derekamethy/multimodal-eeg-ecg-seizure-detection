from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'data/raw/siena-scalp-eeg-1.0.0'

def field(b):
    return b.decode("latin-1", errors="replace").strip()

def parse(path):
    with path.open("rb") as f:
        fixed=f.read(256)
        hb=int(field(fixed[184:192]))
        ns=int(field(fixed[252:256]))
        sh=f.read(hb-256)
    widths=[16,80,8,8,8,8,8,80,8,32]
    names=["label","transducer","phys_dim","phys_min","phys_max","dig_min","dig_max","prefilter","samples","reserved"]
    pos=0
    cols={}
    for name,w in zip(names,widths):
        block=sh[pos:pos+w*ns]; pos+=w*ns
        cols[name]=[field(block[i*w:(i+1)*w]) for i in range(ns)]
    return cols

for p in sorted((ROOT/"PN14").glob("*.edf")):
    cols=parse(p)
    print("\n",p.name)
    for i,label in enumerate(cols["label"]):
        if not label.upper().startswith("EEG "):
            print(i, label, "| transducer=", cols["transducer"][i], "| unit=", cols["phys_dim"][i],
                  "| prefilter=", cols["prefilter"][i], "| samples=", cols["samples"][i])
