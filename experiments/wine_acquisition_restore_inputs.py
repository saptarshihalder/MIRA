"""Restore hash-pinned public inputs without replacing the frozen data audit."""
import io, json
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import wine_acquisition_prepare as p

def materialize(path, payload, expected):
    if p.sha256(payload) != expected:
        raise RuntimeError(f'Pinned bytes differ: {path}. Check recorded runtime versions.')
    if path.exists():
        if p.sha256(path.read_bytes()) != expected: raise RuntimeError(f'Refuse to overwrite changed input: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        p.verify_ignored(path)
        path.write_bytes(payload)

def main():
    audit = json.loads(p.AUDIT.read_text())
    for color in ('red', 'white'):
        source = audit['sources'][color]; raw = p.ROOT/source['path']
        payload = raw.read_bytes() if raw.exists() else urlopen(Request(source['url'], headers={'User-Agent':'MIRA-reproduction/1.0'}),timeout=60).read()
        materialize(raw,payload,source['sha256'])
        d = p.parse(payload,color)
        groups = np.asarray([p.sha256(p.canonical_features(x)) for x in d['raw_X']], dtype='U64')
        split = np.asarray([p.partition_for(g) for g in groups],dtype=np.uint8)
        thresholds = np.quantile(d['raw_X'][split==0],[1/3,2/3],axis=0,method='linear').T
        x = (d['raw_X'][:,:,None]>=thresholds[None,:,:]).sum(2).astype(np.uint8)
        b = io.BytesIO()
        np.savez_compressed(b,X=x,y=d['y'],row_ids=d['row_ids'],groups=groups,partition=split,thresholds=thresholds,feature_names=np.asarray(p.FEATURES))
        out = audit['outputs'][color]
        materialize(p.ROOT/out['path'],b.getvalue(),out['sha256'])
    print('Both public raw inputs and reconstructed NPZ byte hashes match; no fitting or scoring.')

if __name__=='__main__': main()
