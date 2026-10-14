"""Content digest of fetch_v9/fetch_v10 snapshots (array bytes, independent of zip timestamps)."""
import hashlib, os, sys
import numpy as np

for f in sys.argv[1:]:
    d = np.load(f)
    h = hashlib.sha256()
    for k in sorted(d.files):
        h.update(k.encode()); h.update(np.ascontiguousarray(d[k]).tobytes())
    print('DIGEST', os.path.basename(f), h.hexdigest()[:16], d['V'].shape)
