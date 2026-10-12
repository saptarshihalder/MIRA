"""Content digest of an evaluation panel (episode keys, every array of the pool and the condition masks), to check that
panels built on different machines from the same data are identical."""
import hashlib, os, sys
import numpy as np
import torch


def digest(path):
    p = torch.load(path, weights_only=False)
    h = hashlib.sha256('|'.join(p['pool']['keys']).encode())
    for k in ('sx', 'sy', 'sm', 'qx', 'qy', 'qnat'):
        h.update(p['pool'][k].numpy().tobytes())
    for e in sorted(p['conds']):
        h.update(np.ascontiguousarray(p['conds'][e]).tobytes())
    return h.hexdigest()[:16]


if __name__ == '__main__':
    for f in sys.argv[1:]:
        print('PANEL', os.path.basename(f), digest(f))
