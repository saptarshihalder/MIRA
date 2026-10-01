"""Trained selector over fixed support moments; CPU prototype, not TFM evidence."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from feasibility import ROOT, library, basis, transform
from mira.data import Mechanism, Support, _sample

CODES = np.array([b for b in itertools.product((0, 1), repeat=4) if any(b)], dtype=np.uint8)
MATRICES = library(4)


def encode(support):
    if not isinstance(support, Support) or support.observations.mask.shape[1] != 4:
        raise TypeError("Four-bit labeled support only")
    m = support.observations.mask.astype(int)
    parity = 1. - 2. * ((m @ CODES.T) % 2)
    y = 2. * support.labels - 1.
    return np.r_[np.abs(np.mean(parity * y[:, None], axis=0)),
                 np.mean(support.observations.u * y), np.log2(len(y))/8.].astype(np.float32)


class Compiler(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 12))

    def select(self, support, confidence=.8):
        self.eval()
        with torch.no_grad():
            p = self(torch.from_numpy(encode(support))[None]).softmax(-1)[0]
        index = int(p.argmax()) if float(p.max()) >= confidence else 0
        return MATRICES[index].copy(), {"index":index,"max_probability":float(p.max()),"threshold":confidence}

    def forward(self, x):
        return self.net(x)


def dataset(start, count):
    xs, ys = [], []
    for seed in range(start, start+count):
        rng = np.random.default_rng([seed, 777])
        active = tuple(sorted(rng.choice(4, int(rng.integers(1, 5)), replace=False).tolist()))
        gamma = float(rng.choice([0., .25, .5, .9]))
        task = Mechanism("trained_compiler_development", 4, active, int(rng.choice([-1, 1])), gamma,
                         beta=float(rng.uniform(.2, 1.2)))
        obs, targets = _sample(task, int(rng.choice([32, 128])), np.random.default_rng([seed, 777, 1]), np.random.default_rng([seed, 777, 2]))
        xs.append(encode(Support(obs, targets.labels)))
        # Privileged synthetic teacher supplies meta-training targets, never inference inputs.
        a = np.eye(4, dtype=np.uint8) if gamma == 0 else basis([int(j in active) for j in range(4)])
        ys.append(next(i for i, b in enumerate(MATRICES) if np.array_equal(a, b)))
    return np.array(xs), np.array(ys, dtype=np.int64)


def main(out):
    if out.exists():
        raise ValueError("Preserve old outputs; choose a new directory")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(901)
    started = time.monotonic()
    x, y = dataset(81000, 2048)
    vx, vy = dataset(85000, 256)
    tx, ty = dataset(87000, 256)
    model = Compiler()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    losses = []
    for epoch in range(80):
        model.train()
        for indices in torch.randperm(len(xt)).split(128):
            loss = nn.functional.cross_entropy(model(xt[indices]), yt[indices])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    # Fixed final epoch; no evaluation-based checkpoint or threshold selection.
    state = {k: v.detach().numpy() for k,v in model.state_dict().items()}
    checkpoint = out / "trained_parameters.npz"
    np.savez_compressed(checkpoint, **state)
    model.eval()
    with torch.no_grad():
        vp = model(torch.from_numpy(vx)).softmax(-1).numpy()
        tp = model(torch.from_numpy(tx)).softmax(-1).numpy()
    np.savez_compressed(out / "heldout_predictions.npz", p=tp, teacher=ty, features=tx)
    raw = tp.argmax(1)
    guarded = np.where(tp.max(1)>=.8, raw, 0)
    # Fixed moment heuristic, same observable features; not pretraining-budget matched.
    heuristic = []
    for row in tx:
        j = int(row[:15].argmax())
        a = np.eye(4,dtype=np.uint8) if row[j]<.3 else basis(CODES[j])
        heuristic.append(next(i for i,b in enumerate(MATRICES) if np.array_equal(a,b)))
    report = {"status":"trained CPU prototype","parameters":sum(p.numel() for p in model.parameters()),
              "train_tasks":2048,"validation_tasks":256,"heldout_development_tasks":256,"epochs":80,
              "train_seeds":[81000,83047],"validation_seeds":[85000,85255],"heldout_development_seeds":[87000,87255],
              "rng_namespace":777,"seconds":time.monotonic()-started,"paid_usd":0,
              "validation_teacher_accuracy":float(np.mean(vp.argmax(1)==vy)),"raw_teacher_accuracy":float(np.mean(raw==ty)),
              "guarded_teacher_accuracy":float(np.mean(guarded==ty)),"identity_teacher_accuracy":float(np.mean(ty==0)),
              "moment_heuristic_teacher_accuracy":float(np.mean(np.array(heuristic)==ty)),
              "guarded_nonidentity_fraction":float(np.mean(guarded!=0)),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"torch":torch.__version__,
              "feature_encoding":"Fixed permutation-invariant15 absolute parity/label moments,U-label moment,log-context size; learned6092-parameter MLP selector, not learned raw-row encoder.",
              "limits":"Privileged active/gamma teacher during synthetic pretraining; teacher recovery is not downstream loss, calibration, novelty, TFM benefit or external generalization. Confidence fallback has no no-harm guarantee. All new seeds are now used development. Frozen backbone integration and pretraining-budget-matched trained baselines pending."}
    (out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    (out / "training_losses.json").write_text(json.dumps(losses)+"\n")
    print(json.dumps({k:report[k] for k in ("parameters","seconds","raw_teacher_accuracy","guarded_teacher_accuracy","moment_heuristic_teacher_accuracy","identity_teacher_accuracy","paid_usd")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/reports/trained_compiler_v0")
    main(parser.parse_args().out)
