#!/bin/sh
# Protocol v3 (docs/LIFTED_CAVITY_PROTOCOL_V3.md): the exact commands, in order. Run from experiments/lifted_cavity
# with MIRA_BEIJING_DIR pointing at the PRSA_Data_20130301-20170228 folder and MIRA_LC_CACHE at a cache directory.
# Single-threaded throughout (OMP_NUM_THREADS=1); about 12 CPU core-hours in total.
set -e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
G1=v2_s20261301_n256_p5_nl0.4_sr48
G3=v2_s20261303_n128_p16_nl0.4_sr48

# 1. The one new model (final checkpoint, no selection).
python lct_train.py --steps 30000 --seed 1 --out runs/lct_s1

# 2. New synthetic panels and real targets (never scored before this protocol).
python evaluate2.py panel --seed 20261301 --tasks 256 --sensors 5
python evaluate2.py panel --seed 20261303 --tasks 128 --sensors 16
python eval_real.py build --dataset beijing_no2 --seed 3031 --out cache/real_beijing_no2_s3031.pt
python eval_real.py build --dataset beijing_co --seed 3031 --out cache/real_beijing_co_s3031.pt

# 3. Synthetic scores.
for r in lct_s1 pfn_s1 pfnall_s1 lift1_s1 lift1_s2 lift1_s3; do
  for p in $G1 $G3; do python evaluate2.py model --run runs/$r --panel cache/$p.pt; done
done

# 4. Fine-tuning (protocol v2 recipe) and real scores.
for ds in beijing_no2 beijing_co; do
  for r in lct_s1 pfn_s1 lift1_s1 lift1_s2 lift1_s3; do
    python finetune_real.py --dataset $ds --init runs/$r --out runs_real/$ds/${r}_ft
    python eval_real.py score --panel cache/real_${ds}_s3031.pt --runs runs_real/$ds/${r}_ft
  done
done

# 5. Endpoints E5-E7b.
python confirm3.py

# 6. Descriptive only: the lifted cavity transformer on the protocol v2 panels.
for p in v2_s20261201_n256_p5_nl0.4_sr48 v2_s20261202_n128_p8_nl0.4_sr48 v2_s20261203_n128_p16_nl0.4_sr48 \
         v2_s20261204_n128_p5_nl0.8_sr48 v2_s20261205_n128_p5_nl0.0_sr48 v2_s20261206_n128_p5_nl0.4_sr16 \
         v2_s20261207_n128_p5_nl0.4_sr96; do
  python evaluate2.py model --run runs/lct_s1 --panel cache/$p.pt
done
