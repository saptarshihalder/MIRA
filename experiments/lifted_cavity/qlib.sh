# Idempotent job helpers: every step is skipped when its output already exists, so a queue can be relaunched after a restart.
. ./env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
done30k() { [ "$(python3 -c "import json;print(json.load(open('$1/train.json')).get('steps_done',0))" 2>/dev/null)" = "$2" ]; }
train_lct() { done30k $2 30000 && return 0; [ -f $2/ckpt.pt ] || rm -rf $2; python3 lct_train.py --steps 30000 --seed $1 --out $2; }
train_pfn() { done30k $2 30000 && return 0; [ -f $2/ckpt.pt ] || rm -rf $2; python3 pfn_train.py --steps 30000 --seed $1 --batch-tasks 16 --out $2; }
score_syn() { [ -f $1/cells_$2.npz ] || python3 evaluate2.py model --run $1 --panel cache/$2.pt; }
ft() {  # dataset init out panel_tag
  [ -f $3/cells_$4.npz ] && return 0
  if [ ! -f $3/model.pt ]; then
    [ -f $3/ckpt.pt ] || rm -rf $3
    python3 finetune_real.py --dataset $1 --init $2 --out $3 || return 1
  fi
  python3 eval_real.py score --panel cache/$4.pt --runs $3
}
G1=v2_s20261301_n256_p5_nl0.4_sr48
G3=v2_s20261303_n128_p16_nl0.4_sr48
F1=v2_s20261201_n256_p5_nl0.4_sr48
F3=v2_s20261203_n128_p16_nl0.4_sr48
V2PANELS="v2_s20261201_n256_p5_nl0.4_sr48 v2_s20261202_n128_p8_nl0.4_sr48 v2_s20261203_n128_p16_nl0.4_sr48 v2_s20261204_n128_p5_nl0.8_sr48 v2_s20261205_n128_p5_nl0.0_sr48 v2_s20261206_n128_p5_nl0.4_sr16 v2_s20261207_n128_p5_nl0.4_sr96"
