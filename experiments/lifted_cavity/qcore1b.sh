. ./qlib.sh
# Protocol v3 critical path (continued after reordering: descriptive v2-panel scores before the CO fine-tune)
train_lct 1 runs/lct_s1
for p in $G1 $G3; do score_syn runs/lct_s1 $p; done
ft beijing_no2 runs/lct_s1 runs_real/beijing_no2/lct_s1_ft real_beijing_no2_s3031
while [ ! -f runs_real/beijing_no2/pfn_s1_ft/cells_real_beijing_no2_s3031.npz ]; do sleep 60; done
[ -f runs/confirm_v3.json ] || python3 confirm3.py
echo CONFIRM3_DONE
for p in $V2PANELS; do score_syn runs/lct_s1 $p; done
ft beijing_co runs/lct_s1 runs_real/beijing_co/lct_s1_ft real_beijing_co_s3031
echo V3_DESCRIPTIVE_DONE
# Post hoc (robustness, descriptive)
ft beijing runs/lct_s1 runs_real/beijing/lct_s1_ft real_beijing_s2027
train_lct 2 runs/lct_s2
for p in $F1 $F3 $G1 $G3; do score_syn runs/lct_s2 $p; done
ft beijing_no2 runs/lct_s2 runs_real/beijing_no2/lct_s2_ft real_beijing_no2_s3031
echo CORE1_DONE
