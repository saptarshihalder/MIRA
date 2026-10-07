. ./qlib.sh
# Protocol v3 comparators
ft beijing_no2 runs/pfn_s1 runs_real/beijing_no2/pfn_s1_ft real_beijing_no2_s3031
ft beijing_co runs/pfn_s1 runs_real/beijing_co/pfn_s1_ft real_beijing_co_s3031
echo V3_COMPARATORS_DONE
# Post hoc (robustness, descriptive)
train_pfn 2 runs/pfn_s2
for p in $F1 $F3 $G1 $G3; do score_syn runs/pfn_s2 $p; done
ft beijing_no2 runs/pfn_s2 runs_real/beijing_no2/pfn_s2_ft real_beijing_no2_s3031
ft beijing runs/pfn_s2 runs_real/beijing/pfn_s2_ft real_beijing_s2027
[ -f runs_posthoc/gp/cells_$F1.npz ] || python3 posthoc_gp.py $F1
[ -f runs_posthoc/gp/cells_real_beijing_s2027.npz ] || python3 posthoc_gp.py real_beijing_s2027
echo CORE2_DONE
