. ./qlib.sh
# Post hoc: the lifted cavity transformer fine-tuned on the Air Quality targets (same recipe)
ft airq_co runs/lct_s1 runs_real/airq_co/lct_s1_ft real_airq_co_s2027
ft airq_no2 runs/lct_s1 runs_real/airq_no2/lct_s1_ft real_airq_no2_s2027
echo CORE3_DONE
