#!/bin/bash
# Protocol v10/v11 on an L4: after the running chain F and the usgs_ny panel build finish, rerun what is missing
# (the UPT-SF fine-tunes that ran out of GPU memory, everything for usgs_ny) at MIRA_PAR=5, then score, pool, confirm.
: mira_chain_F10_marker
D=/content/drive/MyDrive
while pgrep -f 'colab_v10[.]py .* F$' >/dev/null || pgrep -f 'eval_real[.]py build' >/dev/null; do sleep 30; done
echo "== resume: OOM reruns at MIRA_PAR=5 and usgs_ny $(date -u +%H:%M:%S)" >> $D/mira_chain_F10.log
cd /content/MIRA && git pull -q origin lifted-site-pool-v5 && cd experiments/lifted_cavity || exit 1
MIRA_PAR=5 python3 colab_v10.py --root $D/mira_v10 --v9 $D/mira_v9 --v8 $D/mira_v8 --v7 $D/mira_v7 --v6 $D/mira_v6 \
    panels pools krig ft ftsel score pool confirm confirm11 pack >> $D/mira_chain_F10.log 2>&1
echo "CHAIN_F10_EXIT $?" >> $D/mira_chain_F10.log
