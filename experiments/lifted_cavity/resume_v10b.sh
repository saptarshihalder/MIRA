#!/bin/bash
# After the UPT-SF fine-tunes already running finish, run what is still missing four at a time (L4, 22 GB) with the
# out-of-memory retry of finetune_real.py, then score, pool and confirm.
: mira_chain_F10_marker
D=/content/drive/MyDrive
while pgrep -f 'finetune_real[.]py' >/dev/null; do sleep 30; done
echo "== resume b: MIRA_PAR=4 with OOM retry $(date -u +%H:%M:%S)" >> $D/mira_chain_F10.log
cd /content/MIRA/experiments/lifted_cavity || exit 1
MIRA_PAR=4 python3 colab_v10.py --root $D/mira_v10 --v9 $D/mira_v9 --v8 $D/mira_v8 --v7 $D/mira_v7 --v6 $D/mira_v6 \
    ft ftsel score pool confirm confirm11 pack >> $D/mira_chain_F10.log 2>&1
echo "CHAIN_F10_EXIT $?" >> $D/mira_chain_F10.log
