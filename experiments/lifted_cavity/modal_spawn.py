"""Start TabPFN on Modal as server-side calls of the deployed app (they keep running if this machine disconnects).

    python -m modal deploy modal_tabpfn.py            (once, with MIRA_V10_DATA set)
    python modal_spawn.py tabpfn usgs_wa,dwd_p,aurn_pm25
    python modal_spawn.py build_panel usgs_ny          (then tabpfn usgs_ny)
"""
import sys
import modal

fn = modal.Function.from_name('mira-v10-tabpfn', sys.argv[1])
for net in sys.argv[2].split(','):
    call = fn.spawn(net)
    print(sys.argv[1], net, call.object_id, flush=True)
