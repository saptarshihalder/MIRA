"""Retrieve a completed fixed v4 run; never start compute or score partial results."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

CALL = 'fc-01M4KMG0FQ3CYTVKSRGQVGC6DF'
VOLUME = 'mira-lifted-v4-l40s'
REMOTE = 'lifted_v4_l40s_20261009'
SIDECARS = ('modal_main_resume_runtime.json', 'main_resume_execution_amendment.json', 'main_resume_execution_gate.json', 'modal_main_extended_runtime.json', 'main_extended_execution_amendment.json',
    'main_extended_execution_gate.json', 'modal_identity.json', 'modal_smoke_gate.json',
    'modal_smoke_runtime.json', 'pretrained_weights.json', 'source_profile.json',
    'cache/tabpfn/tabpfn-v2-regressor.ckpt')

def validate_members(z):
    names = set()
    members = z.infolist()
    if len(members) > 10000 or sum(x.file_size for x in members) > 512 * 1024**2:
        raise ValueError('Unexpected archive expansion')
    for entry in members:
        name = entry.filename
        p = PurePosixPath(name)
        if (p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name
                or name in names or stat.S_ISLNK(entry.external_attr >> 16)):
            raise ValueError('Unsafe or duplicate archive member')
        names.add(name)
        if not entry.is_dir() and (p.suffix not in ('.json', '.jsonl', '.npz', '.log')
                or p.parts[0] not in ('runs', 'runs_real', 'smoke', 'logs',
                    'manifest.json', 'run_identity.json', 'sessions.jsonl')):
            raise ValueError('Unexpected scientific archive content')
    if 'runs/confirm_v4.json' not in names:
        raise ValueError('Complete confirmation is required before collection')
    manifest = json.loads(z.read('manifest.json'))
    if manifest.get('failed'):
        raise ValueError('Upstream failures remain; preserve and investigate them')
    identity = json.loads(z.read('run_identity.json'))
    if identity['metadata']['commit'] != 'e1b1d9d09e98667dc4590e6f54a6c6641c3e24a9':
        raise ValueError('Unexpected scientific source identity')
    return members

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collect', action='store_true', help='Requires fully completed Modal call')
    a = ap.parse_args()
    destination = Path(__file__).resolve().parents[1] / 'artifacts/runs/lifted_v4_final'
    models = [f'runs/{m}_s{s}/model.pt' for s in (1, 2, 3) for m in ('pfn_L', 'lct_L')]
    models += [f'runs_real/{d}/{m}_s{s}_ft/model.pt'
        for d in ('beijing_pm10', 'beijing_so2', 'beijing_o3')
        for s in (1, 2, 3) for m in ('pfn_L', 'lct_L')]
    if not a.collect:
        print(json.dumps(dict(call=CALL, destination=str(destination), models=len(models),
            requires='completed call and zero recorded failures; no compute launch')))
        return
    import modal
    result = modal.FunctionCall.from_id(CALL).get(timeout=0)
    if result.get('status') != 'completed' or result.get('exit_code') != 0:
        raise ValueError('Scientific job incomplete or failed; no partial endpoint import')
    destination.mkdir(exist_ok=False)
    (destination / 'INCOMPLETE').write_text('Collection/independent verification pending.\n')
    volume = modal.Volume.from_name(VOLUME)
    def download(relative):
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            for chunk in volume.read_file(REMOTE + '/' + relative):
                f.write(chunk)
        return path
    archive = download('mira_v4_results.zip')
    if archive.stat().st_size > 256 * 1024**2:
        raise ValueError('Unexpected archive size')
    with zipfile.ZipFile(archive) as z:
        for entry in validate_members(z):
            if not entry.is_dir():
                path = destination / entry.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as f:
                    f.write(z.read(entry))
    for relative in SIDECARS + tuple(models):
        download(relative)
    inventory = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file() and path.name != 'INCOMPLETE':
            with path.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            inventory[path.relative_to(destination).as_posix()] = dict(sha256=digest, bytes=path.stat().st_size)
    (destination / 'collection_manifest.json').write_bytes((json.dumps(dict(
        call_id=CALL, result=result, files=inventory, independent_verification='pending'), indent=2)+'\n').encode())
    (destination / 'INCOMPLETE').unlink()
    print(json.dumps(dict(destination=str(destination), files=len(inventory),
        independent_verification='pending; do not update efficacy claims yet')))

if __name__ == '__main__':
    main()
