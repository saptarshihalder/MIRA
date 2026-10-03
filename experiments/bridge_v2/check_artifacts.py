"""Final source/identity checks; these are not a substitute for native PDF compilation."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def main():
    paper = ROOT / 'paper/main.tex'
    source = paper.read_text()
    if r'Current conservative provisions are \$14.05' not in source:
        raise ValueError('Current manuscript budget does not match this completed gate')
    environments = []
    for action, name in re.findall(r'\\(begin|end)\{([^}]+)\}', source):
        if action == 'begin':
            environments.append(name)
        elif not environments or environments.pop() != name:
            raise ValueError('Unbalanced LaTeX environment: ' + name)
    if environments:
        raise ValueError('Unclosed environments')
    bibliography = set(re.findall(r'\\bibitem\{([^}]+)\}', source))
    citations = {key.strip() for item in re.findall(r'\\cite\{([^}]+)\}', source) for key in item.split(',')}
    labels = set(re.findall(r'\\label\{([^}]+)\}', source))
    references = set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', source))
    if not citations <= bibliography or not references <= labels:
        raise ValueError('Unresolved citation/reference')
    files = 0
    for version in (1, 2):
        protocol = ROOT / f'configs/bridge_v{version}.json'
        plan = json.loads(protocol.read_text())
        for relative, expected in plan['file_sha256'].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen source changed: ' + relative)
            files += 1
        audit = json.loads((ROOT / f'artifacts/reports/bridge_v{version}_a100/independent_audit.json').read_text())
        if not audit['passed'] or audit['protocol_sha256'] != hashlib.sha256(protocol.read_bytes()).hexdigest():
            raise ValueError('Audit/protocol mismatch')
    manifest = dict(path=str(paper), source_sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
        native_compiler_status='compile-failed', diagnostic='Unable to find standard directories for platform',
        pdf_verified=False, source_checks=dict(environments_balanced=True, citations_resolved=True, references_resolved=True),
        bridge_frozen_files_rechecked=files,
        limitation='Native compiler environment failure; source checks do not verify PDF layout. Same editor remains open.')
    (ROOT / 'artifacts/manifests/paper_compile.json').write_text(json.dumps(manifest, indent=2) + '\n', newline='\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
