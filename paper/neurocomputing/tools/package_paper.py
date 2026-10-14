"""Package the article and its standalone figures without model artifacts."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

paper = Path(__file__).resolve().parents[1]
workspace = paper.parents[2]
output = workspace / "MIRA_Neurocomputing_Paper.zip"
names = [
    "main.tex", "main.pdf", "main.bbl", "references.bib", "README.md",
    "cas-dc.cls", "cas-sc.cls", "cas-common.sty", "cas-model2-names.bst",
    "cas-refs.bib", "manifest.txt",
]
files = [paper / name for name in names]
for folder in ("sections", "generated", "figures", "evidence"):
    files.extend(p for p in (paper / folder).rglob("*") if p.is_file())
for name in ("draw_figures.py", "verify_manuscript.py", "package_paper.py"):
    files.append(paper / "tools" / name)
files = sorted(set(files))
manifest = {
    "prepared_date": "2026-10-14",
    "source_repository": "https://github.com/saptarshihalder/MIRA",
    "source_revision": "9afab891f5e4c8b553e668e84731caff8f2aa4d3",
    "template_source": "https://github.com/xhonghu/Paper-Template-Utils/tree/master/Elsevier%20%20%28neurocomputing%29",
    "compiler": "Tectonic 0.17.0",
    "layout": "Unmodified Elsevier CAS double column class, a4paper and fleqn",
    "scientific_scope": "Historical v1 to v3 evidence; v4 efficacy withheld",
    "new_figures": 3,
    "figure_exports": ["PDF vector", "SVG vector", "PNG 5184 pixels wide"],
    "files": {
        p.relative_to(paper).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        } for p in files if p.name != "package_manifest.json"
    },
}
manifest_path = paper / "evidence/package_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
if manifest_path not in files:
    files.append(manifest_path)
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for p in files:
        archive.write(p, p.relative_to(paper).as_posix())
with ZipFile(output) as archive:
    assert archive.testzip() is None
print(json.dumps({"zip": str(output), "bytes": output.stat().st_size, "files": len(files)}, indent=2))
