"""Keep verification URLs in evidence and standard publication details in print."""
import json
import re
from pathlib import Path

paper = Path(__file__).resolve().parents[1]
path = paper / "references.bib"
text = path.read_text(encoding="utf-8")
entries = re.split(r"(?=^@\w+\{)", text, flags=re.M)
links = {}
out = []
for entry in entries:
    key_match = re.match(r"@\w+\{([^,]+),", entry)
    if not key_match:
        out.append(entry)
        continue
    key = key_match.group(1)
    url = re.search(r"^\s*url\s*=\s*\{([^}]+)\},?\s*$", entry, re.M)
    doi = re.search(r"^\s*doi\s*=\s*\{([^}]+)\},?\s*$", entry, re.M)
    links[key] = {"primary_url": url.group(1) if url else None, "doi": doi.group(1) if doi else None}
    # Preprints and unpaginated ICLR articles retain compact document links.
    keep_url = "arXiv preprint" in entry or ("ICLR" in entry and not doi)
    if url and not keep_url:
        entry = re.sub(r"^\s*url\s*=\s*\{[^}]+\},?\s*\n", "", entry, flags=re.M)
    out.append(entry)
path.write_text("".join(out), encoding="utf-8")
(paper / "evidence/bibliography_links.json").write_text(
    json.dumps(links, indent=2) + "\n", encoding="utf-8")
print(f"Retained metadata for {len(links)} entries; verification URLs saved separately.")
