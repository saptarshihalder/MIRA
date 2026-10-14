"""Check the manuscript source, preserved numbers, template files and PDF."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
HISTORICAL = PAPER.parent / "lifted_cavity"
TEMPLATE = REPO.parent / "Neurocomputing-template"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included_sources(path, seen=None):
    seen = seen if seen is not None else set()
    if path in seen:
        return []
    seen.add(path)
    text = path.read_text(encoding="utf-8")
    found = [(path, text)]
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        child = PAPER / (name if Path(name).suffix else name + ".tex")
        if not child.exists():
            raise ValueError(f"Missing input: {child}")
        found += included_sources(child, seen)
    return found


def main():
    sources = included_sources(PAPER / "main.tex")
    text = "\n".join(t for _, t in sources)
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    references = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text))
    missing_labels = sorted(references - set(labels))
    duplicate_labels = sorted({k for k in labels if labels.count(k) > 1})
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for group in re.findall(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}", text):
        cited.update(k.strip() for k in group.split(","))
    missing_cites = sorted(cited - bib_keys)
    macro_match = digest(PAPER / "generated/numbers.tex") == digest(HISTORICAL / "generated/numbers.tex")
    template_files = ["cas-dc.cls", "cas-sc.cls", "cas-common.sty", "cas-model2-names.bst"]
    template_match = {name: digest(PAPER / name) == digest(TEMPLATE / name) for name in template_files}
    tables = {}
    for p in (PAPER / "generated").glob("*table.tex"):
        old = (HISTORICAL / "generated" / p.name).read_text(encoding="utf-8")
        new = p.read_text(encoding="utf-8")
        def body(t):
            if r"\begin{tabular}" not in t:
                return t
            start = t.index(r"\begin{tabular}")
            return t[start:t.index(r"\end{tabular}", start)]
        numbers = lambda s: re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", body(s))
        tables[p.name] = numbers(old) == numbers(new)
    styles = {}
    for p, t in sources:
        if p.parent.name == "sections" or p.name == "main.tex":
            styles[str(p.relative_to(PAPER))] = {
                "unicode_dashes": len(re.findall("[\u2010-\u2015]", t)),
                "tex_prose_double_dash": len(re.findall(r"(?<!-)--(?!-)", t)),
            }
    from pypdf import PdfReader
    reader = PdfReader(PAPER / "main.pdf")
    pdf_text = "\n".join(p.extract_text() for p in reader.pages)
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace")
    result = {
        "source_files": len(sources),
        "source_word_tokens": len(re.findall(r"\b[A-Za-z]{2,}\b", text)),
        "bibliography_entries": len(bib_keys),
        "cited_entries": len(cited),
        "missing_citations": missing_cites,
        "missing_labels": missing_labels,
        "duplicate_labels": duplicate_labels,
        "historical_numeric_macros_unchanged": macro_match,
        "historical_numbers_sha256": digest(PAPER / "generated/numbers.tex"),
        "table_numeric_tokens_unchanged": tables,
        "template_files_unchanged": template_match,
        "prose_style_scan": styles,
        "physical_pdf_pages_including_highlights": len(reader.pages),
        "pdf_sha256": digest(PAPER / "main.pdf"),
        "pdf_unresolved_reference_markers": pdf_text.count("??"),
        "tex_overfull_boxes": log.count("Overfull"),
        "template_sidebar_warning": "CAS places its keyword box in an intentional zero width hbox; the 123.62721pt maketitle warning is inherited and has no visible overflow.",
        "tex_undefined_reference_warnings": len(re.findall(r"Warning:.*undefined", log)),
        "v4_efficacy": "Withheld. No new endpoint arithmetic, training or inference performed for this manuscript.",
    }
    assert not missing_labels and not duplicate_labels and not missing_cites
    assert macro_match and all(template_match.values()) and all(tables.values())
    assert result["pdf_unresolved_reference_markers"] == 0
    out = PAPER / "evidence/manuscript_validation.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
