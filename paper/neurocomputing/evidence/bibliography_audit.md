# Bibliography metadata audit

Date: October 14, 2026

The bibliography retains the 42 citation keys in the repository paper. Published titles and author order are retained. This pass completes traceable bibliographic fields rather than adding references to increase the bibliography's length.

## Sources inspected in this pass

The complete local bibliography was compared with both repository audit records:

1. `MIRA/docs/LIFTED_REFERENCE_VERIFICATION.md`, dated October 9, which records primary checks for eight closely related works.
2. `MIRA/docs/LIFTED_REMAINING_REFERENCES.md`, dated October 10 with an October 11 closure, which records primary checks for the other 34 entries.

Those records distinguish publisher and proceedings metadata from author posted manuscripts, and preserve the version choice for Josse et al. The present pass reads and uses their recorded metadata. It does not represent every historical retrieval as a new live inspection.

BrowserOS neo was used to inspect the current official NeurIPS records and their linked BibTeX exports for the following six entries:

| Key | Primary record | Live finding |
| --- | --- | --- |
| `heess2013learning` | [NeurIPS 2013](https://proceedings.neurips.cc/paper/2013/hash/1714726c817af50457d810aae9d27a2e-Abstract.html) | Official export confirms volume 26, publisher Curran Associates, and an empty pages field. |
| `eslami2014just` | [NeurIPS 2014](https://proceedings.neurips.cc/paper_files/paper/2014/hash/c04f25be56ab86371563568dce31808f-Abstract.html) | Official export confirms volume 27, publisher Curran Associates, and an empty pages field. |
| `wu2018mvae` | [NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html) | Official export confirms volume 31, publisher Curran Associates, and an empty pages field. |
| `lemorvan2020neumiss` | [NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/42ae1544956fbe6e09242e6cd752444c-Abstract.html) | Official export confirms volume 33 and pages 5980 to 5990. These pages were added. |
| `zaheer2017deepsets` | [NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) | Official export confirms volume 30, publisher Curran Associates, and an empty pages field. The record and export use different familiar forms of an author's name; the previously verified author identity and order are retained. |
| `deisenroth2012ep` | [NeurIPS 2012](https://proceedings.neurips.cc/paper/2012/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html) | Official export confirms volume 25, publisher Curran Associates, and an empty pages field. |

The official [UAI 2015 proceedings page](https://www.auai.org/uai2015/proceedings.shtml) was inspected for `jitkrittum2015kernel`. It confirms the paper, author order, and paper 235 PDF link, but its listing supplies no page range. No range was added. A navigation to the eScholarship item for `gershman2014amortized` did not produce a readable item page in this pass; that entry uses the exact primary manuscript URL preserved in the earlier audit and no new pagination claim.

The parent preparation session also reports live checks of the PMLR Qu and Lee records, Nature's TabPFN article, and the Copernicus Zimmerman article. Their exact fields added here are independently documented in the repository audits. This note does not substitute that report for a fresh full text inspection by this subtask.

## Fields completed

Every entry now has a primary URL or a DOI resolver URL. Twenty entries have verified DOI fields, including the two already present before this pass. The additional DOI values come directly from the publisher or Crossref metadata identified in the repository audits. A DOI resolver URL is used when the earlier audit recorded a DOI deposit rather than a stable publisher page.

PMLR volumes and series were completed for Garnelo et al., Lee et al., Finn et al., and Qu et al. The Neal chapter now includes its verified chapter number and pages 113 to 162. Verified issue numbers were added for the AAAI SMIL paper, the JMLR NUTS paper, and the Zimmerman article. The IFNet author initials were expanded to the names verified in its publisher record. Póczos and Smola name forms were normalized using the earlier primary metadata audit. Existing Gregor pages 399 to 406 and the Anderson and Rubin publication year 1956 remain supported by the October 11 primary source closure.

The Josse citation remains specifically attached to arXiv version 1 and its four author list. The more recent arXiv author list is not silently substituted. ICLR conference years likewise remain the conference years, rather than the earlier arXiv upload dates.

## Pagination and style warnings

No page ranges were inferred from memory or fabricated to silence BibTeX. Five official NeurIPS exports checked here have empty pages fields. The four ICLR entries do not receive fictitious proceedings pagination. The UAI Jitkrittum and Cognitive Science Gershman entries also retain no page range because a range was not established in this pass or in the recorded metadata being used.

The Elsevier bibliography style may emit empty pages warnings for these entries. Such a warning is not a missing citation key or a compilation error. The primary URL, venue, year, and available volume identify the work. The supplied class and bibliography style are left unchanged.

## Scope

This record concerns bibliographic metadata and the evidence for fields added to `references.bib`. It does not mean that every cited source was read in full during this pass. Reference accuracy and precise attribution are the criteria used here.

## Printed reference presentation

The final presentation uses conventional publication metadata and verified DOIs. Redundant publisher and proceedings URLs are retained in bibliography_links.json rather than repeated in the printed reference list. Short document links remain for preprints and unpaginated ICLR papers. This formatting change preserves all citation keys, titles, authors, years and verified publication details.

