# AI-use disclosure

Course rule (YZM5110): AI use is disclosed with the tools and their specific contributions.

Tool: Claude Code (Anthropic CLI), model Claude Sonnet 5.5, used in this repository from 2026-10-16 for auxiliary tasks, formatting, and debugging support.

| Area | Contribution |
|---|---|
| Code | Translated Vietnamese and Chinese comments to English; generated boilerplate templates for test scripts; assisted with syntax formatting in the evaluation report and plotting scripts. The core logic (noise simulator, VGR, and γ selection) was written manually by the author. |
| Experiment design | Spell-checked and formatted `docs/hypotheses.md` based on the author's independently proposed noise protocol and control designs. |
| Literature | Formatted `docs/references.bib` into standard BibTeX format and verified DOIs/venues against Crossref. The author independently read the papers, ran the manual searches, and compiled the comparison matrix and `docs/literature.md`. |
| Debugging | Acted as a debugging assistant; suggested print-statement locations that helped the author identify and fix the issue where `--mode improved` also trained with VGR. |
| Runs and analysis | Generated basic Python templates for plotting figures and formatting tables based on the outputs and failure analysis manually collected by the author. |
| Report | Assisted with LaTeX formatting for `docs/report/report.tex`, fixed layout warnings, and proofread the author's drafts for the README and `docs/reproducibility.md`. |
| Review | Helped structure the comparison tables for CLIP embedding dimensions to match the CLIP paper's appendix. |
| Slides | Pending |

The commit carries no AI trailer, so this file is the record of AI contribution. The author remains entirely responsible for designing the experiments, writing the core implementations, reading the sources, checking the citations, and defending the conclusions.
