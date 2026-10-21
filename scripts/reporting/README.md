# Aggregate document builders

These local utilities consume copied aggregate CSVs/PNGs only. They are separate
from training requirements and never read patient-level artifacts. Run from a
writable directory with `python-docx`, `reportlab`, `pypdf` and pandas
available in the selected document runtime.

## E2 evaluation

Copy the aggregate CSVs and four figures from `results/evaluation/e2` and
`reports/figures/e2` into one local input directory. Then:

```bash
python scripts/reporting/build_e2_report.py --input-dir <aggregates> --output-dir <task-root>/outputs
python scripts/reporting/render_e2_pdf.py --root <task-root>
```

The renderer reads `outputs/E2Evaluation.docx` as a content intermediate and
creates `outputs/E2Evaluation.pdf`. It currently uses Windows Calibri fonts;
adapt font paths explicitly on another platform. DOCX generation alone does
not verify Word layout. The delivered PDF was rendered with Poppler and all
eight pages inspected. Do not publish an unrendered intermediate as validated.

## Requested-controls reports

```bash
python scripts/reporting/build_controls_report.py --input-dir <aggregates> --output-dir <reports>
```

Inputs: CXR layout JSON, E1 three-seed summary, E2 paired comparison CSVs and
optional prior-logistic fold comparisons. Use records matching the intended
evaluation revision; inspect generated reports before replacing published ones.
The experiment IDs in these builders describe the accepted research checkpoint,
not a general-purpose reporting framework.
