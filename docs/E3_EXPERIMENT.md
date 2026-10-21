# E3 fixed fusion experiment

Declared after the completed E2 evaluation and before scoring E3. Reuse the
persisted cohort, inner partitions, E2 preprocessing and same fold/seed encoders.
No architecture or hyperparameter search. All images remain untimed.

## Arms

- Late average: unweighted arithmetic mean of the four E2 probabilities within
  each fold and seed. Choose thresholds on the corresponding averaged validation
  probabilities; no fitted stacker. Retain three seed-specific OOF sets.
- Early: concatenate four masked pooled 128-dimensional E2 vectors and train a
  512-to-128-to-1 MLP with ReLU and dropout 0.1. Encoders stay in eval mode and
  receive no gradients. Cache deterministic vectors once per fold and seed.
- Joint: the same head, with the four E2 encoders initialized from the matching
  fold and seed and fine-tuned. The upstream CheXpert DenseNet remains frozen.

Early and joint each run 5 folds x 3 seeds, 30 fits total. AdamW learning rate
0.0001, weight decay 0.01, batch 64, maximum 50 epochs, patience 10 on validation
AUROC, class-weighted BCE using training counts only, bf16 and gradient clip 1.
The best validation checkpoint determines predictions. Maximum-F1 and recall
target thresholds use validation only. Save test/validation predictions and
per-epoch RNG/optimizer state. A signature protects encoder inputs and settings.
Do not refit on validation or select a seed by test performance.

No logistic stacker or optional LSTM is included in this allocation. Stacking
needs an explicit fitting/threshold-selection separation and is lower priority.
This preserves the protocol's priority of early and joint fusion.

## Evaluation and resources

Average the three seed probabilities per patient, and choose ensemble thresholds
from averaged validation predictions. Report seed scores, fold dispersion,
1,000-draw patient intervals, paired differences against late average, E1 EHR
and E1 all-modality models. Do not treat seed copies as separate patients.
Count-aware E1 references remain important because temporal E2 encoders use masks.

One supported x86 GPU, two-hour allocation, no CPU/memory GPU overrides. The current A6000 allocation has at most 360 SU; a
one-L40 allocation has at most 480 SU. E2 plus smoke used roughly 64.2 SU from
actual allocated seconds; confirm accounting before submission. Refuse SSD
usage >=90 GB. Checkpoint on every epoch; resume requires reviewing an interrupted
job. Stop on a nonfinite loss, test AUROC >0.98, signature mismatch or budget gate.
The 21,000 SU stage threshold remains in force.

## Reproduce

Submit only after E2 completion and CPU tests succeed, and do not duplicate an
active E3 job:

```bash
source /projects/icumultimodal/cacheconfig.sh
cd "$P/code"
sbatch --gres=gpu:a6000:1 scripts/slurm/train_fusion.sbatch
```

For the E2 PDF, aggregate on the cluster with scripts/slurm/evaluate_e2.sbatch,
then copy only its aggregate CSVs and PNGs off cluster. The local reporting
builder accepts --input-dir and --output-dir. render_e2_pdf.py accepts --root,
whose outputs/E2Evaluation.docx is its document-content intermediate. Windows
Calibri fonts and bundled python-docx/reportlab/PyPDF are used. Verify rendered
PDF pages before delivery; do not distribute an unrendered DOCX as validated.
