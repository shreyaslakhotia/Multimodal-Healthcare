# E4 design amendment and primary comparison

Declared at the user's request after E2 acceptance, before any E4 scoring.
The primary RQ2 test is EHR-only cross-attention versus EHR-only concatenation.
Neither primary model receives CXR values, CXR presence indicators, CXR tokens or
an unused CXR encoder. The matched cohort remains fixed; this tests architecture
within that cohort, not all ICU patients. CXR timing remains unknown.

## Primary RQ2

Train EHR concatenation with context, vitals and labs, using their matching E2
fold/seed encoder initializers, masked pooling, concatenation 384 -> 128 -> 1,
ReLU, dropout 0.1 and joint fine-tuning. Compare against E4a EHR-only with the
same three encoders, two synchronous cross-attention blocks, width 128, four
heads, FFN width 256 and the same 384 -> 128 -> 1 classifier. Queries are each
stream's tokens; keys/values are the other two streams. This holds input set,
initialization, classifier width and training settings fixed while changing
fusion. Parameter counts must be reported; cross-attention has extra parameters.

Use five fixed folds and seeds 0,1,2. Primary delta: E4a EHR minus EHR joint
concatenation. Report AUROC/AP, paired conditional patient intervals, all five
outer-fold deltas and sign counts, and pair-specific approximate resolution.
An interval spanning zero or a difference below that resolution is unresolved,
not equal. A null comparison is a valid result and must be reported plainly.
No outcome-based redesign follows an unresolved result.

## Secondary CXR arm and E4b

Run the four-stream version as a secondary, explicitly untimed-CXR arm:
context 1 x 128, vitals/labs 48 x 128 each, CXR 49 x 128. The four pooled
vectors form 512 -> 128 -> 1. Its matched comparator is E3 joint with CXR.
CXR currently has no demonstrated incremental E1 benefit; this is not evidence
that the modality can never help. Do not obtain PhysioNet metadata; the user
will discuss that with supervisors. Finish the CXR layout report before E4.

E4a has no modality dropout. E4b independently drops each active non-context
stream with probability 0.15 per sample. Thus EHR-only E4b drops vitals/labs;
the full E4b also drops CXR. Context is never randomly dropped. Preserve both
E4a/E4b, label the primary comparison in advance, and report every result.

Self-attention within each E2 temporal encoder still sees imputed tokens;
unobserved hours are excluded from pooling. In cross-attention, invalid hourly
keys and entirely missing streams are masked. Missing query streams are replaced
by a learned pooled stand-in. A zero null key prevents all-masked softmax NaNs.
Hidden modalities are zeroed before encoding so NaNs/values cannot leak through.

Attention mass is an aggregate descriptive diagnostic only: average over valid
queries and available patients, separately by block/source/target; no self-target
attention. It is not modality importance. Importance is evaluated below.

## Preserve E5

- Retrain E4a leaving out each modality: context/vitals/labs for the primary EHR
  setting, and each of all four modalities for the secondary setting. Each
  retraining uses the same outer partitions and three seeds; keep the input-set
  changes explicit. Do not substitute test masking for retraining.
- Mask each individual modality at test time and report AUROC/AP/precision/recall
  changes with frozen validation thresholds, paired by patient and outer fold.
- Permute each modality across test patients with a fixed declared permutation.
- CXR missingness curves at 0,25,50,75,100 percent for full E3 joint, E4a and E4b.
  Use identical patient subsets and missingness draws across models, fixed
  before looking at scores. EHR-only models are unaffected reference lines.
- Age/sex subgroup summaries remain lower priority as in the brief.

No logistic stacking. Optional MedFuse-style LSTM remains dropped/lower priority.
Training retains batch 64, AdamW 1e-4/weight decay .01, max 50 epochs, patience
10 on inner-validation AUROC, bf16 and gradient clipping 1. No test-based model,
epoch, seed, threshold or architecture selection. Preserve the mask warning.
