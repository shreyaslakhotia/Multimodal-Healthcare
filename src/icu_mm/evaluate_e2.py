"""Aggregate locked E2 predictions without refitting encoders or selecting seeds."""
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from .aggregate import align_patients
from .bootstrap import patient_intervals, paired_bootstrap
from .metrics import score, select_threshold

P = Path('/projects/icumultimodal')
MODALITIES = ('context', 'vitals', 'labs', 'cxr')


def average_seeds(frames):
    """Require identical patient, fold and label coverage before averaging."""
    ordered = [frame.sort_values('subject_id').reset_index(drop=True) for frame in frames]
    first = ordered[0].copy()
    for other in ordered[1:]:
        align_patients(first, other)
        if not first[['subject_id', 'fold', 'y_true']].equals(other[['subject_id', 'fold', 'y_true']]):
            raise ValueError('Seed rows differ')
    first['y_prob'] = np.mean([frame.y_prob.to_numpy() for frame in ordered], axis=0)
    return first


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Evaluation requires a CPU Slurm allocation')
    root = P/'results/deep/e2'
    completion = json.loads((root/'complete.json').read_text())
    if not completion['complete']:
        raise RuntimeError('E2 incomplete')
    out = P/'results/evaluation/e2'
    out.mkdir(parents=True, exist_ok=True)
    figures = P/'reports/figures/e2'
    figures.mkdir(parents=True, exist_ok=True)
    records, folds, comparisons = [], [], []
    predictions = {}
    table = pd.read_csv(P/'results/summary_table.csv')
    for modality in MODALITIES:
        seeds = []
        for seed in range(3):
            prediction = pd.read_parquet(P/f'results/oof/e2__{modality}__seed{seed}.parquet')
            seeds.append(prediction)
            records.append(dict(modality=modality, estimator=f'seed{seed}', **score(prediction.y_true, prediction.y_prob, prediction.threshold)))
            for fold in range(5):
                metrics = json.loads((root/modality/f'fold{fold}'/f'seed{seed}'/'metrics.json').read_text())
                folds.append(dict(modality=modality, fold=fold, seed=seed, **metrics))
        ensemble = average_seeds(seeds)
        # Average validation probabilities within the same fold, then select
        # ensemble thresholds there. Test probabilities never choose thresholds.
        for fold in range(5):
            validation = average_seeds([pd.read_parquet(root/modality/f'fold{fold}'/f'seed{seed}'/'validation.parquet') for seed in range(3)])
            chosen = ensemble.fold == fold
            for column, target in [('threshold', None), ('recall80_threshold', .8), ('recall90_threshold', .9)]:
                ensemble.loc[chosen, column] = select_threshold(validation.y_true, validation.y_prob, target)
        ensemble.to_parquet(out/f'{modality}_ensemble.parquet', index=False)
        predictions[modality] = ensemble
        record = dict(modality=modality, estimator='mean_probability_ensemble', **score(ensemble.y_true, ensemble.y_prob, ensemble.threshold))
        interval = patient_intervals(ensemble.y_true, ensemble.subject_id, ensemble.y_prob)
        for metric in ('auroc', 'auprc'):
            for bound in ('lower', 'upper'):
                record[f'{metric}_{bound}'] = interval[metric][bound]
        for target in (80, 90):
            operating = score(ensemble.y_true, ensemble.y_prob, ensemble[f'recall{target}_threshold'])
            for name in ('precision', 'recall', 'specificity'):
                record[f'validation_recall{target}_{name}'] = operating[name]
        records.append(record)
        baselines = table.loc[(table.arm == modality) & (table.selection == 'main')]
        for _, baseline in baselines.iterrows():
            first, second = ensemble, pd.read_parquet(P/f'results/oof/{baseline.experiment}__seed0.parquet')
            paired = align_patients(first, second)
            result = paired_bootstrap(paired.y_true_first, paired.subject_id, paired.y_prob_first, paired.y_prob_second)
            row = dict(modality=modality, baseline=baseline.experiment, baseline_auroc=float(baseline.pooled_auroc) if 'pooled_auroc' in baseline else None)
            for metric in ('auroc', 'auprc'):
                for key in ('delta', 'lower', 'upper'):
                    row[f'{metric}_{key}'] = result[metric][key]
            comparisons.append(row)
        print('E2_EVALUATED', modality, flush=True)
    summary = pd.DataFrame(records)
    summary.to_csv(out/'summary.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'fold_metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv', index=False)
    colors = ['#475569', '#2563eb', '#059669', '#d97706']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    caption = '6,225 patients, 705 deaths; 5 patient folds; 3 seeds averaged per patient. CXR timing unknown.'
    def save(fig, name, extra=''):
        fig.text(.5, .015, caption+'\n'+extra, ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .075, 1, 1))
        fig.savefig(figures/f'{name}.png', dpi=170, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for modality, color in zip(MODALITIES, colors):
        data = predictions[modality]
        fpr, tpr, _ = roc_curve(data.y_true, data.y_prob)
        precision, recall, _ = precision_recall_curve(data.y_true, data.y_prob)
        row = summary.loc[(summary.modality == modality) & (summary.estimator == 'mean_probability_ensemble')].iloc[0]
        axes[0].plot(fpr, tpr, color=color, label=f'{modality.title()} ({row.auroc:.3f})')
        axes[1].plot(recall, precision, color=color, label=f'{modality.title()} ({row.auprc:.3f})')
    axes[0].plot([0, 1], [0, 1], '--', color='#aaaaaa')
    axes[1].axhline(705/6225, ls='--', color='#aaaaaa', label='Prevalence 0.113')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curves and AUROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision recall curves and average precision')
    for ax in axes:
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    save(fig, 'roc_pr', 'Locked out-of-fold probabilities; curves are descriptive, without confidence bands.')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, metric in enumerate(('auroc', 'auprc')):
        for index, (modality, color) in enumerate(zip(MODALITIES, colors)):
            rows = summary.loc[summary.modality == modality]
            ensemble = rows.loc[rows.estimator == 'mean_probability_ensemble'].iloc[0]
            value = ensemble[metric]
            axes[k].errorbar(index, value, yerr=[[value-ensemble[f'{metric}_lower']], [ensemble[f'{metric}_upper']-value]], fmt='o', color=color, capsize=4)
            axes[k].scatter(np.array([index-.13, index, index+.13]), rows.iloc[:3][metric], marker='x', color=color, alpha=.65)
        axes[k].set_xticks(range(4), ['Context', 'Vitals', 'Labs', 'CXR'])
        axes[k].set_title('AUROC' if k == 0 else 'Average precision')
        axes[k].grid(axis='y', alpha=.2)
    save(fig, 'performance', 'Circles: ensemble with 95% patient bootstrap CI (1,000 draws); crosses: individual seed estimates.')
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, modality in zip(axes, MODALITIES):
        row = summary.loc[(summary.modality == modality) & (summary.estimator == 'mean_probability_ensemble')].iloc[0]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(matrix, cmap='Blues')
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center', color='white' if value > matrix.max()/2 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Survive', 'Die'], yticklabels=['Survive', 'Die'], xlabel='Predicted', title=modality.title())
    axes[0].set_ylabel('Actual')
    save(fig, 'confusion', 'Maximum F1 thresholds selected on each fold validation set after averaging seeds.')
    curves = []
    for line in (P/'reports/icue2-215538.out').read_text().splitlines():
        match = re.match(r'E2_EPOCH (\w+) fold (\d+) seed (\d+) epoch (\d+) validation_auroc ([\d.]+)', line)
        if match:
            m, fold, seed, epoch, auc = match.groups()
            curves.append(dict(modality=m, fold=int(fold), seed=int(seed), epoch=int(epoch)+1, validation_auroc=float(auc)))
    pd.DataFrame(curves).to_csv(out/'learning_curves.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, modality, color in zip(axes.flat, MODALITIES, colors):
        data = pd.DataFrame(curves)
        for _, case in data.loc[data.modality == modality].groupby(['fold', 'seed']):
            ax.plot(case.epoch, case.validation_auroc, color=color, alpha=.25, lw=.9)
        ax.set(title=modality.title(), xlabel='Epoch', ylabel='Validation AUROC')
        ax.grid(alpha=.2)
    save(fig, 'learning_curves', '15 fold and seed trajectories per modality; validation only; curves end at early stopping.')
    metadata = {'training': completion, 'patients': 6225, 'deaths': 705, 'folds': 5, 'seeds': [0, 1, 2],
                'bootstrap_draws': 1000, 'evaluation': 'three-seed mean probability; thresholds from matching averaged validation probabilities',
                'uncertainty': 'patient bootstrap conditional on trained models; does not include retraining or external-site variation'}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('E2_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
