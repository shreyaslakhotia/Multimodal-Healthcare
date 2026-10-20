"""Finalize valid-value tensors and persist immutable patient partitions."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .hourly import aggregate_hourly,summary_features,sum_gcs_components,fahrenheit_to_celsius
from ..splits import make_partitions
from ..guards import assert_feature_names,assert_landmark_cohort

P = Path('/projects/icumultimodal')


def clean_channels(events, specifications):
    chunks,counts = [],{}
    for channel,spec in specifications.items():
        subset = events.loc[events.itemid.isin(spec['ids'])].copy()
        if not subset.valueuom.isin(spec['units']).all():
            raise RuntimeError('Unexpected unit in configured channel: '+channel)
        value = subset.valuenum.to_numpy(dtype=float)
        if channel=='temperature':
            fahrenheit = subset.itemid.to_numpy()==223761
            if not (subset.loc[fahrenheit,'valueuom']=='°F').all() or not (subset.loc[~fahrenheit,'valueuom']=='°C').all():
                raise RuntimeError('Temperature item/unit disagreement')
            value[fahrenheit] = fahrenheit_to_celsius(value[fahrenheit])
        numeric = np.isfinite(value)
        valid = numeric & (value>=spec['range'][0]) & (value<=spec['range'][1])
        counts[channel] = {'events':len(subset),'non_numeric':int((~numeric).sum()),
                           'out_of_range':int((numeric&~valid).sum()),'retained':int(valid.sum())}
        subset['value'] = value
        subset['channel'] = channel
        chunks.append(subset.loc[valid,['stay_id','hour','channel','value']])
    return pd.concat(chunks,ignore_index=True),counts


def clean_gcs(events, specification):
    chunks,counts = [],{}
    for component in ('eye','verbal','motor'):
        spec = specification[component]
        subset = events.loc[events.itemid==spec['id']].copy()
        if subset.valueuom.notna().any():
            raise RuntimeError('Unexpected unit on GCS component')
        original = subset.value.fillna('').astype(str)
        ett = original.str.contains('ETT',case=False,regex=False).to_numpy()
        value = subset.valuenum.to_numpy(dtype=float)
        fallback = pd.to_numeric(original.str.extract(specification['text_score_pattern'],expand=False),errors='coerce').to_numpy(dtype=float)
        missing = ~np.isfinite(value)
        value[missing] = fallback[missing]
        valid = ~ett & np.isfinite(value) & (value>=spec['range'][0]) & (value<=spec['range'][1])
        counts[component] = {'events':len(subset),'ett_unassessable':int(ett.sum()),
                             'text_fallback_used':int((missing&valid).sum()),'retained':int(valid.sum()),
                             'other_invalid':int((~valid&~ett).sum())}
        subset['value'] = value
        subset['component'] = component
        chunks.append(subset.loc[valid,['stay_id','hour','component','value']])
    total = sum_gcs_components(pd.concat(chunks,ignore_index=True))
    if not total.value.between(3,15).all():
        raise RuntimeError('Calculated GCS total outside [3,15]')
    return total,counts


def save_dataset(directory,cohort,vitals,vitals_mask,labs,labs_mask,vital_names,lab_names):
    directory.mkdir(parents=True,exist_ok=True)
    fold,parts = make_partitions(cohort.label.to_numpy(),cohort.subject_id.to_numpy())
    cohort = cohort.copy()
    cohort['fold'] = fold
    cohort.to_parquet(directory/'cohort.parquet',index=False)
    np.savez(directory/'partitions.npz',**{f'fold{fold}_{name}':indices for fold,part in enumerate(parts) for name,indices in part.items()})
    for name,values in [('vitals',vitals),('vitals_mask',vitals_mask),('labs',labs),('labs_mask',labs_mask)]:
        np.save(directory/f'{name}.npy',values,allow_pickle=False)
    for include_counts in (False,True):
        vital_tab,vital_columns = summary_features(vitals,vitals_mask,vital_names,include_counts)
        lab_tab,lab_columns = summary_features(labs,labs_mask,lab_names,include_counts)
        names = vital_columns+lab_columns
        assert_feature_names(names)
        suffix = 'withcounts' if include_counts else 'withoutcounts'
        np.save(directory/f'summary_{suffix}.npy',np.concatenate([vital_tab,lab_tab],axis=1),allow_pickle=False)
        (directory/f'summary_{suffix}.json').write_text(json.dumps({'vitals':vital_columns,'labs':lab_columns}))
    positives = [int(cohort.loc[cohort.fold==fold,'label'].sum()) for fold in range(5)]
    return {'stays':len(cohort),'positives':int(cohort.label.sum()),'prevalence':float(cohort.label.mean()),
            'fold_positives':positives,'low_positive_fold_warning':any(count<40 for count in positives),
            'vital_observed_hour_rates':dict(zip(vital_names,vitals_mask.mean(axis=(0,1)).tolist())),
            'lab_observed_hour_rates':dict(zip(lab_names,labs_mask.mean(axis=(0,1)).tolist()))}


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Feature build requires a CPU Slurm allocation')
    raw = P/'data/interim'
    output = P/'data/processed'
    output.mkdir(parents=True,exist_ok=True)
    configuration = Path('configs/features.yaml').read_bytes()
    config_hash = hashlib.sha256(configuration).hexdigest()
    report_path = P/'results/cohort/features.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get('complete'):
            if previous.get('config_hash')!=config_hash:
                raise RuntimeError('QC configuration changed after final folds were persisted')
            print('FEATURES_ALREADY_COMPLETE',flush=True)
            return
    config = yaml.safe_load(configuration)
    s0 = pd.read_parquet(raw/'cohort_s0_candidates.parquet')
    candidates = pd.read_parquet(raw/'cohort_candidates.parquet')
    vitals_raw = pd.read_parquet(raw/'vitals_events.parquet')
    labs_raw = pd.read_parquet(raw/'labs_events.parquet')
    vitals,vital_qc = clean_channels(vitals_raw,config['vitals'])
    gcs,gcs_qc = clean_gcs(vitals_raw,config['gcs'])
    vitals = pd.concat([vitals,gcs],ignore_index=True)
    labs,lab_qc = clean_channels(labs_raw,config['labs'])
    # Subsets include a few candidates removed by the initial HR/lab criterion.
    vitals = vitals.loc[vitals.stay_id.isin(s0.stay_id)]
    labs = labs.loc[labs.stay_id.isin(s0.stay_id)]
    vital_names = list(config['vitals'])+['gcs']
    lab_names = list(config['labs'])
    v,vm = aggregate_hourly(vitals,s0.stay_id.tolist(),vital_names)
    l,lm = aggregate_hourly(labs,s0.stay_id.tolist(),lab_names)
    eligible = vm[:,:,0].any(axis=1) & lm.any(axis=(1,2))
    removed = int((~eligible).sum())
    s0 = s0.loc[eligible].reset_index(drop=True)
    v,vm,l,lm = v[eligible],vm[eligible],l[eligible],lm[eligible]
    main_mask = s0.stay_id.isin(candidates.stay_id).to_numpy()
    cohort = s0.loc[main_mask].reset_index(drop=True)
    if len(cohort)<2000 or cohort.label.sum()<150:
        raise RuntimeError('Final valid-value cohort below minimum support')
    audit = pd.read_parquet(raw/'cohort_audit.parquet').set_index('stay_id').loc[cohort.stay_id]
    assert_landmark_cohort((audit.outtime-audit.intime).dt.total_seconds().to_numpy()/3600,
                          (audit.deathtime-audit.intime).dt.total_seconds().to_numpy()/3600)
    main_report = save_dataset(output/'main',cohort,v[main_mask],vm[main_mask],l[main_mask],lm[main_mask],vital_names,lab_names)
    s0_report = save_dataset(output/'s0',s0,v,vm,l,lm,vital_names,lab_names)
    report = {'complete':True,'config_hash':config_hash,'main':main_report,'s0':s0_report,
              's0_rows_removed_by_valid_value_qc':removed,'vital_qc':vital_qc,'lab_qc':lab_qc,'gcs_qc':gcs_qc,
              'gcs_ett_policy':'Unassessable ETT verbal responses excluded, not assigned a score of one',
              'normalization':'None globally; train-only preprocessing is fitted within each partition',
              's0_folds':'Independent seed-42 grouped stratified folds for the demonstration cohort only'}
    report_path.write_text(json.dumps(report,indent=2))
    print('FEATURES_AND_FOLDS_READY','stays',len(cohort),'positives',int(cohort.label.sum()),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        reason = str(error) if isinstance(error,RuntimeError) else type(error).__name__
        (P/'reports/BLOCKED.md').write_text(f'Feature build stopped: {reason}. No automatic retries.\n')
        print('FEATURES_STOPPED',reason,flush=True)
        raise SystemExit(1)
