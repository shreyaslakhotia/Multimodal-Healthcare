"""Build candidate landmark/S0 cohorts and compact window-restricted events.

This is not final model input construction. Feature QC must recheck observation
eligibility before assigning and persisting final folds. All patient artifacts
remain on SSD and no patient rows or image paths are printed.
"""
import io
import hashlib
import json
import os
import traceback
from pathlib import Path
import zipfile
import duckdb
import numpy as np
import pandas as pd
from PIL import Image
from .inspection import parse_paths, VITAL_IDS, LAB_IDS

P = Path('/projects/icumultimodal')
A = Path('/projects/icumultimodalarchive')


def select_image_paths(ap, pa):
    ap, pa = parse_paths(ap), parse_paths(pa)
    if ap:
        return ap[0], 'AP', ap
    if pa:
        return pa[0], 'PA', pa
    return None, None, []


def resolve_member(path, names, by_basename):
    normalized = path.replace('\\', '/').lstrip('/')
    if normalized in names:
        return normalized
    candidates = by_basename.get(Path(normalized).name, [])
    suffix = [name for name in candidates if normalized.endswith('/' + name)]
    if len(suffix) == 1:
        return suffix[0]
    return candidates[0] if len(candidates) == 1 else None


def landmark_eligibility(frame):
    end = frame.intime + pd.Timedelta(hours=48)
    inconsistent = ((frame.label == 1) & frame.deathtime.isna()) | ((frame.label == 0) & frame.deathtime.notna())
    return (frame.outtime >= end) & (frame.deathtime.isna() | (frame.deathtime > end)) & ~inconsistent


def count_stage(name, frame):
    return {'stage': name, 'stays': len(frame), 'patients': int(frame.subject_id.nunique()),
            'positives': int(frame.label.sum()), 'prevalence': float(frame.label.mean()) if len(frame) else None}


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Cohort build requires a CPU Slurm allocation')
    output = P / 'data' / 'interim'
    output.mkdir(parents=True, exist_ok=True)
    reports = P / 'results' / 'cohort'
    reports.mkdir(parents=True, exist_ok=True)
    if (reports/'summary.json').exists():
        previous = json.loads((reports/'summary.json').read_text())
        if previous.get('complete') and all((output/name).exists() for name in ('cohort_candidates.parquet','cohort_s0_candidates.parquet','vitals_events.parquet','labs_events.parquet')):
            print('COHORT_CANDIDATES_ALREADY_COMPLETE', flush=True)
            return
    root = P / 'raw' / 'mimiciv'
    icu = pd.read_csv(root / 'icu/icustays.csv', usecols=['subject_id','hadm_id','stay_id','intime','outtime','first_careunit'], parse_dates=['intime','outtime'])
    admissions = pd.read_csv(root / 'hosp/admissions.csv', usecols=['subject_id','hadm_id','hospital_expire_flag','deathtime','admission_type'], parse_dates=['deathtime'])
    admissions = admissions.rename(columns={'subject_id':'admission_subject_id','hospital_expire_flag':'label'})
    patients = pd.read_csv(root / 'hosp/patients.csv', usecols=['subject_id','gender','anchor_age','anchor_year','dod'], parse_dates=['dod'])
    frame = icu.merge(admissions, on='hadm_id', how='inner', validate='many_to_one').merge(patients, on='subject_id', how='inner', validate='many_to_one')
    if not (frame.subject_id == frame.admission_subject_id).all():
        raise RuntimeError('Patient/admission linkage mismatch')
    if not set(frame.label.unique()).issubset({0,1}):
        raise RuntimeError('Unexpected mortality labels')
    if frame[['intime','outtime']].isna().any().any():
        raise RuntimeError('Missing ICU stay timestamps')
    funnel = [count_stage('All linked ICU stays', frame)]
    manifest = json.loads((P / 'results/acquisition/cxrmetadata.json').read_text())
    metadata = pd.concat([pd.read_csv(P / f'raw/mimiccxr/metadata{i}.csv', usecols=['subject_id','AP','PA']) for i in range(len(manifest))], ignore_index=True)
    if metadata.subject_id.duplicated().any():
        raise RuntimeError('CXR metadata has repeated patient rows across files')
    frame = frame.loc[frame.subject_id.isin(metadata.subject_id)].copy()
    funnel.append(count_stage('Patient present in CXR metadata', frame))
    frame = frame.sort_values(['subject_id','intime','stay_id']).drop_duplicates('subject_id', keep='first').copy()
    funnel.append(count_stage('First ICU stay per patient', frame))
    frame['age'] = frame.anchor_age + frame.intime.dt.year - frame.anchor_year
    frame['sex'] = frame.gender.map({'F':0, 'M':1})
    if frame.sex.isna().any():
        raise RuntimeError('Unrecognized sex category requires explicit encoding')
    frame = frame.loc[frame.age >= 18].copy()
    funnel.append(count_stage('Adults', frame))
    eligible = landmark_eligibility(frame)
    preimage_landmark = frame.loc[eligible].copy()
    funnel.append(count_stage('ICU >=48h and alive at landmark, consistent labels', preimage_landmark))
    inconsistencies = {'expired_flag_without_deathtime': int(((frame.label==1)&frame.deathtime.isna()).sum()),
                       'survival_flag_with_deathtime': int(((frame.label==0)&frame.deathtime.notna()).sum()),
                       'dod_day_disagrees_with_recorded_deathtime': int((frame.dod.notna() & frame.deathtime.notna() & (frame.dod.dt.normalize()!=frame.deathtime.dt.normalize())).sum())}
    candidates = frame.merge(metadata, on='subject_id', validate='one_to_one')
    selected, failure = [], {'no_frontal_paths':0, 'selected_path_missing_or_ambiguous':0, 'decode_failure':0}
    with zipfile.ZipFile(A / 'zips/mimiccxr.zip') as archive:
        names = set(archive.namelist())
        by_basename = {}
        for name in names:
            by_basename.setdefault(Path(name).name, []).append(name)
        for row in candidates.itertuples(index=False):
            path, view, alternatives = select_image_paths(row.AP, row.PA)
            if path is None:
                failure['no_frontal_paths'] += 1
                continue
            member = resolve_member(path, names, by_basename)
            if member is None:
                failure['selected_path_missing_or_ambiguous'] += 1
                continue
            try:
                with Image.open(io.BytesIO(archive.read(member))) as image:
                    image.load()
            except Exception:
                failure['decode_failure'] += 1
                continue
            # Deterministic last available image from the selected view. If no
            # different image exists, the main image is retained and counted.
            alternate = member
            for alternative in reversed(alternatives):
                possible = resolve_member(alternative, names, by_basename)
                if possible is not None:
                    try:
                        with Image.open(io.BytesIO(archive.read(possible))) as image:
                            image.load()
                        alternate = possible
                        break
                    except Exception:
                        continue
            selected.append({'subject_id':row.subject_id, 'selected_image':member, 'alternate_image':alternate, 'view':view})
    if not selected:
        raise RuntimeError('No selected images could be resolved and decoded')
    candidates = frame.merge(pd.DataFrame(selected), on='subject_id', validate='one_to_one')
    landmark_ids = set(preimage_landmark.stay_id)
    funnel.append(count_stage('Selected image exists and decodes', candidates.loc[candidates.stay_id.isin(landmark_ids)]))
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute(f"SET temp_directory='{os.environ['TMPDIR']}'")
    con.execute("SET memory_limit='16GB'")  # DuckDB spill limit, not a Slurm resource request.
    window = candidates[['subject_id','stay_id','intime']].copy()
    window_hash = hashlib.sha256(pd.util.hash_pandas_object(window,index=False).values.tobytes()).hexdigest()
    con.register('windows', window)
    chart = root / 'icu/chartevents.csv'
    labs = root / 'hosp/labevents.csv'
    for table, path, ids, join_key in [('vitals',chart,VITAL_IDS,'stay_id'),('labs',labs,LAB_IDS,'subject_id')]:
        destination = output / f'{table}_events.parquet'
        partial = destination.with_suffix('.parquet.part')
        checkpoint = reports/f'{table}_checkpoint.json'
        signature = {'window_hash':window_hash,'source_bytes':path.stat().st_size,'version':1}
        reusable = destination.exists() and checkpoint.exists() and json.loads(checkpoint.read_text())==signature
        sql = f"""
            COPY (
              WITH e AS (
                SELECT try_cast({join_key} AS BIGINT) AS join_id,
                       try_cast(charttime AS TIMESTAMP) AS charttime,
                       try_cast(itemid AS INTEGER) AS itemid,
                       try_cast(valuenum AS DOUBLE) AS valuenum,
                       value, valueuom
                FROM read_csv('{path.as_posix()}', header=true, all_varchar=true)
                WHERE try_cast(itemid AS INTEGER) IN ({','.join(map(str,ids))})
              )
              SELECT w.subject_id, w.stay_id, e.itemid, e.valuenum, e.value, e.valueuom,
                     floor(epoch(e.charttime-w.intime)/3600)::SMALLINT AS hour
              FROM e JOIN windows w ON e.join_id=w.{join_key}
                AND e.charttime>=w.intime AND e.charttime<w.intime+INTERVAL '48 hours'
            ) TO '{partial.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        if not reusable:
            if partial.exists():
                partial.unlink()
            con.execute(sql)
            partial.replace(destination)
            checkpoint.write_text(json.dumps(signature))
        units = con.execute(f"SELECT itemid, valueuom, count(*) AS events, count(valuenum) AS numeric_events FROM read_parquet('{destination.as_posix()}') GROUP BY itemid,valueuom ORDER BY itemid,events DESC").df()
        units.to_json(reports / f'{table}_unitcounts.json', orient='records', indent=2)
        print('WINDOWED_EVENT_SUBSET_READY', table, flush=True)
    vitals_path = (output/'vitals_events.parquet').as_posix()
    labs_path = (output/'labs_events.parquet').as_posix()
    hr_ids = set(con.execute(f"SELECT DISTINCT stay_id FROM read_parquet('{vitals_path}') WHERE itemid=220045 AND isfinite(valuenum) AND valuenum BETWEEN 0 AND 300").fetchnumpy()['stay_id'])
    lab_ids = set(con.execute(f"SELECT DISTINCT stay_id FROM read_parquet('{labs_path}') WHERE isfinite(valuenum)").fetchnumpy()['stay_id'])
    s0 = candidates.loc[candidates.stay_id.isin(hr_ids)].copy()
    funnel.append(count_stage('At least one valid heart-rate value', s0.loc[s0.stay_id.isin(landmark_ids)]))
    s0 = s0.loc[s0.stay_id.isin(lab_ids)].copy()
    final = s0.loc[s0.stay_id.isin(landmark_ids)].copy()
    funnel.append(count_stage('At least one numeric listed lab (before feature QC)', final))
    columns = ['stay_id','subject_id','hadm_id','intime','label','selected_image','alternate_image','view','age','sex','admission_type','first_careunit']
    final[columns].to_parquet(output/'cohort_candidates.parquet', index=False)
    s0[columns].to_parquet(output/'cohort_s0_candidates.parquet', index=False)
    # Outcome/time audit metadata is separate from every future model matrix.
    audit_columns = ['stay_id','intime','outtime','deathtime','label']
    s0[audit_columns].to_parquet(output/'cohort_audit.parquet', index=False)
    report = {'source_icustays':len(icu),'funnel':funnel, 's0':count_stage('S0 before feature QC',s0), 'image_failures':failure,
              'label_consistency':inconsistencies, 'alternate_differs':int((final.selected_image!=final.alternate_image).sum()),
              'status':'CANDIDATES_ONLY_FINAL_FOLDS_AFTER_FEATURE_QC', 'cxr_timing':'UNTIMED_VARIANT_A',
              'unit_review_required_before_features':True,'complete':False}
    (reports/'summary.json').write_text(json.dumps(report,indent=2))
    if len(final)<2000 or final.label.sum()<150:
        raise RuntimeError('Candidate cohort below required size or positive-count minimum')
    report['complete']=True
    (reports/'summary.json').write_text(json.dumps(report,indent=2))
    print('COHORT_CANDIDATES_READY', 'stays', len(final), 'positives', int(final.label.sum()), flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        reason = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        trace = traceback.extract_tb(error.__traceback__)
        location = ' > '.join(f'{Path(item.filename).name}:{item.lineno}:{item.name}' for item in trace)
        (P/'reports/BLOCKED.md').write_text(f'Cohort build stopped: {reason}. Location: {location}. No automatic retries.\n')
        print('COHORT_STOPPED', reason, 'LOCATION', location, flush=True)
        raise SystemExit(1)
