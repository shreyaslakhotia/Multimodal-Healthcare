"""Aggregate-only source discovery. No patient examples are exported."""
import ast
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd

P = Path('/projects/icumultimodal')
A = Path('/projects/icumultimodalarchive')
VITAL_IDS = [220045, 220210, 224690, 220277, 220050, 220179, 220051, 220180,
             220052, 220181, 223762, 223761, 220739, 223900, 223901]
LAB_IDS = [50931, 50809, 50820, 51222, 50811, 51301, 51300, 51265,
           50912, 51006, 50983, 50824, 50971, 50822, 50882, 50803, 50813]


def parse_paths(value):
    """Accept a string or literal list/tuple of strings; never evaluate code."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not isinstance(value, str):
        raise ValueError('Unsupported CXR path-column value type')
    value = value.strip()
    if not value or value.lower() in ('nan', 'none', '[]'):
        return []
    if value.startswith(('[', '(')):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)) or not all(isinstance(item, str) for item in parsed):
            raise ValueError('Unsupported CXR path-list structure')
        return list(parsed)
    return [value]


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Discovery requires a CPU Slurm allocation')
    output = P / 'results' / 'discovery'
    output.mkdir(parents=True, exist_ok=True)
    item_rows = []
    for name, ids in [('icu/d_items.csv', VITAL_IDS), ('hosp/d_labitems.csv', LAB_IDS)]:
        path = P / 'raw' / 'mimiciv' / name
        header = pd.read_csv(path, nrows=0).columns
        columns = [column for column in ('itemid', 'label', 'unitname', 'fluid', 'category') if column in header]
        frame = pd.read_csv(path, usecols=columns)
        selected = frame.loc[frame.itemid.isin(ids)].copy()
        if set(selected.itemid) != set(ids):
            raise RuntimeError('Required item IDs missing from source dictionary')
        item_rows.append({'table': name, 'items': json.loads(selected.to_json(orient='records'))})
    # Dictionaries contain schema definitions, not patient records.
    (output / 'itemdictionaries.json').write_text(json.dumps(item_rows, indent=2))
    metadata_manifest = json.loads((P / 'results' / 'acquisition' / 'cxrmetadata.json').read_text())
    records = []
    all_subjects, sampled_paths = set(), []
    rng = np.random.default_rng(42)
    for index in range(len(metadata_manifest)):
        path = P / 'raw' / 'mimiccxr' / f'metadata{index}.csv'
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if not {'subject_id', 'AP', 'PA'}.issubset(columns):
            raise RuntimeError('CXR metadata layout lacks required subject/AP/PA fields')
        usecols = [column for column in ('subject_id', 'AP', 'PA', 'Lateral', 'image', 'view') if column in columns]
        frame = pd.read_csv(path, usecols=usecols)
        if frame.subject_id.isna().any():
            raise RuntimeError('Missing CXR patient identifiers')
        ids = frame.subject_id.astype('int64')
        all_subjects.update(ids.tolist())
        counts, path_types = {}, {}
        for column in ('AP', 'PA', 'Lateral', 'image'):
            if column not in frame:
                continue
            # Column structure is summarized, never raw values or patient paths.
            parsed = frame[column].map(parse_paths)
            lengths = parsed.map(len)
            counts[column] = {'rows_with_paths': int((lengths > 0).sum()),
                              'listed_paths': int(lengths.sum()), 'max_per_row': int(lengths.max())}
            path_types[column] = dict(Counter('list' if isinstance(value, str) and value.strip().startswith(('[', '(')) else 'string_or_empty' for value in frame[column]))
            if column in ('AP', 'PA'):
                available = [paths[0] for paths in parsed if paths]
                if available:
                    indices = rng.choice(len(available), min(100, len(available)), replace=False)
                    sampled_paths.extend(available[i] for i in indices)
        # Inspect only 20 rows of text per CSV, in memory on compute. Export
        # structural comparisons and lengths, never original or augmented text.
        text_columns = [column for column in ('text', 'text_augment') if column in columns]
        text_summary = {}
        if text_columns:
            text = pd.read_csv(path, usecols=text_columns, nrows=20).fillna('')
            for column in text_columns:
                lengths = text[column].astype(str).str.len()
                text_summary[column] = {'inspected_rows': len(text), 'mean_characters': float(lengths.mean())}
            if len(text_columns) == 2:
                text_summary['identical_fraction'] = float((text['text'] == text['text_augment']).mean())
                text_summary['provenance'] = 'Augmentation mechanism cannot be inferred reliably from string comparison alone'
        records.append({'metadata_index': index, 'columns': columns, 'rows': len(frame),
                        'unique_subjects': int(ids.nunique()), 'duplicate_subject_rows': int(ids.duplicated().sum()),
                        'path_counts': counts, 'path_structures': path_types,
                        'possible_time_columns': [column for column in columns if re.search(r'date|time', column, re.I)],
                        'text_structure': text_summary})
    sample = sampled_paths
    if len(sample) > 200:
        sample = [sample[i] for i in rng.choice(len(sample), 200, replace=False)]
    with zipfile.ZipFile(A / 'zips' / 'mimiccxr.zip') as archive:
        names = set(archive.namelist())
        basenames = Counter(Path(name).name for name in names)
        by_basename = defaultdict(list)
        for name in names:
            by_basename[Path(name).name].append(name)
        exact = suffix = unique_basename = 0
        for path in sample:
            normalized = path.replace('\\', '/').lstrip('/')
            if normalized in names:
                exact += 1
            elif any(normalized.endswith('/' + name) for name in by_basename[Path(normalized).name]):
                suffix += 1
            elif basenames[Path(normalized).name] == 1:
                unique_basename += 1
        embedded_ids = sum(bool(re.search(r'p\d+/p\d+/s\d+/', path)) for path in sample)
    report = {'metadata': records, 'unique_subjects_across_metadata': len(all_subjects),
              'path_sample_size': len(sample), 'exact_archive_matches': exact,
              'suffix_archive_matches': suffix, 'unique_basename_matches': unique_basename,
              'sample_paths_with_original_study_structure': embedded_ids,
              'variant': 'A_UNTIMED_UNLESS_DISCOVERY_ESTABLISHES_OTHERWISE',
              'caveat': 'Archive member existence is not image decoding; selected images must still decode before cohort inclusion.'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    print('DISCOVERY_COMPLETE', 'metadata_tables', len(records), 'unique_subjects', len(all_subjects), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        reason = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        (P / 'reports' / 'BLOCKED.md').write_text(f'Discovery stopped: {reason}. Inspect locally.\n')
        print('DISCOVERY_STOPPED', reason, flush=True)
        raise SystemExit(1)
