"""Source row counts and report-container diagnostics; aggregate output only."""
import ast
import json
import os
from pathlib import Path
import re
import duckdb
import pandas as pd
from .acquisition import NEEDED
from .inspection import parse_paths
from ..report_audit import SECTIONS,normalize_report,HEADER

P=Path('/projects/icumultimodal')


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Source verification requires a CPU allocation')
    output=P/'results/data_quality'
    output.mkdir(exist_ok=True)
    con=duckdb.connect()
    con.execute('SET threads=8')
    counts=[]
    for relative in NEEDED:
        path=P/'raw/mimiciv'/relative
        marker=output/(relative.replace('/','__')+'.json')
        if marker.exists():
            record=json.loads(marker.read_text())
        else:
            count=con.execute(f"SELECT count(*) FROM read_csv('{path.as_posix()}',header=true,all_varchar=true)").fetchone()[0]
            record={'table':relative,'rows':count,'bytes':path.stat().st_size}
            marker.write_text(json.dumps(record,indent=2))
        counts.append(record)
        print('SOURCE_ROWS_VERIFIED',relative,record['rows'],flush=True)
    cohort=pd.read_parquet(P/'data/processed/main/cohort.parquet',columns=['subject_id'])
    manifest=json.loads((P/'results/acquisition/cxrmetadata.json').read_text())
    data=pd.concat([pd.read_csv(P/f'raw/mimiccxr/metadata{i}.csv',usecols=['subject_id','text','image']) for i in range(len(manifest))])
    data=data.set_index('subject_id').loc[cohort.subject_id]
    counters={'multiple_text_entries':0,'multiple_distinct_texts':0,'multiple_studies':0,
              'text_count_matches_image_count':0,'text_count_matches_study_count':0}
    anywhere={section:0 for section in SECTIONS}
    anchored={section:0 for section in SECTIONS}
    any_header=re.compile(r'\b('+'|'.join(SECTIONS)+r')\s*:',re.I)
    for row in data.itertuples():
        parsed=ast.literal_eval(row.text)
        texts=parsed if isinstance(parsed,list) else [str(parsed)]
        images=parse_paths(row.image)
        studies={match.group(1) for path in images if (match:=re.search(r'/(s\d+)/',path.replace('\\','/')))}
        counters['multiple_text_entries']+=int(len(texts)>1)
        counters['multiple_distinct_texts']+=int(len(set(str(text).strip() for text in texts))>1)
        counters['multiple_studies']+=int(len(studies)>1)
        counters['text_count_matches_image_count']+=int(len(texts)==len(images))
        counters['text_count_matches_study_count']+=int(len(texts)==len(studies))
        normalized,_=normalize_report(row.text)
        any_names={match.group(1).upper() for match in any_header.finditer(normalized)}
        line_names={match.group(1).upper() for match in HEADER.finditer(normalized)}
        for section in SECTIONS:
            anywhere[section]+=int(section in any_names)
            anchored[section]+=int(section in line_names)
    report={'source_row_counts':counts,'patients':len(data),'report_structure_counts':counters,
            'known_headers_anywhere_patient_counts':anywhere,'known_headers_at_line_start_patient_counts':anchored}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print('SOURCE_QUALITY_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        reason=str(error) if isinstance(error,RuntimeError) else type(error).__name__
        (P/'reports/BLOCKED.md').write_text(f'Source verification stopped: {reason}. Inspect before continuing.\n')
        print('SOURCE_QUALITY_STOPPED',reason,flush=True)
        raise SystemExit(1)
