"""Aggregate CXR path and selection layout; no timestamp metadata acquisition."""
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
from .inspection import parse_paths
from .cohort import select_image_paths,resolve_member

P=Path('/projects/icumultimodal')


def describe(values):
    return dict(zip(('min','q25','median','q75','max'),map(float,np.percentile(values,[0,25,50,75,100]))),mean=float(np.mean(values)),total=int(np.sum(values)))


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Layout inspection requires CPU allocation')
    manifest=json.loads((P/'results/acquisition/cxrmetadata.json').read_text())
    data=pd.concat([pd.read_csv(P/f'raw/mimiccxr/metadata{i}.csv',usecols=['subject_id','image','AP','PA']) for i in range(len(manifest))])
    subjects=set(pd.read_parquet(P/'data/processed/main/cohort.parquet',columns=['subject_id']).subject_id)
    with zipfile.ZipFile('/projects/icumultimodalarchive/zips/mimiccxr.zip') as archive:
        names=set(archive.namelist())
    by_basename=defaultdict(list)
    for name in names:
        if name.lower().endswith(('.jpg','.jpeg','.png')): by_basename[Path(name).name].append(name)
    reports={}
    for population,subset in [('all_metadata',data),('main_cohort',data.loc[data.subject_id.isin(subjects)])]:
        counter=defaultdict(int); image_counts=[]; unique_counts=[]; study_counts=[]; ap_counts=[]; pa_counts=[]
        for row in subset.itertuples(index=False):
            images=parse_paths(row.image); ap=parse_paths(row.AP); pa=parse_paths(row.PA)
            image_counts.append(len(images)); unique_counts.append(len(set(images)))
            studies={match.group(1) for path in images if (match:=re.search(r'/(s\d+)/',path.replace('\\','/')))}
            study_counts.append(len(studies)); ap_counts.append(len(ap)); pa_counts.append(len(pa))
            counter['image_paths']+=len(images)
            for path in images:
                counter['paths_with_study_id']+=int(bool(re.search(r'/(s\d+)/',path.replace('\\','/'))))
                counter['filenames_matching_dicom_id_shape']+=int(bool(re.fullmatch(r'[0-9a-f]{40}',Path(path.replace('\\','/')).stem.replace('-',''),flags=re.I)))
            counter['patients_with_ap']+=int(bool(ap)); counter['patients_with_pa']+=int(bool(pa))
            counter['patients_with_multiple_images']+=int(len(set(images))>1)
            counter['patients_with_multiple_studies']+=int(len(studies)>1)
            selected,view,alternatives=select_image_paths(row.AP,row.PA)
            if selected:
                counter['selected_ap' if view=='AP' else 'selected_pa']+=1
                counter['patients_with_additional_frontal_image']+=int(len(set(ap+pa)-{selected})>0)
                counter['patients_with_same_view_listed_alternative']+=int(len(set(alternatives)-{selected})>0)
                if view=='AP':
                    counter['ap_selected_patients_with_ap_listed_alternative']+=int(len(set(ap)-{selected})>0)
                if population=='main_cohort':
                    available={member for path in alternatives if (member:=resolve_member(path,names,by_basename)) is not None}
                    chosen=resolve_member(selected,names,by_basename)
                    counter['patients_with_same_view_available_alternative']+=int(bool(available-{chosen}))
                    if view=='AP': counter['ap_selected_patients_with_ap_available_alternative']+=int(bool(available-{chosen}))
        reports[population]={'patients':len(subset),'counts':dict(counter),'listed_images_per_patient':describe(image_counts),'unique_listed_images_per_patient':describe(unique_counts),'studies_per_patient':describe(study_counts),'ap_images_per_patient':describe(ap_counts),'pa_images_per_patient':describe(pa_counts)}
    output=P/'results/data_quality/cxr_layout.json'
    output.write_text(json.dumps({'populations':reports,'timing':'No timestamps; no PhysioNet metadata requested or obtained','dicom_id':'Filename structural match only; not independently joined to canonical metadata','availability':'ZIP central directory, not repeated image decoding'},indent=2))
    print('CXR_LAYOUT_COMPLETE',flush=True)


if __name__=='__main__': main()
