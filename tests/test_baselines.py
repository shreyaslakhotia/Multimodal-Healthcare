import numpy as np
import pandas as pd
from icu_mm import baselines
from icu_mm.splits import make_partitions


def test_oof_is_one_patient_each_and_restart_does_not_refit(tmp_path,monkeypatch):
    monkeypatch.setattr(baselines,'P',tmp_path)
    rng=np.random.default_rng(123)
    frame=pd.DataFrame({'subject_id':np.arange(200),'stay_id':np.arange(200)+1000,
                        'label':np.arange(200)%2,'age':rng.uniform(20,90,200),'sex':np.zeros(200),
                        'admission_type':['A']*200,'first_careunit':['UNIT']*200})
    folds,parts=make_partitions(frame.label,frame.subject_id)
    frame['fold']=folds
    directory=tmp_path/'data'
    directory.mkdir()
    import json
    np.save(directory/'summary_withoutcounts.npy',rng.normal(size=(200,3)))
    (directory/'summary_withoutcounts.json').write_text(json.dumps({'vitals':['hr_mean','rr_mean'],'labs':['glucose_mean']}))
    partition_path=directory/'partitions.npz'
    np.savez(partition_path,**{f'fold{fold}_{name}':indices for fold,part in enumerate(parts) for name,indices in part.items()})
    data={'frame':frame,'directory':directory,'parts':np.load(partition_path)}
    first=baselines.run_arm(data,'synthetic',('context','vitals','labs'),'logistic',False)
    oof=pd.read_parquet(next((tmp_path/'results/oof').glob('*.parquet')))
    assert len(oof)==200 and not oof.subject_id.duplicated().any()
    assert first['n']==200
    def forbidden_refit(*args,**kwargs):
        raise AssertionError('Restart refitted a completed fold')
    monkeypatch.setitem(baselines.LEARNERS,'logistic',forbidden_refit)
    second=baselines.run_arm(data,'synthetic',('context','vitals','labs'),'logistic',False)
    assert first==second
