import pandas as pd
from icu_mm.data.features import clean_channels,clean_gcs


def test_temperature_qc_converts_before_applying_celsius_range():
    events = pd.DataFrame({'stay_id':[1,1],'hour':[0,1],'itemid':[223761,223762],
                           'valueuom':['°F','°C'],'valuenum':[98.6,100.],'value':['98.6','100']})
    cleaned,report = clean_channels(events,{'temperature':{'ids':[223761,223762],'units':['°F','°C'],'range':[20,45]}})
    assert len(cleaned)==1 and abs(cleaned.iloc[0].value-37)<1e-6
    assert report['temperature']['out_of_range']==1


def test_ett_is_missing_despite_numeric_verbal_score():
    events = pd.DataFrame({'stay_id':[1,1,1],'hour':[0,0,0],'itemid':[220739,223900,223901],
                           'valueuom':[None]*3,'valuenum':[4.,1.,6.],'value':['4 Spontaneously','No Response-ETT','6 Obeys Commands']})
    spec={'eye':{'id':220739,'range':[1,4]},'verbal':{'id':223900,'range':[1,5]},
          'motor':{'id':223901,'range':[1,6]},'text_score_pattern':r'^\s*([1-6])(?:\s|$)'}
    cleaned,report=clean_gcs(events,spec)
    assert cleaned.empty
    assert report['verbal']['ett_unassessable']==1
