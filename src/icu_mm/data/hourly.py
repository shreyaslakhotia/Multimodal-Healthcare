"""Unit-independent hourly aggregation and tabular summaries.

Callers must verify units, map concepts and apply documented valid-value QC
before these functions. No fitted population statistic is used here.
"""
import numpy as np
import pandas as pd


def fahrenheit_to_celsius(values):
    return (np.asarray(values,dtype=float)-32.) * (5./9.)


def aggregate_hourly(events, stay_ids, channels):
    if len(set(stay_ids)) != len(stay_ids) or len(set(channels)) != len(channels):
        raise ValueError('Stay IDs and channel names must be unique')
    required = {'stay_id','hour','channel','value'}
    if not required.issubset(events.columns):
        raise ValueError('Missing standardized event fields')
    if not events.stay_id.isin(stay_ids).all() or not events.channel.isin(channels).all():
        raise ValueError('Events include an unexpected stay or channel')
    hour = events.hour.to_numpy(dtype=float)
    if not np.isfinite(hour).all() or np.any((hour<0)|(hour>=48)|(hour!=np.floor(hour))):
        raise ValueError('Expected integer hourly bins in [0,48)')
    valid = events.loc[np.isfinite(events.value)].copy()
    means = valid.groupby(['stay_id','hour','channel'],sort=False).value.mean().reset_index()
    tensor = np.full((len(stay_ids),48,len(channels)),np.nan,dtype=np.float32)
    row_lookup = {stay:i for i,stay in enumerate(stay_ids)}
    channel_lookup = {channel:i for i,channel in enumerate(channels)}
    row = means.stay_id.map(row_lookup).to_numpy(dtype=int)
    column = means.channel.map(channel_lookup).to_numpy(dtype=int)
    tensor[row,means.hour.to_numpy(dtype=int),column] = means.value.to_numpy()
    return tensor,np.isfinite(tensor)


def sum_gcs_components(component_events):
    if not set(component_events.component).issubset({'eye','verbal','motor'}):
        raise ValueError('Unexpected GCS component')
    pivot = component_events.pivot_table(index=['stay_id','hour'],columns='component',values='value',aggfunc='mean')
    pivot = pivot.reindex(columns=['eye','verbal','motor'])
    total = pivot.sum(axis=1,min_count=3).dropna().rename('value').reset_index()
    total['channel'] = 'gcs'
    return total[['stay_id','hour','channel','value']]


def summary_features(values, observed, channel_names, include_counts=True):
    values = np.asarray(values,dtype=float)
    observed = np.asarray(observed,dtype=bool)
    if values.ndim!=3 or values.shape[1]!=48 or values.shape!=observed.shape or values.shape[2]!=len(channel_names):
        raise ValueError('Expected [patients,48,channels] and aligned names/masks')
    valid = observed & np.isfinite(values)
    count = valid.sum(axis=1)
    mean = np.where(valid,values,0).sum(axis=1)/np.maximum(count,1)
    variance = np.where(valid,(values-mean[:,None,:])**2,0).sum(axis=1)/np.maximum(count,1)
    minimum = np.where(valid,values,np.inf).min(axis=1)
    maximum = np.where(valid,values,-np.inf).max(axis=1)
    first_index = valid.argmax(axis=1)
    last_index = 47-np.flip(valid,axis=1).argmax(axis=1)
    first = np.take_along_axis(values,first_index[:,None,:],axis=1)[:,0,:]
    last = np.take_along_axis(values,last_index[:,None,:],axis=1)[:,0,:]
    blocks = {'min':minimum,'max':maximum,'mean':mean,'std':np.sqrt(variance),'first':first,'last':last}
    result,names = [],[]
    for statistic,block in blocks.items():
        result.append(np.where(count>0,block,np.nan))
        names += [f'{channel}_{statistic}' for channel in channel_names]
    if include_counts:
        result.append(count.astype(float))
        names += [f'{channel}_observed_hours' for channel in channel_names]
    return np.concatenate(result,axis=1),names
