"""Interval-unioned temporal occupancy and EPG exposure, no metadata features."""
import argparse,json,time
import numpy as np
import pandas as pd
from .config import load_config
from .corrected_target_experiment import ROOT
from .context_experiment import _schedule_for
from .evaluate import load_prepared
from .fourth_pass import prepare_window,evaluate_variant
from .eighth_pass import internal,history
from .frequency_association import combine,cohorts
from .fifth_pass import _airing_rows
from .raw_context import RAW_CACHE
from .pipeline import sha256_file

def union_intervals(owner,start,stop):
    order=np.lexsort((stop,start,owner));u=owner[order];s=start[order];e=stop[order]
    if not len(u):return u,s,e
    cumulative=pd.Series(e).groupby(u,sort=False).cummax().to_numpy()
    previous=np.roll(cumulative,1);boundary=np.r_[True,u[1:]!=u[:-1]];previous[boundary]=s[boundary]
    s=np.maximum(s,previous);valid=e>s
    return u[valid],s[valid],e[valid]

def split_bins(owner,start,stop,nrows,weights=None):
    out=np.zeros((nrows,48));s=np.asarray(start,dtype=np.int64).copy();e=np.asarray(stop,dtype=np.int64)
    u=np.asarray(owner,dtype=np.int64);weights=np.ones(len(s)) if weights is None else weights
    while len(s):
        end=np.minimum(e,(s//3600+1)*3600)
        weekday=(s//86400+3)%7;bin=(s//3600)%24+24*(weekday>=5)
        out+=np.bincount(u*48+bin,weights=(end-s)*weights,minlength=nrows*48).reshape(nrows,48)
        active=end<e;s=end[active];e=e[active];u=u[active];weights=weights[active]
    return out

def profiles(raw,lo,hi,nusers):
    mask=(raw['start_ts']>=lo)&(raw['stop_ts']<=hi)&(raw['stop_ts']>raw['start_ts'])
    u,s,e=union_intervals(raw['uidx'][mask],raw['start_ts'][mask],raw['stop_ts'][mask])
    seconds=split_bins(u,s,e,nusers)
    exposure=split_bins(np.array([0]),np.array([lo]),np.array([hi]),1)[0]
    active=seconds.sum(axis=1)>0
    out={}
    for mode in (24,48):
        c=seconds if mode==48 else seconds[:,:24]+seconds[:,24:]
        ex=exposure if mode==48 else exposure[:24]+exposure[24:]
        prior=c[active].sum(axis=0)/(max(active.sum(),1)*np.maximum(ex,1))
        out[mode]=(c+7*3600*prior[None,:])/(ex[None,:]+7*3600)
        assert np.isfinite(out[mode]).all() and out[mode].min()>=0 and out[mode].max()<=1+1e-12
    return out

def schedule_bins(schedule,lo,hi,candidates):
    a=_airing_rows(schedule,pd.Timestamp(lo,unit='s'),pd.Timestamp(hi,unit='s'))
    ids=a.tv_show_id.to_numpy();pos=np.searchsorted(candidates,ids);valid=pos<len(candidates)
    valid &= candidates[np.clip(pos,0,len(candidates)-1)]==ids
    a=a.loc[valid];duration=a.duration.to_numpy(float)
    return split_bins(pos[valid],a.start.to_numpy(dtype='datetime64[s]').astype('int64'),a.end.to_numpy(dtype='datetime64[s]').astype('int64'),len(candidates),1/duration)

def main():
    cfg=load_config();p=load_prepared(cfg);raw=np.load(RAW_CACHE)
    windows=json.loads((ROOT/'corrected_plan.json').read_text())['selection_windows'];rows=[];details=[]
    for w in windows:
        begun=time.perf_counter();x=prepare_window(p,cfg,w);repeat,global_factor,mf,_=internal(p,w,x);count,age=history(x)
        baseline,ap=evaluate_variant(x,'submitted_confidence',combine(x,repeat,global_factor,mf,count,age));rows.append(baseline)
        lo,cut,hi=[pd.Timestamp(w[k]).value//10**9 for k in ('fit_start','cutoff','end')]
        profile=profiles(raw,lo,cut,len(p['user_ids']));uidx=np.searchsorted(p['user_ids'],x['users'])
        past=pd.concat([p['schedules']['export_arh_11-20-final.csv'],p['schedules']['export_arh_21-30-final.csv']],ignore_index=True)
        before=schedule_bins(past,lo,cut,x['candidates']);after=schedule_bins(_schedule_for(w,p),cut,hi,x['candidates'])
        for mode in (24,48):
            b=before if mode==48 else before[:,:24]+before[:,24:]
            a=after if mode==48 else after[:,:24]+after[:,24:]
            prob=profile[mode][uidx]
            br=np.minimum(prob@b.T/((cut-lo)/86400),2);ar=np.minimum(prob@a.T/((hi-cut)/86400),2)
            temporal=np.clip(np.sqrt((ar+.2)/(br+.2)),.5,1.5)
            for blend in (.25,.5):
                factor=(1-blend)*global_factor[None,:]+blend*temporal
                # combine expects a candidate-only factor: apply the personalized factor into raw first.
                result,aps=evaluate_variant(x,f'time{mode}_b{blend:g}',combine(x,repeat*factor,np.ones_like(global_factor),mf,count,age),ap)
                rows.append(result);details.append({'window':w['name'],'variant':result['variant'],'cohorts':cohorts(aps,ap,count,age)})
        elapsed=time.perf_counter()-begun
        if elapsed>180:raise TimeoutError('declared window budget exceeded')
        details.append({'window':w['name'],'seconds':elapsed});print(w['name'],'complete',flush=True)
    (ROOT/'temporal_selection.json').write_text(json.dumps({'plan_sha256':sha256_file(ROOT/'temporal_plan.json'),'rows':rows,'details':details},indent=2))
if __name__=='__main__':main()
