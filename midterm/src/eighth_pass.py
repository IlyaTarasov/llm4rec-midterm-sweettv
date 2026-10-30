"""Past-defined confidence cohorts and personalized schedule exposure ablations."""
import argparse,json
import numpy as np
import pandas as pd
from scipy import sparse
from .config import load_config,resolve
from .corrected_target_experiment import ROOT
from .evaluate import load_prepared,fuse_scores
from .fourth_pass import prepare_window,fused,evaluate_variant
from .fifth_pass import component_scores,fuse_component,_airing_rows
from .sixth_pass import factor
from .raw_context import RAW_CACHE
from .pipeline import sha256_file,cfg_with
from .models import MFModel

def internal(p,w,x):
    components,_=component_scores(p,w,x)
    raw=components['decayed_h56']/x['opportunity'][None,:]
    del components
    f=factor(p,w,x['candidates'],.2,.5)
    path=resolve(f'midterm/artifacts/seventh/{w["name"]}_mf_k16_a5_r0.05.npy')
    if path.exists(): mf=np.load(path)
    else:
        model=MFModel().fit(x['train'],cfg_with(load_config(),{'mf':{'factors':16,'alpha':5.,'reg':.05,'iterations':15,'weight':'count'}}),42)
        mf=model.score(np.searchsorted(p['user_ids'],x['users']),np.searchsorted(p['item_ids'],x['candidates']),x['train'])
        np.save(path,mf)
    m=mf.max(axis=1,keepdims=True);mf=mf/np.where(m>0,m,1)
    return raw,f,mf,fuse_component(x,raw*f[None,:])+.25*mf

def history(x):
    data=x['train'].interactions
    count=data.groupby('user_id')['view_count'].sum().reindex(x['users'],fill_value=0).to_numpy(float)
    last=data.groupby('user_id')['last_view'].max().reindex(x['users'])
    age=(pd.Timestamp(x['window']['cutoff'])-last).dt.total_seconds().to_numpy()/86400
    return count,np.where(np.isfinite(age),age,np.inf)

def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=('cohorts','confidence','channel','final'),required=True);args=a.parse_args()
    cfg=load_config();p=load_prepared(cfg);windows=json.loads((ROOT/'corrected_plan.json').read_text())['selection_windows'];rows=[]
    if args.phase=='final':
        decision=json.loads((ROOT/'eighth_final_decision.json').read_text())
        if decision['confidence_sha256']!=sha256_file(ROOT/'eighth_confidence.json') or decision['channel_sha256']!=sha256_file(ROOT/'eighth_channel.json'):raise RuntimeError('selection changed')
        windows=[json.loads((ROOT/'corrected_plan.json').read_text())['diagnostic_window']]
    for w in windows:
        x=prepare_window(p,cfg,w);base,bap=evaluate_variant(x,'fourth_submitted',fused(x,28.,2.,.25,opportunity=True))
        raw,f,mf,current=internal(p,w,x);control,cap=evaluate_variant(x,'internal_control',current,bap);rows.extend([base,control])
        count,age=history(x)
        if args.phase=='final':
            pop=fuse_scores({'popularity':x['pop']},{'popularity':1.},{'popularity':'global'})
            confidence=count/(count+5.)*np.exp(-np.minimum(age,1e9)/56.)
            result,_=evaluate_variant(x,'frozen_confidence_k5_tau56',pop+confidence[:,None]*(current-pop),bap);rows.append(result)
        if args.phase=='cohorts':
            for dimension,masks in [('count',{'0':count==0,'1-4':(count>0)&(count<5),'5-19':(count>=5)&(count<20),'20-99':(count>=20)&(count<100),'100+':count>=100}),('age',{'0-3':age<=3,'3-14':(age>3)&(age<=14),'14+':(age>14)&np.isfinite(age)})]:
                for label,mask in masks.items():
                    delta=cap[mask]-bap[mask]
                    rows.append({'window':w['name'],'dimension':dimension,'cohort':label,'n':int(mask.sum()),'fourth_ap':float(bap[mask].mean()) if mask.any() else None,'internal_ap':float(cap[mask].mean()) if mask.any() else None,'delta':float(delta.mean()) if mask.any() else None,'gained':int((delta>1e-12).sum()),'lost':int((delta< -1e-12).sum())})
        if args.phase=='confidence':
            pop=fuse_scores({'popularity':x['pop']},{'popularity':1.},{'popularity':'global'})
            for k in (5.,20.):
                for tau in (28.,56.,np.inf):
                    confidence=count/(count+k)*np.exp(-np.minimum(age,1e9)/tau)
                    result,_=evaluate_variant(x,f'k{k:g}_tau{tau:g}',pop+confidence[:,None]*(current-pop),bap);rows.append(result)
        if args.phase=='channel':
            for alpha in (20.,100.):
                pf=personal_factor(p,w,x,alpha)
                for blend in (.25,.5):
                    combined=(1-blend)*f[None,:]+blend*pf
                    result,_=evaluate_variant(x,f'channel_a{alpha:g}_b{blend:g}',fuse_component(x,raw*combined)+.25*mf,bap);rows.append(result)
        print(w['name'],args.phase,'complete',flush=True)
    note='Frozen final choice on previously inspected reused diagnostic; no tuning or submission CSV' if args.phase=='final' else 'Past-defined cohorts; repeated overlapping selection periods; no diagnostic or submission CSV'
    (ROOT/f'eighth_{args.phase}.json').write_text(json.dumps({'rows':rows,'note':note},indent=2))

def personal_factor(p,w,x,alpha):
    start,cutoff,end=map(pd.Timestamp,(w['fit_start'],w['cutoff'],w['end']))
    raw=np.load(RAW_CACHE);channels=np.unique(np.concatenate([raw['channel'],*[s['channel_id'].unique() for s in p['schedules'].values()]]))
    mask=(raw['start_ts']>=start.value//10**9)&(raw['stop_ts']<=cutoff.value//10**9)&(raw['stop_ts']>raw['start_ts'])
    c=np.searchsorted(channels,raw['channel'][mask]);u=raw['uidx'][mask]
    counts=np.bincount(u*len(channels)+c,minlength=len(p['user_ids'])*len(channels)).reshape(len(p['user_ids']),len(channels))
    global_counts=counts.sum(axis=0);prior=global_counts/max(global_counts.sum(),1)
    probabilities=(counts+alpha*prior[None,:])/(counts.sum(axis=1,keepdims=True)+alpha)
    probabilities=probabilities[np.searchsorted(p['user_ids'],x['users'])]
    past=pd.concat([p['schedules']['export_arh_11-20-final.csv'],p['schedules']['export_arh_21-30-final.csv']],ignore_index=True)
    def exposure(schedule,lo,hi):
        s=_airing_rows(schedule,lo,hi);idx=np.searchsorted(x['candidates'],s['tv_show_id'].to_numpy())
        valid=idx<len(x['candidates']);valid &= x['candidates'][np.clip(idx,0,len(x['candidates'])-1)]==s['tv_show_id'].to_numpy()
        matrix=sparse.csr_matrix((np.ones(valid.sum()),(idx[valid],np.searchsorted(channels,s.loc[valid,'channel_id']))),shape=(len(x['candidates']),len(channels)))
        return np.minimum(matrix.dot(probabilities.T).T/((hi-lo).total_seconds()/86400),2.)
    from .context_experiment import _schedule_for
    before=exposure(past,start,cutoff);after=exposure(_schedule_for(w,p),cutoff,end)
    return np.clip(np.sqrt((after+.2)/(before+.2)),.5,1.5)

if __name__=='__main__':main()
