"""Small opportunity refinement; selection-only, no submission output."""
import json
import itertools
import time
import numpy as np
import pandas as pd
from .config import load_config
from .corrected_target_experiment import ROOT
from .context_experiment import _schedule_for
from .evaluate import load_prepared
from .fourth_pass import prepare_window, fused, evaluate_variant
from .fifth_pass import component_scores, fuse_component, _airing_rows
from .pipeline import sha256_file

def factor(p,w,candidates,s,e):
    start,cutoff,end=map(pd.Timestamp,(w['fit_start'],w['cutoff'],w['end']))
    past=pd.concat([p['schedules']['export_arh_11-20-final.csv'],p['schedules']['export_arh_21-30-final.csv']],ignore_index=True)
    def rate(schedule,lo,hi):
        a=_airing_rows(schedule,lo,hi)
        count=a.groupby('tv_show_id').size().reindex(candidates,fill_value=0).to_numpy(dtype=float)
        return np.minimum(count/((hi-lo).total_seconds()/86400),2)
    return np.clip(((rate(_schedule_for(w,p),cutoff,end)+s)/(rate(past,start,cutoff)+s))**e,.5,1.5)

def main():
    cfg=load_config();p=load_prepared(cfg)
    cp=json.loads((ROOT/'corrected_plan.json').read_text())
    rows=[];stats=[]
    for w in cp['selection_windows']:
        t=time.perf_counter();x=prepare_window(p,cfg,w)
        baseline,ap=evaluate_variant(x,'fourth_submitted',fused(x,28.,2.,.25,opportunity=True));rows.append(baseline)
        components,_=component_scores(p,w,x)
        integrated=components['decayed_h56']/x['opportunity'][None,:]
        del components
        factors={(s,e):factor(p,w,x['candidates'],s,e) for s,e in itertools.product((.05,.1,.2),(.25,.5))}
        for rep,s,e in itertools.product(('last28','integrated56'),(.05,.1,.2),(.25,.5)):
            f=factors[s,e]
            raw=x['repeat'][28.] if rep=='last28' else integrated
            row,_=evaluate_variant(x,f'{rep}_s{s:g}_e{e:g}',fuse_component(x,raw*f[None,:]),ap)
            row.update({'representation':rep,'smoothing':s,'exponent':e});rows.append(row)
        stats.append({'window':w['name'],'runtime_seconds':round(time.perf_counter()-t,2)})
        print(w['name'],'completed',flush=True)
    (ROOT/'sixth_selection.json').write_text(json.dumps({'plan_sha256':sha256_file(ROOT/'sixth_plan.json'),'rows':rows,'stats':stats},indent=2))
if __name__=='__main__':main()
