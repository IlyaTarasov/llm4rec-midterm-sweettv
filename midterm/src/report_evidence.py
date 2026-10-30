"""Reproduce one coherent corrected-method/report comparison; no model selection."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
import numpy as np
import pandas as pd
from .config import load_config,resolve
from .evaluate import load_prepared,recommend_from_scores,fuse_scores
from .fourth_pass import prepare_window,evaluate_variant
from .eighth_pass import internal,history
from .data import build_item_metadata
from .models import PopularityModel,AssociationModel,ContentModel,MFModel,PageRankModel
from .pipeline import cfg_for_method,cfg_with
from .corrected_target_experiment import ROOT
from .metrics import ap_at_k

def main():
    cfg=load_config();p=load_prepared(cfg);w=json.loads((ROOT/'corrected_plan.json').read_text())['selection_windows'][1]
    x=prepare_window(p,cfg,w);cutoff=pd.Timestamp(w['cutoff']);uids=x['users'];uidx=np.searchsorted(p['user_ids'],uids);cidx=np.searchsorted(p['item_ids'],x['candidates'])
    selected=json.loads(resolve('midterm/results/selected_model.json').read_text())
    chosen=[int(selected['users']['heavy']),int(selected['users']['light'])]
    meta=build_item_metadata([p['schedules']['export_arh_11-20-final.csv'],p['schedules']['export_arh_21-30-final.csv']],cutoff)
    x['train'].item_meta=meta;variants={};runtimes={}
    for name,cls,mc in [
        ('Popularity',PopularityModel,cfg_with(cfg,{'popularity':{'weight':'users'}})),
        ('Association',AssociationModel,cfg_with(cfg,{'association':{'min_support':10,'top_k':100,'half_life_days':30.,'lift_population':'training_active'}})),
        ('Content',ContentModel,cfg_for_method(cfg,'content',selected['method_configs']['content'])),
        ('MF',MFModel,cfg_with(cfg,{'mf':{'factors':16,'alpha':5.,'reg':.05,'iterations':15,'weight':'count'}})),
        ('PPR',PageRankModel,cfg_for_method(cfg,'pagerank',selected['method_configs']['pagerank']))]:
        t=time.perf_counter();m=cls().fit(x['train'],mc,42);variants[name]=m.score(uidx,cidx,x['train']);runtimes[name]=time.perf_counter()-t
    t=time.perf_counter();_,_,_,current=internal(p,w,x);count,age=history(x)
    pop=fuse_scores({'popularity':x['pop']},{'popularity':1.},{'popularity':'global'})
    variants['Final']=pop+(count/(count+5)*np.exp(-np.minimum(age,1e9)/56))[:,None]*(current-pop)
    runtimes['Final_incremental']=time.perf_counter()-t
    display=pd.concat(list(p['schedules'].values()),ignore_index=True).drop_duplicates('tv_show_id').set_index('tv_show_id')
    rows=[];examples=[];userstats=[]
    for user in chosen:
        inter=x['train'].interactions.loc[x['train'].interactions.user_id==user]
        userstats.append({'user_id':str(user),'qualified_airings':int(inter.view_count.sum()),'qualified_items':len(inter),'target_ids':sorted(x['gt'].get(user,set())),'target_size':len(x['gt'].get(user,set())),'training_top_items':inter.sort_values('view_count',ascending=False).head(5)[['item_id','view_count']].to_dict('records')})
    for name,score in variants.items():
        result,_=evaluate_variant(x,name,score);rows.append({**result,'fit_score_seconds':runtimes.get(name)})
        recs,_=recommend_from_scores(score,uids,x['candidates'],x['old'],'allow',x['fallback'],cold_mask=x['cold'])
        for user in chosen:
            seen=set(x['train'].interactions.loc[x['train'].interactions.user_id==user,'item_id'])
            for rank,item in enumerate(recs[user],1):
                info=display.loc[item];examples.append({'user_id':str(user),'method':name,'rank':rank,'id':item,'title':info.tv_show_title,'category':info.tv_show_category,'hit':item in x['gt'].get(user,set()),'seen_training':item in seen,'ap5':ap_at_k(recs[user],x['gt'].get(user,set())) or 0.})
    frozen=json.loads((ROOT/'eighth_confidence.json').read_text())['rows']
    target=next(r for r in frozen if r['window']==w['name'] and r['variant']=='k5_tau56')
    assert abs(next(r for r in rows if r['variant']=='Final')['map5_eligible5']-target['map5_eligible5'])<1e-12
    out={'window':w,'n_users':len(uids),'n_candidates':len(x['candidates']),'n_eligible5':len(x['eligible']),'qualified_cold':int(x['cold'].sum()),'methods':rows,'examples':examples,'users':userstats,'runtimes':runtimes,'executor':'assistant-executed reproduction of frozen models, no new selection'}
    (ROOT/'report_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'methods':rows,'users':userstats},ensure_ascii=True,indent=2))
if __name__=='__main__':main()
