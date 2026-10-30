"""Refit the frozen confidence/integrated-decay/MF strategy and write one CSV."""
import argparse,csv,io,json,platform,sys,time
import numpy as np
import pandas as pd
import scipy
from .config import load_config,resolve
from .context_experiment import _schedule_for,_empty_metadata
from .corrected_target_experiment import ROOT,_qualified_training
from .data import candidate_items
from .evaluate import load_prepared,build_train_data,fuse_scores,recommend_from_scores
from .eighth_pass import history
from .fifth_pass import component_scores,fuse_component
from .sixth_pass import factor
from .fourth_pass import opportunity_factor,evaluate_variant
from .models import PopularityModel,AssociationModel,MFModel
from .pipeline import cfg_with,sha256_file
from .qualified_airings import qualify_airings

CONFIG=resolve('midterm/configs/candidate_confidence_integrated_mf.json')
OUTPUT=resolve('midterm/submission_final_confidence_integrated_mf.csv')

def prior_hashes():
    expected=json.loads((ROOT/'eighth_manifest.json').read_text())['submitted_hashes']
    actual={n:sha256_file(resolve('midterm')/n) for n in expected}
    if actual!=expected:raise RuntimeError('immutable submitted file changed')
    return actual

def refit(p,cfg,w):
    start,cutoff,end=map(pd.Timestamp,(w['fit_start'],w['cutoff'],w['end']))
    candidates=candidate_items(_schedule_for(w,p),cutoff,end)
    users=p['submission']['user_id'].to_numpy(dtype=np.uint64)
    uidx=np.searchsorted(p['user_ids'],users);cidx=np.searchsorted(p['item_ids'],candidates)
    q=qualify_airings(p['joined'],start,cutoff)
    assert (q['sched_end']<=cutoff).all() and (q['screen_time']>=.8).all()
    train=_qualified_training(q,cfg,p,cutoff)
    old=build_train_data(cfg,p['joined'],p['user_ids'],p['item_ids'],_empty_metadata(),cutoff)
    pop=PopularityModel().fit(train,cfg_with(cfg,{'popularity':{'weight':'users'}}),42).score(uidx,cidx,train)
    fallback=PopularityModel().fit(old,cfg_with(cfg,{'popularity':{'weight':'users'}}),42).score(uidx,cidx,old)[0]
    assoc=AssociationModel().fit(train,cfg_with(cfg,{'association':{'min_support':10,'top_k':100,'half_life_days':30.,'lift_population':'training_active'}}),42).score(uidx,cidx,train)
    x={'window':w,'users':users,'candidates':candidates,'train':train,'old':old,'pop':pop,'assoc':assoc,'fallback':fallback,
       'cold':np.diff(train.X_binary.indptr)[uidx]==0,'opportunity':opportunity_factor(p,w,candidates)}
    components,counts=component_scores(p,w,x)
    raw=components['decayed_h56']/x['opportunity'][None,:];del components
    repeat=raw*factor(p,w,candidates,.2,.5)[None,:]
    current=fuse_component(x,repeat)
    model=MFModel().fit(train,cfg_with(cfg,{'mf':{'factors':16,'alpha':5.,'reg':.05,'iterations':15,'weight':'count'}}),42)
    mf=model.score(uidx,cidx,train);maximum=mf.max(axis=1,keepdims=True);mf/=np.where(maximum>0,maximum,1)
    current+=.25*mf
    count,age=history(x);confidence=count/(count+5.)*np.exp(-np.minimum(age,1e9)/56.)
    global_pop=fuse_scores({'popularity':pop},{'popularity':1.},{'popularity':'global'})
    scores=global_pop+confidence[:,None]*(current-global_pop)
    if not np.isfinite(scores).all():raise RuntimeError('nonfinite final scores')
    recs,stats=recommend_from_scores(scores,users,candidates,old,'allow',fallback,cold_mask=x['cold'])
    for row,user in enumerate(users):
        if x['cold'][row]:
            expected,_=recommend_from_scores(fallback[None,:],np.array([user],dtype=np.uint64),candidates,old,'allow',fallback,cold_mask=np.array([True]))
            if recs[int(user)]!=expected[int(user)]:raise RuntimeError('cold fallback mismatch')
    return x,scores,recs,stats,counts

def main():
    a=argparse.ArgumentParser();a.add_argument('--verify-existing',action='store_true');a.add_argument('--audit-diagnostic',action='store_true');args=a.parse_args();started=time.perf_counter()
    immutable=prior_hashes();config=json.loads(CONFIG.read_text())
    if config['selected_strategy']!='frozen_confidence_k5_tau56':raise RuntimeError('wrong final strategy')
    decision=json.loads((ROOT/'eighth_final_decision.json').read_text())
    if decision['confidence_sha256']!=sha256_file(ROOT/'eighth_confidence.json') or decision['channel_sha256']!=sha256_file(ROOT/'eighth_channel.json'):raise RuntimeError('frozen selection changed')
    old_manifest=json.loads((ROOT/'fourth_candidate_manifest.json').read_text())
    data_dir=resolve('midterm/data/sweettv-tv-program-recommender')
    if not all(sha256_file(data_dir/n)==h for n,h in old_manifest['input_hashes'].items()):raise RuntimeError('source integrity failure')
    cfg=load_config();p=load_prepared(cfg)
    w=json.loads((ROOT/'corrected_plan.json').read_text())['diagnostic_window'] if args.audit_diagnostic else {
        'name':'final_full_refit','fit_start':config['fit_start'],'cutoff':config['fit_cutoff'],'end':config['candidate_end_exclusive'],'schedule':'31-42'}
    x,scores,recs,stats,counts=refit(p,cfg,w)
    if args.audit_diagnostic:
        cutoff,end=map(pd.Timestamp,(w['cutoff'],w['end']))
        from .qualified_airings import top5_frequency
        q=qualify_airings(p['joined'],cutoff,end);gt,_=top5_frequency(q,x['candidates'])
        j=p['joined'];one=j.loc[(j.sched_start>=cutoff)&(j.sched_start<end)&(j.sched_end<=end)&(j.view_end<=end)&(j.overlap_fraction>=.8),['user_id','item_id','channel_id','sched_start','sched_end']].drop_duplicates()
        single,_=top5_frequency(one,x['candidates'])
        x.update({'gt':gt,'eligible':np.array([u for u in x['users'] if len(gt.get(int(u),()))==5],dtype=np.uint64),'gt_single':single,'single_eligible':np.array([u for u in x['users'] if len(single.get(int(u),()))==5],dtype=np.uint64)})
        row,_=evaluate_variant(x,'fresh_refit_frozen_confidence',scores)
        frozen=json.loads((ROOT/'eighth_final.json').read_text())['rows'][-1]
        for key in ('map5_all_users','map5_eligible5','map5_single_eligible5'):
            if abs(row[key]-frozen[key])>1e-12:raise RuntimeError('refit differs from frozen diagnostic')
        (ROOT/'final_refit_equivalence.json').write_text(json.dumps({'metrics':row,'matches_frozen':True,'runtime_seconds':time.perf_counter()-started},indent=2));print('fresh refit exactly reproduces frozen diagnostic',flush=True);return
    with (data_dir/'submission.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;template=list(reader)
    if fields!=['user_id','tv_show_id'] or len(template)!=1260:raise RuntimeError('template schema mismatch')
    ids=[r['user_id'] for r in template]
    if len(set(ids))!=1260 or ids!=[str(int(u)) for u in x['users']]:raise RuntimeError('string-exact template mismatch')
    allowed=set(map(int,x['candidates']));buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for user in ids:
        items=recs[int(user)]
        if len(items)!=5 or len(set(items))!=5 or any(i<=0 or i not in allowed for i in items):raise RuntimeError('invalid list')
        writer.writerow({'user_id':user,'tv_show_id':' '.join(map(str,items))})
    content=buffer.getvalue().encode('utf-8')
    if args.verify_existing:
        if OUTPUT.read_bytes()!=content:raise RuntimeError('deterministic rerun mismatch')
    else:
        with OUTPUT.open('xb') as f:f.write(content)
    with OUTPUT.open(newline='',encoding='utf-8') as f:written=list(csv.DictReader(f))
    if [r['user_id'] for r in written]!=ids:raise RuntimeError('written user string mismatch')
    fourth=pd.read_csv(resolve('midterm/submission_candidate_opportunity.csv'),dtype=str)
    new=[recs[int(u)] for u in ids];old=[list(map(int,s.split())) for s in fourth.tv_show_id]
    files=['context_experiment.py','qualified_airings.py','corrected_target_experiment.py','fifth_pass.py','sixth_pass.py','eighth_pass.py','models.py','evaluate.py','final_confidence_candidate.py']
    manifest={'csv':str(OUTPUT),'csv_sha256':sha256_file(OUTPUT),'config_sha256':sha256_file(CONFIG),'configuration':config,
      'deterministic_regeneration_match':bool(args.verify_existing),'n_users':1260,'n_candidates':len(allowed),'fallback_users':int(x['cold'].sum()),'fallback_stats':stats,'training_counts':counts,
      'cutoffs':w,'diagnostic_equivalence':json.loads((ROOT/'final_refit_equivalence.json').read_text()),
      'comparison_fourth':{'changed_ranked_users':sum(a!=b for a,b in zip(new,old)),'changed_sets':sum(set(a)!=set(b) for a,b in zip(new,old)),'mean_common_top5':sum(len(set(a)&set(b)) for a,b in zip(new,old))/1260},
      'source_hashes':old_manifest['input_hashes'],'prepared_cache_hashes':{n:sha256_file(resolve('midterm/artifacts')/n) for n in ['joined_all.pkl','vocab.npz']},
      'code_hashes':{n:sha256_file(resolve('midterm/src')/n) for n in files},
      'evidence_hashes':{n:sha256_file(ROOT/n) for n in ['eighth_plan.json','eighth_final_decision.json','eighth_confidence.json','eighth_final.json','poisson_final_decision.json']},
      'prior_submitted_hashes':prior_hashes(),'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'platform':platform.platform()},'runtime_seconds':round(time.perf_counter()-started,3),
      'caveat':'Repeated overlapping validation windows; union fragments and target tie handling locally specified; local gain predicts a direction, not a Kaggle score guarantee. Best confirmed fourth private0.39382.'}
    (ROOT/'final_confidence_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ['csv','csv_sha256','deterministic_regeneration_match','comparison_fourth','runtime_seconds']},indent=2),flush=True)
if __name__=='__main__':main()
