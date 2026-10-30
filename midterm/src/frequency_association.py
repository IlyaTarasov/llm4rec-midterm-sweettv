"""Bounded count-ratio and regularized-rule tests, without CSV generation."""
import argparse,json,time
import numpy as np
from scipy import sparse
from .config import load_config
from .evaluate import load_prepared,fuse_scores
from .corrected_target_experiment import ROOT
from .fourth_pass import prepare_window,evaluate_variant
from .eighth_pass import internal,history
from .models import AssociationModel
from .pipeline import cfg_with,sha256_file

class RegularizedAssociation(AssociationModel):
    def fit_regularized(self,data,cfg,penalty,beta):
        started=time.perf_counter();super().fit(data,cfg,42)
        co=(data.X_binary.T@data.X_binary).tocsr()
        edges=self.S.tocoo();support=np.asarray(co[edges.row,edges.col]).ravel()
        boost=np.minimum((self.n_users/np.maximum(self.pop[edges.col],1))**beta,3.)
        values=edges.data*support/(support+penalty)*boost
        self.S=sparse.csr_matrix((values,(edges.row,edges.col)),shape=self.S.shape)
        if time.perf_counter()-started>60:raise TimeoutError('declared association budget exceeded')
        return self

def combine(x,raw,f,mf,count,age,power=None,assoc=None):
    repeat=raw if power is None else np.expm1(raw)**power
    s=fuse_scores({'popularity':x['pop'],'association':x['assoc'] if assoc is None else assoc,'repeat':repeat*f[None,:]},
                 {'popularity':1.,'association':.25,'repeat':2.},{'popularity':'global','association':'user','repeat':'user'})+.25*mf
    pop=fuse_scores({'popularity':x['pop']},{'popularity':1.},{'popularity':'global'})
    confidence=count/(count+5)*np.exp(-np.minimum(age,1e9)/56)
    return pop+confidence[:,None]*(s-pop)

def cohorts(aps,base,count,age):
    result=[]
    for label,mask in {'n1-4':(count>0)&(count<5),'n5-19':(count>=5)&(count<20),'n20+':count>=20,'recent3':age<=3,'age3-14':(age>3)&(age<=14),'age14+':(age>14)&np.isfinite(age)}.items():
        delta=aps[mask]-base[mask]
        result.append({'cohort':label,'n':int(mask.sum()),'mean_delta':float(delta.mean()) if len(delta) else None,'gained':int((delta>1e-12).sum()),'lost':int((delta< -1e-12).sum())})
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('count','association'),required=True);args=parser.parse_args()
    cfg=load_config();p=load_prepared(cfg);windows=json.loads((ROOT/'corrected_plan.json').read_text())['selection_windows'];rows=[];details=[]
    power=None
    if args.phase=='association':
        d=json.loads((ROOT/'frequency_count_decision.json').read_text())
        if d['result_sha256']!=sha256_file(ROOT/'frequency_count.json'):raise RuntimeError('count selection changed')
        power=d['retained_power']
    for w in windows:
        x=prepare_window(p,cfg,w);raw,f,mf,_=internal(p,w,x);count,age=history(x)
        baseline,bap=evaluate_variant(x,'submitted_confidence',combine(x,raw,f,mf,count,age));rows.append(baseline)
        running,rap=evaluate_variant(x,'running_control',combine(x,raw,f,mf,count,age,power));rows.append(running)
        variants={}
        if args.phase=='count':
            variants={f'power{v:g}':combine(x,raw,f,mf,count,age,v) for v in (.5,1.,1.5)}
        else:
            acfg=cfg_with(cfg,{'association':{'min_support':10,'top_k':100,'half_life_days':30.,'lift_population':'training_active'}})
            uidx=np.searchsorted(p['user_ids'],x['users']);cidx=np.searchsorted(p['item_ids'],x['candidates'])
            pop=np.asarray(x['train'].X_binary.sum(axis=0)).ravel()[cidx];high=pop>=np.quantile(pop,.9)
            top=np.argsort(-x['assoc'],axis=1,kind='stable')[:,:5]
            details.append({'window':w['name'],'baseline_association_top5_fraction_in_training_popularity_top_decile':float(high[top].mean())})
            for penalty in (10.,50.):
                for beta in (.25,.5):
                    model=RegularizedAssociation().fit_regularized(x['train'],acfg,penalty,beta)
                    score=model.score(uidx,cidx,x['train'])
                    variants[f'assoc_l{penalty:g}_b{beta:g}']=combine(x,raw,f,mf,count,age,power,score)
        for name,score in variants.items():
            row,ap=evaluate_variant(x,name,score,rap);rows.append(row)
            details.append({'window':w['name'],'variant':name,'cohorts':cohorts(ap,rap,count,age)})
        print(w['name'],args.phase,'complete',flush=True)
    (ROOT/f'frequency_{args.phase}.json').write_text(json.dumps({'plan_sha256':sha256_file(ROOT/'frequency_association_plan.json'),'rows':rows,'details':details,'reuse':'all periods are overlapping reused development data'},indent=2))
if __name__=='__main__':main()
