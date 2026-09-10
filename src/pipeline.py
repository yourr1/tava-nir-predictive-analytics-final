"""Executable catalog -> key matching -> temporal join -> training -> prediction.

Controlled synthetic mechanism test, NOT a real-world validation of inferred metadata.
Bounds and entity types are supplied by the generator, never inferred from labels.
All policies run on the same rows, chronological split, learner and base features.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .matching import name_score
from .temporal import select_version, timestamp

SEEDS=[7,17,27,37,47]

def generate(seed,n=1500):
    rng=np.random.default_rng(seed)
    latent=rng.normal(size=n); plan=rng.normal(size=n)
    y=rng.binomial(1,1/(1+np.exp(-(1.2*latent+1.4*plan))))
    times=pd.date_range('2020-01-01',periods=n,tz='UTC')
    base=pd.DataFrame({'entity_id':np.arange(n),'prediction_time':times,
                       'base_signal':latent+rng.normal(0,2,n),'target':y})
    catalog=[]
    day=pd.Timedelta(days=1)
    specifications=[('prior_score','historical'),('delayed_outcome','historical'),
                    ('published_plan','scheduled'),('unversioned','historical'),('revised_score','historical')]
    for feature,kind in specifications:
        rows=[]
        for i,p in enumerate(times):
            if feature=='prior_score':
                values=[(latent[i]+rng.normal(0,.5),p-day,p-day,p-day)]
            elif feature=='delayed_outcome':
                values=[(float(y[i])+rng.normal(0,.03),p-day,p+2*day,p+2*day)]
            elif feature=='published_plan':
                values=[(plan[i],p+2*day,p-2*day,p-day)]
            elif feature=='unversioned':
                values=[(rng.normal(),p-day,pd.NaT,pd.NaT)]
            else:
                values=[(latent[i]+rng.normal(0,.7),p-5*day,p-4*day,p-4*day),
                        (float(y[i])+rng.normal(0,.03),p-5*day,p+3*day,p+3*day)]
            for version,(value,event,lo,hi) in enumerate(values):
                rows.append({'entity id':i,'event_time':event,'available_lower':lo,
                             'available_upper':hi,'evidence_id':'' if pd.isna(hi) else 'synthetic_contract',
                             'value':value,'version':version})
        catalog.append(dict(feature=feature,entity_type='customer',kind=kind,rows=rows))
    return base,catalog


def assemble(base,catalog,policy):
    result=base[['base_signal']].copy(); audit=[]
    for source in catalog:
        # Local discovery over metadata; an entity type is a supplied contract.
        if source['entity_type']!='customer':
            continue
        columns=list(source['rows'][0])
        key=max(columns,key=lambda c:name_score('entity_id',c))
        if name_score('entity_id',key)<.8:
            continue
        groups={}
        for row in source['rows']:
            groups.setdefault(row[key],[]).append(row)
        values=[]
        for _,row in base.iterrows():
            records=groups.get(row.entity_id,[])
            p=row.prediction_time
            if policy=='event_only':
                usable=[r for r in records if timestamp(r['event_time'])<=p]
                chosen=max(usable,key=lambda r:(r['event_time'],r['version'])) if usable else None
                decision='accept' if chosen else 'reject'; reason='event_time_only'
            else:
                if policy=='exact_only':
                    records=[r for r in records if pd.notna(r['available_lower']) and r['available_lower']==r['available_upper']]
                chosen,verdict=select_version(records,p,historical=source['kind']=='historical')
                decision,reason=verdict.decision,verdict.reason
            values.append(chosen['value'] if chosen else np.nan)
            # Unknown truth stays unknown. Count only demonstrably late chosen versions.
            late=bool(chosen is not None and pd.notna(chosen['available_lower']) and chosen['available_lower']>p)
            unknown=bool(chosen is not None and pd.isna(chosen['available_upper']))
            audit.append(dict(entity_id=int(row.entity_id),feature=source['feature'],policy=policy,
                              decision=decision,reason=reason,value=values[-1],
                              selected_version=chosen['version'] if chosen else None,
                              provably_late=late,unproven_accepted=unknown))
        result[source['feature']]=values
    return result,pd.DataFrame(audit)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]; all_predictions=[]; all_audits=[]; splits=[]
    for seed in SEEDS:
        base,catalog=generate(seed)
        cutoff=base.prediction_time.iloc[1050]
        train=(base.prediction_time+pd.Timedelta(days=7)<cutoff).to_numpy()
        test=(base.prediction_time>=cutoff).to_numpy()
        for policy in ['event_only','exact_only','interval_contract']:
            X,audit=assemble(base,catalog,policy)
            # All-empty columns carry no signal and are excluded using TRAIN only.
            columns=X.columns[X.loc[train].notna().any()].tolist()
            model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                                LogisticRegression(max_iter=2000,tol=1e-7,random_state=seed))
            model.fit(X.loc[train,columns],base.loc[train,'target'])
            prob=model.predict_proba(X.loc[test,columns])[:,1]
            rows.append(dict(seed=seed,policy=policy,n_train=int(train.sum()),n_test=int(test.sum()),
                n_features=len(columns),roc_auc=roc_auc_score(base.loc[test,'target'],prob),
                log_loss=log_loss(base.loc[test,'target'],prob),
                selected_cells=int(audit.decision.eq('accept').sum()),
                provably_late_cells=int(audit.provably_late.sum()),
                unproven_accepted_cells=int(audit.unproven_accepted.sum())))
            audit['seed']=seed;all_audits.append(audit)
            pred=base.loc[test,['entity_id','prediction_time','target']].copy()
            pred['probability']=prob;pred['seed']=seed;pred['policy']=policy;all_predictions.append(pred)
        split=base[['entity_id','prediction_time','target']].copy();split['seed']=seed
        split['split']=np.where(train,'train',np.where(test,'test','purged'));splits.append(split)
        print(f'Synthetic pipeline seed={seed} completed',flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(out/'runs.csv',index=False)
    summary=result.groupby('policy',as_index=False).agg(seeds=('seed','count'),
        roc_auc_mean=('roc_auc','mean'),roc_auc_std=('roc_auc','std'),
        provably_late_cells=('provably_late_cells','sum'),unproven_accepted_cells=('unproven_accepted_cells','sum'))
    summary.to_csv(out/'summary.csv',index=False)
    pd.concat(all_predictions).to_csv(out/'predictions.csv',index=False)
    pd.concat(all_audits).to_csv(out/'cell_audit.csv',index=False)
    pd.concat(splits).to_csv(out/'splits.csv',index=False)
    paired=result.pivot(index='seed',columns='policy',values='roc_auc')
    delta=(paired.interval_contract-paired.exact_only).to_numpy()
    rng=np.random.default_rng(42)
    boot=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
    config=dict(seeds=SEEDS,n_entities_per_seed=1500,horizon_days=7,first_test_row=1050,
                positive_control='interval vs exact-only: safe retention of a published future plan',
                delta_auc_mean=float(delta.mean()),bootstrap_seed_ci95=np.quantile(boot,[.025,.975]).tolist(),
                caution='Five synthetic seeds; mechanism test, not external validity or algorithmic novelty.')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)
