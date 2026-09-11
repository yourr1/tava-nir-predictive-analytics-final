"""Controlled metadata-loss experiment. No claim of real-world availability inference.

Run: python run_uncertainty.py --out results
All methods share chronological rows, learner, and truth. Masking is independent
of labels. Late features are a deliberate leakage control, not a usable model.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from temporal import verify

def run(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    runs=[]; predictions=[]; truth_rows=[]
    seeds=list(range(10))
    n=1000
    policies=['event_only','exact_only','interval','oracle']
    for seed in seeds:
        rng=np.random.default_rng(seed)
        z=rng.normal(size=n)
        y=rng.binomial(1,1/(1+np.exp(-1.7*z)))
        base=z+rng.normal(0,2,n)
        external=np.column_stack([z+rng.normal(0,.5,n),z+rng.normal(0,1,n),y+rng.normal(0,.08,n)])
        p=np.arange(n,dtype=float)
        event=np.repeat((p-5)[:,None],3,axis=1)
        jitter=rng.uniform(0,1,(n,3))
        hide=rng.uniform(size=(n,3))
        train=np.arange(n)<693  # seven-day label maturity before test at 700
        test=np.arange(n)>=700
        for delay in [1,4,8]:
            actual=event+delay+jitter
            actual[:,2]=p+1+jitter[:,2]  # explicitly unavailable target proxy
            truth=actual<=p[:,None]
            for missing in [0.,.5,1.]:
                for width in [1.,4.,8.]:
                    masked=hide<missing
                    # Coarsened calendar bins, not bounds inferred from labels.
                    lo=np.where(masked,np.floor(actual/width)*width,actual)
                    hi=np.where(masked,lo+width,actual)
                    assert np.all((lo<=actual)&(actual<=hi))
                    decisions=np.empty((n,3),dtype=object)
                    origin=pd.Timestamp('2020-01-01',tz='UTC')
                    for i in range(n):
                        for j in range(3):
                            decisions[i,j]=verify(origin+pd.Timedelta(days=p[i]),
                                origin+pd.Timedelta(days=lo[i,j]),origin+pd.Timedelta(days=hi[i,j]),
                                evidence_id='synthetic_calendar_bin').decision
                    masks={'event_only':event<=p[:,None],
                           'exact_only':(~masked)&truth,
                           'interval':decisions=='accept','oracle':truth}
                    assert not np.any(masks['interval']&~truth)
                    assert np.all(~masks['exact_only']|masks['interval'])
                    for policy in policies:
                        start=time.perf_counter()
                        accepted=masks[policy]
                        X=np.column_stack([base,np.where(accepted,external,np.nan)])
                        keep=np.any(~np.isnan(X[train]),axis=0)
                        model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),
                            StandardScaler(),LogisticRegression(max_iter=1000,random_state=seed))
                        model.fit(X[train][:,keep],y[train])
                        prob=model.predict_proba(X[test][:,keep])[:,1]
                        a=accepted[test]; t=truth[test]
                        unknown=(decisions[test]=='unknown') if policy=='interval' else (masked[test] if policy=='exact_only' else np.zeros_like(a))
                        row=dict(seed=seed,delay=delay,missing=missing,width=width,policy=policy,
                            n_test=int(test.sum()),accepted=int(a.sum()),late=int((a&~t).sum()),
                            coverage=float(a.mean()),unknown=float(unknown.mean()),
                            late_rate=float((a&~t).sum()/a.sum()) if a.sum() else None,
                            auc=float(roc_auc_score(y[test],prob)),log_loss=float(log_loss(y[test],prob)),
                            fit_predict_seconds=time.perf_counter()-start)
                        runs.append(row)
                        for i,pr in zip(np.flatnonzero(test),prob):
                            predictions.append(dict(seed=seed,delay=delay,missing=missing,width=width,
                                policy=policy,entity=int(i),target=int(y[i]),probability=float(pr)))
            for i in range(n):
                for j in range(3):
                    truth_rows.append(dict(seed=seed,delay=delay,entity=i,feature=j,
                        prediction=p[i],event=event[i,j],actual=actual[i,j],mask_uniform=hide[i,j],
                        value=external[i,j],base=base[i],target=y[i],split='train' if train[i] else 'test' if test[i] else 'purged'))
        print(f'seed {seed} complete',flush=True)
    df=pd.DataFrame(runs)
    df.to_csv(out/'runs.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(truth_rows).to_csv(out/'truth.csv',index=False)
    df.groupby(['delay','missing','width','policy'],as_index=False)[['auc','coverage','unknown','late_rate','log_loss']].mean().to_csv(out/'summary.csv',index=False)
    selected=df[(df.delay==1)&(df.missing==.5)&(df.width==4)]
    paired=selected.pivot(index='seed',columns='policy',values='auc')
    delta=(paired['interval']-paired['exact_only']).to_numpy()
    boot=np.random.default_rng(123).choice(delta,(10000,len(delta)),replace=True).mean(axis=1)
    config=dict(seeds=seeds,n=n,first_test=700,label_maturity_days=7,
        scenarios=27,fits=len(df),primary_scenario=dict(delay=1,missing=.5,width=4),
        delta_auc=float(delta.mean()),paired_seed_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
        caveat='Synthetic mechanism study. Valid calendar bounds supplied. No real metadata inference. Other scenarios exploratory.',
        files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.csv')})
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(selected.groupby('policy')[['auc','coverage','unknown','late']].mean().to_string())
    print(json.dumps(config,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results');args=ap.parse_args();run(args.out)
