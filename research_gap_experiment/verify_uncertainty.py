"""Recalculate decisions from saved truth and metrics from saved predictions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('--results',default='results')
args=ap.parse_args()
p=Path(args.results)
c=json.loads((p/'config.json').read_text())
for name,digest in c['files'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
r=pd.read_csv(p/'runs.csv'); t=pd.read_csv(p/'truth.csv'); pred=pd.read_csv(p/'predictions.csv')
keys=['seed','delay','missing','width','policy']
groups=pred.groupby(keys)
for row in r.itertuples():
    truth=t[(t.seed==row.seed)&(t.delay==row.delay)&(t.split=='test')]
    hidden=truth.mask_uniform<row.missing
    lower=np.where(hidden,np.floor(truth.actual/row.width)*row.width,truth.actual)
    upper=np.where(hidden,lower+row.width,truth.actual)
    if row.policy=='event_only':
        accepted=truth.event<=truth.prediction; unknown=np.zeros(len(truth),bool)
    elif row.policy=='oracle':
        accepted=truth.actual<=truth.prediction; unknown=np.zeros(len(truth),bool)
    elif row.policy=='exact_only':
        accepted=(~hidden)&(truth.actual<=truth.prediction);unknown=hidden
    else:
        accepted=upper<=truth.prediction
        unknown=(lower<=truth.prediction)&(upper>truth.prediction)
    assert accepted.sum()==row.accepted
    assert (accepted&(truth.actual>truth.prediction)).sum()==row.late
    assert np.isclose(accepted.mean(),row.coverage)
    assert np.isclose(np.mean(unknown),row.unknown)
    pr=groups.get_group(tuple(getattr(row,k) for k in keys))
    assert sorted(pr.entity.tolist())==list(range(700,1000))
    labels=truth.drop_duplicates('entity').set_index('entity').target
    assert np.array_equal(labels.loc[pr.entity].values,pr.target.values)
    assert np.isclose(roc_auc_score(pr.target,pr.probability),row.auc)
    assert np.isclose(log_loss(pr.target,pr.probability),row.log_loss)
assert len(r)==1080
assert r.loc[r.policy=='interval','late'].sum()==0
print('PASS: 1080 runs, identical test objects, saved-truth decisions, AUC/log-loss and checksums')
