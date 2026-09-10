"""Rolling temporal leakage control on a monthly prefix sample of NYC 311.

This is a supervised negative control with manually declared input fields.
The snapshot has no historical versions of intake fields, so it does not prove
automatic temporal verification on real sources.
"""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SAFE=['agency','complaint_type','location_type','incident_zip','borough','open_data_channel_type']

def prepare(path):
    raw=pd.read_csv(path,dtype=str)
    if raw.unique_key.duplicated().any():
        raise ValueError('Duplicate unique_key')
    d=raw.copy()
    # Socrata NYC timestamps are LOCAL floating timestamps. Keep them local.
    d['created_date']=pd.to_datetime(d.created_date,errors='coerce')
    d['closed_date']=pd.to_datetime(d.closed_date,errors='coerce')
    delta=(d.closed_date-d.created_date).dt.total_seconds()/86400
    invalid=d.created_date.isna() | (delta<0)
    # Malformed nonempty closure strings are unknown, not unresolved cases.
    invalid |= raw.closed_date.notna() & d.closed_date.isna()
    d=d.loc[~invalid].copy();delta=delta.loc[d.index]
    d['target']=(delta.notna() & (delta<=7)).astype(int)
    d['label_available_at']=d.created_date+pd.Timedelta(days=7)
    d['closure_days_negative_control']=delta.fillna(30).clip(upper=30)
    d['created_hour']=d.created_date.dt.hour
    for col in SAFE:
        d[col]=d[col].fillna('__MISSING__')
    d=d.sort_values(['created_date','unique_key']).reset_index(drop=True)
    return d,dict(raw_rows=len(raw),excluded_invalid_rows=int(invalid.sum()),
                  unresolved_included=int(d.closed_date.isna().sum()),kept_rows=len(d),
                  first_created=str(d.created_date.min()),last_created=str(d.created_date.max()))

def temporal_split(data,cutoff,end):
    train=data.label_available_at<cutoff
    test=(data.created_date>=cutoff)&(data.created_date<end)
    if not train.any() or not test.any():
        raise ValueError('No mature train or test rows. A short snapshot is not a valid temporal study.')
    if not data.loc[train,'label_available_at'].max()<data.loc[test,'created_date'].min():
        raise AssertionError('Training labels would not yet exist at prediction time')
    return train,test

def run(path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d,meta=prepare(path)
    rows=[];predictions=[];splits=[]
    for month in range(8,13):
        cutoff=pd.Timestamp(year=2023,month=month,day=1)
        end=cutoff+pd.offsets.MonthBegin(1)
        train,test=temporal_split(d,cutoff,end)
        for mode in ['intake_assumed','leaky_negative_control']:
            numeric=['created_hour']+(['closure_days_negative_control'] if mode=='leaky_negative_control' else [])
            transform=ColumnTransformer([
                ('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=5),SAFE),
                ('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric)])
            model=make_pipeline(transform,LogisticRegression(max_iter=4000,tol=1e-6,random_state=42))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(d.loc[train,SAFE+numeric],d.loc[train,'target'])
            prob=model.predict_proba(d.loc[test,SAFE+numeric])[:,1]; y=d.loc[test,'target']
            rows.append(dict(test_month=month,mode=mode,n_train=int(train.sum()),n_test=int(test.sum()),
                             positive_rate=float(y.mean()),roc_auc=roc_auc_score(y,prob),
                             average_precision=average_precision_score(y,prob),
                             macro_f1=f1_score(y,prob>=.5,average='macro'),log_loss=log_loss(y,prob)))
            pred=d.loc[test,['unique_key','created_date','target']].copy()
            pred['probability']=prob;pred['mode']=mode;pred['test_month']=month;predictions.append(pred)
        split=d.loc[train|test,['unique_key','created_date','label_available_at']].copy()
        split['split']=np.where(train.loc[split.index],'train','test');split['test_month']=month;splits.append(split)
        print(f'NYC fold month={month}: train={train.sum()}, test={test.sum()}',flush=True)
    results=pd.DataFrame(rows);results.to_csv(out/'fold_results.csv',index=False)
    summary=results.groupby('mode',as_index=False).agg(folds=('test_month','count'),
               roc_auc_mean=('roc_auc','mean'),roc_auc_std=('roc_auc','std'),
               macro_f1_mean=('macro_f1','mean'),log_loss_mean=('log_loss','mean'))
    summary.to_csv(out/'summary.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    pd.concat(splits).to_csv(out/'splits.csv',index=False)
    meta.update(dict(horizon_days=7,test_months=list(range(8,13)),safe_fields=SAFE,
                     model='LogisticRegression C=1, lbfgs, max_iter=4000, tol=1e-6',
                     negative_control='Closed-created in days clipped at 30, unresolved filled with 30. Directly derived from the target definition.',
                     input_availability='Manually assumed intake fields; historical versions unavailable.',
                     label_assumption='Unclosed at 2026 snapshot implies not closed within 7 days of 2023 creation. Reopened requests can violate this assumption.',
                     generalization='First 2000 rows of each month. Not representative of all NYC 311.'))
    (out/'config.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)
