"""Independent arithmetic/integrity checks over saved experiment outputs.

Does not import experiment metric functions. Checks are reproducibility checks,
not evidence that assumptions about real-world field availability are true.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss,f1_score,average_precision_score

ROOT=Path(__file__).resolve().parent

def close(a,b):
    if not np.isclose(a,b,atol=1e-9,rtol=1e-8,equal_nan=True):
        raise AssertionError(f'Values disagree: {a} != {b}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results');args=p.parse_args();out=args.results
    b=pd.read_csv(out/'01_benchmark/candidate_predictions.csv')
    summary=pd.read_csv(out/'01_benchmark/summary.csv')
    for row in summary.itertuples():
        d=b if row.subset=='all' else b[b.hidden.eq(row.subset=='hidden')]
        invalid=d['gt'].eq('temporal_invalid');valid=~invalid
        reject=d[row.method].eq('temporal_invalid');accept=d[row.method].eq('valid');unknown=d[row.method].eq('unknown')
        tp=int((invalid&reject).sum());fp=int((valid&reject).sum());fn=int((invalid&~reject).sum())
        close(row.f1_invalid_all,2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        close(row.coverage,float((~unknown).mean()))
        close(row.recall_invalid_all,tp/invalid.sum())
        close(row.invalid_accepted,int((invalid&accept).sum()))
        close(row.risk_among_accepted,(invalid&accept).sum()/accept.sum() if accept.any() else np.nan)
        close(row.n_accept,int(accept.sum()));close(row.n_reject,int(reject.sum()));close(row.n_unknown,int(unknown.sum()))
    print('PASS v4: all 15 summary rows, Unknown accounting, acceptance risk')
    runs=pd.read_csv(out/'02_pipeline/runs.csv');pred=pd.read_csv(out/'02_pipeline/predictions.csv')
    cells=pd.read_csv(out/'02_pipeline/cell_audit.csv')
    for r in runs.itertuples():
        d=pred[(pred.seed==r.seed)&(pred.policy==r.policy)]
        close(r.roc_auc,roc_auc_score(d.target,d.probability));close(r.log_loss,log_loss(d.target,d.probability))
        a=cells[(cells.seed==r.seed)&(cells.policy==r.policy)]
        close(r.provably_late_cells,a.provably_late.sum())
        close(r.unproven_accepted_cells,a.unproven_accepted.sum())
        if r.policy!='event_only':
            assert not a.provably_late.any() and not a.unproven_accepted.any()
    for seed,d in pd.read_csv(out/'02_pipeline/splits.csv').groupby('seed'):
        train=pd.to_datetime(d.loc[d.split=='train','prediction_time'],utc=True)
        test=pd.to_datetime(d.loc[d.split=='test','prediction_time'],utc=True)
        assert train.max()+pd.Timedelta(days=7)<test.min()
    print('PASS pipeline: 15 model runs, saved cell counts, chronological label maturity')
    pairs=pd.read_csv(out/'03_matching/valentine_pilot_pair_results.csv')
    matched=pd.read_csv(out/'03_matching/matched_columns.csv')
    for r in pairs.itertuples():
        gt=json.loads((ROOT/'data/valentine'/r.pair/'ground_truth.json').read_text())
        truth={(x['source_column'],x['target_column']) for x in gt['matches']}
        m=matched[(matched.pair==r.pair)&(matched.method==r.method)]
        guess=set(zip(m.source_column,m.target_column));tp=len(guess&truth)
        close(r.precision,tp/len(guess));close(r.recall,tp/len(truth))
        close(r.f1,2*tp/(len(guess)+len(truth)));close(r.true_positive,tp)
    for r in pd.read_csv(out/'03_matching/valentine_pilot_summary.csv').itertuples():
        d=pairs[pairs.method==r.method]
        close(r.mean_f1,d.f1.mean());close(r.mean_top1_recall,d.top1_recall.mean())
    print('PASS matching: 20 assignments checked against ground truth, aggregate F1')
    pred=pd.read_csv(out/'04_nyc/predictions.csv');folds=pd.read_csv(out/'04_nyc/fold_results.csv')
    for r in folds.itertuples():
        d=pred[(pred.test_month==r.test_month)&(pred['mode']==r.mode)]
        close(r.roc_auc,roc_auc_score(d.target,d.probability))
        close(r.average_precision,average_precision_score(d.target,d.probability))
        close(r.log_loss,log_loss(d.target,d.probability))
        close(r.macro_f1,f1_score(d.target,d.probability>=.5,average='macro'))
    for month,d in pd.read_csv(out/'04_nyc/splits.csv').groupby('test_month'):
        train=d[d.split=='train'];test=d[d.split=='test']
        assert not set(train.unique_key)&set(test.unique_key)
        assert pd.to_datetime(train.label_available_at).max()<pd.to_datetime(test.created_date).min()
        ps=pred[pred.test_month==month]
        groups=[set(x.unique_key) for _,x in ps.groupby('mode')]
        assert groups[0]==groups[1]
    for r in pd.read_csv(out/'04_nyc/summary.csv').itertuples():
        d=folds[folds['mode']==r.mode]
        close(r.roc_auc_mean,d.roc_auc.mean());close(r.roc_auc_std,d.roc_auc.std())
    print('PASS NYC: 10 model runs, same test cohorts, all label cutoffs')
    manifest=out/'run_manifest.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        for item in record['inputs']:
            file=ROOT/item['path']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=item['sha256']:
                raise AssertionError(f'Input/code changed since run: {file}')
        print(f"PASS run manifest: {len(record['inputs'])} input/code checksums")
    print('ALL CHECKS PASSED')

if __name__=='__main__':
    main()
