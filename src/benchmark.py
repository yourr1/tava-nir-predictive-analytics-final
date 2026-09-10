"""Corrected historical v4 audit. No end-to-end oracle test masking.

Text model is deliberately retained as an uncalibrated heuristic baseline.
Ground truth and strata are read only by the evaluator, never by inference.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


def explicit_label(base, candidate, time_col='available_time'):
    if time_col not in candidate:
        return 'unknown'
    if not base.entity_id.is_unique or not candidate.entity_id.is_unique:
        raise ValueError('Expected unique entity_id per candidate')
    paired = base[['entity_id','prediction_time']].merge(
        candidate[['entity_id',time_col]], on='entity_id', how='left', validate='one_to_one')
    p = pd.to_datetime(paired.prediction_time, utc=True, errors='coerce')
    t = pd.to_datetime(paired[time_col], utc=True, errors='coerce')
    # A candidate column must be usable for every requested row.
    if (t > p).any():
        return 'temporal_invalid'
    if p.isna().any() or t.isna().any() or len(paired) == 0:
        return 'unknown'
    return 'valid'


def metrics(frame, col):
    known = frame[frame['gt'].isin(['valid','temporal_invalid'])]
    gt, pred = known['gt'], known[col]
    invalid, valid = gt.eq('temporal_invalid'), gt.eq('valid')
    decided, accepted, rejected = pred.ne('unknown'), pred.eq('valid'), pred.eq('temporal_invalid')
    tp, fp = int((invalid & rejected).sum()), int((valid & rejected).sum())
    fn = int((invalid & ~rejected).sum())  # Unknown is not a detected invalid.
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / int(invalid.sum()) if invalid.any() else 0.0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    cond = precision_recall_fscore_support(invalid[decided], rejected[decided],
                                         average='binary', zero_division=0)[2] if decided.any() else np.nan
    return dict(n_all=len(frame), n_known=len(known), n_invalid=int(invalid.sum()),
                n_valid=int(valid.sum()), n_accept=int(accepted.sum()),
                n_reject=int(rejected.sum()), n_unknown=int((~decided).sum()),
                coverage=float(decided.mean()), precision_invalid=precision,
                recall_invalid_all=recall, f1_invalid_all=f1, f1_invalid_decided=cond,
                false_rejection_all_valid=fp/max(1,int(valid.sum())),
                invalid_accepted=int((invalid & accepted).sum()),
                risk_among_accepted=float((invalid & accepted).sum()/accepted.sum()) if accepted.any() else np.nan)


def load_tasks(root):
    rows=[]
    tasks=sorted(Path(root).glob('task_*'))
    if len(tasks)!=80:
        raise ValueError(f'Expected all 80 tasks, got {len(tasks)}')
    for path in tasks:
        meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        base=pd.read_csv(path/'base.csv')
        for c in meta['candidates']:
            data=pd.read_csv(path/c['file'])
            rows.append(dict(task=int(meta['task_id']),candidate=c['candidate_id'],
                text=f"{c['feature']} {c['description']}", gt=c['ground_truth'],
                hidden=c['hidden_availability'],stratum=c['stratum'],
                explicit=explicit_label(base,data),event_only=explicit_label(base,data,'event_time')))
    return pd.DataFrame(rows)


def run(root, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    frame=load_tasks(root)
    train=frame[frame.task<40].copy(); hold=frame[frame.task.between(40,55)].copy()
    test=frame[frame.task>=56].copy()
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=1,sublinear_tf=True)
    clf=LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42)
    clf.fit(vec.fit_transform(train.text),train['gt'])
    for split in (hold,test):
        probs=clf.predict_proba(vec.transform(split.text))
        split['semantic']=clf.classes_[probs.argmax(axis=1)]
        split['confidence']=probs.max(axis=1)
    def hybrid(split,threshold):
        return np.where(split.explicit!='unknown',split.explicit,
                        np.where(split.confidence>=threshold,split.semantic,'unknown'))
    tuning=[]
    for th in np.linspace(.5,.95,19):
        hold['hybrid']=hybrid(hold,th)
        tuning.append(dict(threshold=float(th),**metrics(hold,'hybrid')))
    tuning=pd.DataFrame(tuning)
    eligible=tuning[tuning.false_rejection_all_valid<=.15]
    # Deterministic dev-only selection; full recall, not conditional recall.
    selected=eligible.sort_values(['recall_invalid_all','risk_among_accepted','coverage','threshold'],
                                  ascending=[False,True,False,True]).iloc[0]
    threshold=float(selected.threshold)
    test['hybrid']=hybrid(test,threshold)
    # Original threshold retained only as a sensitivity point, not chosen on test.
    test['hybrid_legacy_threshold']=hybrid(test,.625)
    columns=['explicit','event_only','semantic','hybrid','hybrid_legacy_threshold']
    summaries=[]
    for subset,data in [('all',test),('hidden',test[test.hidden]),('explicit',test[~test.hidden])]:
        for col in columns:
            summaries.append(dict(subset=subset,method=col,**metrics(data,col)))
    test.to_csv(out/'candidate_predictions.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    tuning.to_csv(out/'dev_thresholds.csv',index=False)
    # Repeated descriptions across train/test undermine external generalization.
    overlap=test.text.isin(set(train.text))
    config=dict(tasks=80,train_tasks=list(range(40)),validation_tasks=list(range(40,56)),
                test_tasks=list(range(56,80)),seed=42,threshold=threshold,
                test_candidates=len(test),test_text_seen_in_train=int(overlap.sum()),
                known_class_counts=test['gt'].value_counts().to_dict(),
                warning='Synthetic repeated templates. Semantic confidence is not a probability of temporal correctness.')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).query("subset=='all'").to_string(index=False),flush=True)
