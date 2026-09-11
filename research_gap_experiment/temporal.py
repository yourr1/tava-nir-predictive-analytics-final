"""Small executable temporal contract; text confidence is NOT evidence.

Times are normalized to UTC. Naive timestamps are interpreted as UTC by contract.
Bounds must bound availability of THIS VERSION, not merely the event date.
Correctness is conditional on those supplied bounds being true.
"""
from dataclasses import dataclass
from typing import Literal
import pandas as pd

Decision = Literal['accept', 'reject', 'unknown']


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str


def timestamp(value):
    if isinstance(value, pd.Timestamp):
        return value.tz_localize('UTC') if value.tzinfo is None else value.tz_convert('UTC')
    return pd.to_datetime(value, utc=True, errors='coerce')


def verify(prediction_time, available_lower=None, available_upper=None,
           evidence_id: str = '', contradictory: bool = False) -> Verdict:
    p, lo, hi = map(timestamp, (prediction_time, available_lower, available_upper))
    if pd.isna(p):
        return Verdict('unknown', 'missing_prediction_time')
    if contradictory:
        return Verdict('unknown', 'contradictory_evidence')
    if not evidence_id or pd.isna(lo) or pd.isna(hi):
        return Verdict('unknown', 'missing_availability_evidence')
    if lo > hi:
        return Verdict('unknown', 'inverted_interval')
    if hi <= p:
        return Verdict('accept', 'upper_bound_at_or_before_prediction')
    if lo > p:
        return Verdict('reject', 'lower_bound_after_prediction')
    return Verdict('unknown', 'interval_crosses_prediction')


def derive_bounds(event_time, delay_min_days, delay_max_days, evidence_id):
    """Derive an interval only from an explicitly documented delay contract."""
    event = timestamp(event_time)
    if (not evidence_id or pd.isna(event) or delay_min_days is None
            or delay_max_days is None or delay_min_days > delay_max_days):
        return pd.NaT, pd.NaT
    return (event + pd.Timedelta(days=delay_min_days),
            event + pd.Timedelta(days=delay_max_days))


def aggregate_bounds(inputs, computation_time):
    """All inputs and the computation itself must already be available."""
    pairs = [(timestamp(lo), timestamp(hi)) for lo, hi in inputs]
    done = timestamp(computation_time)
    if not pairs or pd.isna(done) or any(pd.isna(a) or pd.isna(b) or a > b for a,b in pairs):
        return pd.NaT, pd.NaT
    return max([a for a,b in pairs] + [done]), max([b for a,b in pairs] + [done])


def select_version(records, prediction_time, historical=True):
    """Select latest event, then latest accepted version. No silent tie breaking.

    `records` is already restricted to the requested entity and feature.
    Scheduled/forecast events may be in the future but must be published before p.
    """
    p = timestamp(prediction_time)
    if pd.isna(p):
        return None, Verdict('unknown', 'missing_prediction_time')
    accepted, decisions = [], []
    for record in records:
        event = timestamp(record.get('event_time'))
        if pd.isna(event):
            decisions.append(Verdict('unknown', 'missing_event_time'))
            continue
        if historical and event > p:
            decisions.append(Verdict('reject', 'future_historical_observation'))
            continue
        result = verify(p, record.get('available_lower'), record.get('available_upper'),
                        record.get('evidence_id', ''), record.get('contradictory', False))
        decisions.append(result)
        if result.decision == 'accept':
            accepted.append(record)
    if not accepted:
        if decisions and all(v.decision == 'reject' for v in decisions):
            return None, Verdict('reject', 'all_versions_unavailable')
        return None, Verdict('unknown', 'no_proven_available_version')
    ordered = sorted(accepted, key=lambda r: (timestamp(r['event_time']), timestamp(r['available_upper'])))
    best = ordered[-1]
    ties = [r for r in ordered if (timestamp(r['event_time']), timestamp(r['available_upper'])) ==
            (timestamp(best['event_time']), timestamp(best['available_upper']))]
    if len({str(r['value']) for r in ties}) > 1:
        return None, Verdict('unknown', 'conflicting_versions_at_same_time')
    return best, Verdict('accept', 'latest_proven_available_version')
