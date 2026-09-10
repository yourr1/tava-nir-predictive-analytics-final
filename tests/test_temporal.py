import unittest
import pandas as pd
from src.temporal import verify, derive_bounds, aggregate_bounds, select_version
from src.benchmark import metrics, explicit_label

class TemporalTests(unittest.TestCase):
    def test_exact_boundary(self):
        self.assertEqual(verify('2020-01-02','2020-01-02','2020-01-02','doc').decision,'accept')
    def test_future(self):
        self.assertEqual(verify('2020-01-02','2020-01-03','2020-01-03','doc').decision,'reject')
    def test_straddling(self):
        self.assertEqual(verify('2020-01-02','2020-01-01','2020-01-03','doc').decision,'unknown')
    def test_missing_evidence(self):
        self.assertEqual(verify('2020-01-02','2020-01-01','2020-01-01').decision,'unknown')
    def test_missing_date(self):
        self.assertEqual(verify('2020-01-02',None,None,'doc').decision,'unknown')
    def test_missing_prediction(self):
        self.assertEqual(verify(None,'2020-01-01','2020-01-01','doc').decision,'unknown')
    def test_contradiction(self):
        self.assertEqual(verify('2020-01-02','2020-01-01','2020-01-01','doc',True).decision,'unknown')
    def test_inverted(self):
        self.assertEqual(verify('2020-01-02','2020-01-03','2020-01-01','doc').decision,'unknown')
    def test_timezone(self):
        self.assertEqual(verify('2020-01-02T12:00:00+03:00','2020-01-02T09:00:00Z','2020-01-02T09:00:00Z','doc').decision,'accept')
    def test_delay_interval(self):
        lo,hi=derive_bounds('2020-01-01',1,3,'contract')
        self.assertEqual(verify('2020-01-03',lo,hi,'contract').decision,'unknown')
    def test_missing_delay_contract(self):
        self.assertTrue(all(pd.isna(t) for t in derive_bounds('2020-01-01',1,3,'')))
    def test_aggregate(self):
        lo,hi=aggregate_bounds([('2020-01-01','2020-01-01'),('2020-01-03','2020-01-03')],'2020-01-03')
        self.assertEqual(verify('2020-01-02',lo,hi,'lineage').decision,'reject')
    def test_aggregate_unknown(self):
        self.assertTrue(all(pd.isna(x) for x in aggregate_bounds([(None,None)],'2020-01-01')))
    def record(self,value=1,available='2020-01-01',event='2020-01-01'):
        return dict(value=value,event_time=event,available_lower=available,available_upper=available,evidence_id='doc')
    def test_backdated_correction(self):
        selected,v=select_version([self.record(1),self.record(9,'2020-01-03')],'2020-01-02')
        self.assertEqual(selected['value'],1)
    def test_scheduled_future(self):
        rec=self.record(event='2020-01-05')
        self.assertEqual(select_version([rec],'2020-01-02',False)[1].decision,'accept')
        self.assertEqual(select_version([rec],'2020-01-02',True)[1].decision,'reject')
    def test_conflicting_versions(self):
        self.assertEqual(select_version([self.record(1),self.record(2)],'2020-01-02')[1].decision,'unknown')
    def test_empty(self):
        self.assertEqual(select_version([],'2020-01-02')[1].decision,'unknown')
    def test_no_unknown_reward(self):
        d=pd.DataFrame({'gt':['temporal_invalid','temporal_invalid','valid'],'p':['temporal_invalid','unknown','valid']})
        m=metrics(d,'p')
        self.assertEqual(m['recall_invalid_all'],.5)
        self.assertEqual(m['f1_invalid_decided'],1.)
    def test_mixed_event_invalid(self):
        b=pd.DataFrame({'entity_id':[1,2],'prediction_time':['2020-01-02']*2})
        d=pd.DataFrame({'entity_id':[2,1],'event_time':['2020-01-03','2020-01-01']})
        self.assertEqual(explicit_label(b,d,'event_time'),'temporal_invalid')
    def test_missing_row_unknown(self):
        b=pd.DataFrame({'entity_id':[1,2],'prediction_time':['2020-01-02']*2})
        d=pd.DataFrame({'entity_id':[1],'available_time':['2020-01-01']})
        self.assertEqual(explicit_label(b,d),'unknown')

if __name__=='__main__':
    unittest.main()
