import unittest
import pandas as pd
from src.nyc import temporal_split

class LabelMaturityTests(unittest.TestCase):
    def test_old_short_snapshot_cannot_be_used(self):
        p=pd.date_range('2023-01-01',periods=40,freq='h')
        data=pd.DataFrame({'created_date':p,'label_available_at':p+pd.Timedelta(days=7)})
        with self.assertRaises(ValueError):
            temporal_split(data,p[28],p[-1]+pd.Timedelta(hours=1))
    def test_purge_boundary_strict(self):
        p=pd.date_range('2023-01-01',periods=31,freq='D')
        d=pd.DataFrame({'created_date':p,'label_available_at':p+pd.Timedelta(days=7)})
        train,test=temporal_split(d,pd.Timestamp('2023-01-20'),pd.Timestamp('2023-02-01'))
        self.assertEqual(int(train.sum()),12)
        self.assertEqual(int(test.sum()),12)
        self.assertTrue(d.loc[train,'label_available_at'].max()<d.loc[test,'created_date'].min())

if __name__=='__main__':
    unittest.main()
