import unittest
import numpy as np
import pandas as pd
from midterm.src.temporal_opportunity import union_intervals,split_bins,profiles,schedule_bins

class TestTemporalOpportunity(unittest.TestCase):
    def test_hour_midnight_weekend_union(self):
        seconds=lambda s:pd.Timestamp(s).value//10**9
        start=seconds('2020-01-03 23:30');stop=seconds('2020-01-04 00:30')
        u,s,e=union_intervals(np.array([0,0]),np.array([start,start]),np.array([stop,stop]))
        b=split_bins(u,s,e,1)
        self.assertEqual(b.sum(),3600);self.assertEqual(b[0,23],1800);self.assertEqual(b[0,24],1800)
    def test_cutoff_empty_history_and_airing_duplicates(self):
        lo=pd.Timestamp('2020-01-01').value//10**9;hi=lo+86400
        raw={'uidx':np.array([0,0]),'start_ts':np.array([lo,hi-10]),'stop_ts':np.array([lo+3600,hi+10])}
        a=profiles(raw,lo,hi,2)
        b=profiles({'uidx':raw['uidx'][:1],'start_ts':raw['start_ts'][:1],'stop_ts':raw['stop_ts'][:1]},lo,hi,2)
        self.assertTrue(np.array_equal(a[48],b[48]));self.assertTrue(np.isfinite(a[48][1]).all())
        self.assertGreater(a[48][1,0],0) # empty user gets training-only prior
        start=pd.Timestamp(lo+1800,unit='s')
        schedule=pd.DataFrame({'tv_show_id':[10],'channel_id':[1],'start':[start],'duration':[3600]})
        one=schedule_bins(schedule,lo,hi,np.array([10]))
        two=schedule_bins(pd.concat([schedule,schedule]),lo,hi,np.array([10]))
        self.assertTrue(np.array_equal(one,two));self.assertEqual(one.sum(),1)

if __name__=='__main__':unittest.main()
