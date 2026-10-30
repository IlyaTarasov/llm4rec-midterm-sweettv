import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from midterm.src.eighth_pass import personal_factor

class TestPersonalExposure(unittest.TestCase):
    def test_future_sessions_and_duplicate_airings_do_not_change_profile(self):
        def schedule(channel, day):
            start=pd.Timestamp(day)
            return pd.DataFrame({'tv_show_id':[10],'channel_id':[channel],'start':[start],
                                 'duration':[60],'end':[start+pd.Timedelta(seconds=60)]})
        past=schedule(1,'2020-01-02');future=schedule(2,'2020-01-12')
        p={'user_ids':np.array([2**64-3],dtype=np.uint64),'schedules':{
            'export_arh_11-20-final.csv':past,'export_arh_21-30-final.csv':future}}
        w={'fit_start':'2020-01-01','cutoff':'2020-01-11','end':'2020-01-21','schedule':'21-30'}
        x={'users':p['user_ids'],'candidates':np.array([10])}
        seconds=lambda day:pd.Timestamp(day).value//10**9
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.npz'
            np.savez(path,uidx=np.array([0,0]),channel=np.array([1,2]),
                     start_ts=np.array([seconds('2020-01-02'),seconds('2020-01-12')]),
                     stop_ts=np.array([seconds('2020-01-03'),seconds('2020-01-13')]))
            with patch('midterm.src.eighth_pass.RAW_CACHE',path):
                before=personal_factor(p,w,x,20.)
                p['schedules']['export_arh_21-30-final.csv']=pd.concat([future,future],ignore_index=True)
                after=personal_factor(p,w,x,20.)
            self.assertTrue(np.allclose(before,after))
            self.assertAlmostEqual(before[0,0],np.sqrt(.2/.3))

if __name__=='__main__':unittest.main()
