"""Independent standard-library verification of the single final upload file."""
import csv,json,hashlib
from datetime import datetime
from .config import resolve

def main():
    root=resolve('midterm');data=root/'data/sweettv-tv-program-recommender';out=root/'submission_final_confidence_integrated_mf.csv'
    with (data/'submission.csv').open(newline='',encoding='utf-8') as f:template=list(csv.DictReader(f))
    with out.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);header=reader.fieldnames;rows=list(reader)
    assert header==['user_id','tv_show_id']
    assert len(rows)==1260 and len({r['user_id'] for r in rows})==1260
    assert [r['user_id'] for r in rows]==[r['user_id'] for r in template]
    allowed=set();lo=datetime(2020,7,27);hi=datetime(2020,10,20)
    with (data/'export_arh_31-42-final.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            item=int(row['tv_show_id'])
            if item and lo<=datetime.strptime(row['start_time'],'%d.%m.%Y %H:%M:%S')<hi:allowed.add(item)
    for row in rows:
        assert row['user_id'].isdigit() and str(int(row['user_id']))==row['user_id']
        ids=row['tv_show_id'].split(' ')
        assert len(ids)==5 and len(set(ids))==5
        assert all(i.isdigit() and int(i)>0 and int(i) in allowed for i in ids)
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    manifest=json.loads((root/'results/continuation/final_confidence_manifest.json').read_text())
    assert digest==manifest['csv_sha256'] and manifest['deterministic_regeneration_match']
    for name,expected in manifest['prior_submitted_hashes'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
    result={'rows':len(rows),'candidate_ids_from_source_schedule':len(allowed),'schema':header,'string_exact_template_ids':True,'five_unique_positive_eligible_ids':True,'sha256':digest,'all_four_prior_hashes_unchanged':True}
    (root/'results/continuation/final_independent_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
