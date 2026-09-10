"""Optional explicit refresh. The default run uses the bundled frozen data.

Systematic MONTHLY prefix sample, not a representative probability sample.
Refuses to overwrite any existing destination. No addresses are requested.
"""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import pandas as pd

FIELDS=['unique_key','created_date','closed_date','agency','complaint_type',
        'location_type','incident_zip','borough','open_data_channel_type','status']
ENDPOINT='https://data.cityofnewyork.us/resource/erm2-nwe9.csv'

def fetch(month):
    end=f"2023-{month:02d}-{calendar.monthrange(2023,month)[1]:02d}T23:59:59"
    query={'$select':','.join(FIELDS),'$where':f"created_date between '2023-{month:02d}-01T00:00:00' and '{end}'",
           '$order':'created_date,unique_key','$limit':2000}
    url=ENDPOINT+'?'+urlencode(query)
    with urlopen(url,timeout=90) as response:
        raw=response.read()
    frame=pd.read_csv(io.BytesIO(raw),dtype=str)
    if len(frame)!=2000 or not set(FIELDS).issubset(frame.columns):
        raise ValueError(f'Month {month}: unexpected response shape {frame.shape}')
    return frame,dict(month=month,url=url,rows=len(frame),response_sha256=hashlib.sha256(raw).hexdigest())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    if args.out.exists() or args.out.with_suffix('.metadata.json').exists():
        raise FileExistsError('Choose a NEW filename for a refreshed snapshot')
    with ThreadPoolExecutor(max_workers=3) as pool:
        chunks=list(pool.map(fetch,range(1,13)))
    combined=pd.concat([x[0] for x in chunks],ignore_index=True).sort_values(['created_date','unique_key'])
    if combined.unique_key.duplicated().any():
        raise ValueError('Duplicate request ids across months')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    combined.to_csv(args.out,index=False,lineterminator='\n')
    meta=dict(retrieved_at_utc=datetime.now(timezone.utc).isoformat(),year=2023,
              selection='First 2000 requests of each calendar month in created_date,unique_key order',
              representative=False,rows=len(combined),columns=FIELDS,requests=[x[1] for x in chunks],
              sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
              min_created=combined.created_date.min(),max_created=combined.created_date.max())
    args.out.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='requests'},indent=2),flush=True)

if __name__=='__main__':
    main()
