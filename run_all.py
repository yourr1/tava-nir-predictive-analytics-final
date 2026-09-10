"""Run the complete clean research package OFFLINE with no manual path edits."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'rerun_results',
                        help='NEW output folder. Bundled reference results are never overwritten.')
    parser.add_argument('--only',choices=['benchmark','pipeline','matching','nyc'])
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():
        raise FileExistsError(f'Output exists; choose a new --out path: {out}')
    out.mkdir(parents=True)
    env=dict(os.environ,PYTHONUTF8='1',PYTHONHASHSEED='42',PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def call(name,cmd):
        start=time.monotonic()
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}. See {out/name}.log')
        return dict(step=name,seconds=round(time.monotonic()-start,3),exit_code=code)
    status=[]
    started=datetime.now(timezone.utc).isoformat()
    status.append(call('00_tests',[sys.executable,'-m','unittest','discover','-s','tests','-v']))
    commands={
        'benchmark':['-c',f"from src.benchmark import run; run({str(ROOT/'data/benchmark_v4')!r},{str(out/'01_benchmark')!r})"],
        'pipeline':['-c',f"from src.pipeline import run; run({str(out/'02_pipeline')!r})"],
        'matching':['-m','src.matching','--data-dir',str(ROOT/'data/valentine'),'--outdir',str(out/'03_matching')],
        'nyc':['-c',f"from src.nyc import run; run({str(ROOT/'data/nyc311_monthly_2023.csv')!r},{str(out/'04_nyc')!r})"]}
    for name,args_cmd in commands.items():
        if args.only is None or args.only==name:
            status.append(call(name,[sys.executable,*args_cmd]))
    inputs=[]
    source_files=[*ROOT.glob('*.py'),*ROOT.glob('requirements*.txt'),*(ROOT/'src').glob('*.py'),*(ROOT/'tests').glob('*.py')]
    for file in sorted([*(ROOT/'data').rglob('*'),*source_files]):
        if file.is_file():
            inputs.append(dict(path=file.relative_to(ROOT).as_posix(),bytes=file.stat().st_size,
                               sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    package_names=['numpy','pandas','scipy','scikit-learn','rapidfuzz']
    record=dict(started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
                python=sys.version,platform=platform.platform(),
                dependencies={p:importlib.metadata.version(p) for p in package_names},
                scope=args.only or 'all',steps=status,inputs=inputs)
    (out/'run_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'SUCCESS: {out}',flush=True)

if __name__=='__main__':
    main()
