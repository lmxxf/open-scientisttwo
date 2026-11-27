import argparse,json,time,statistics
from pathlib import Path
from data import vocabulary,streams,reference
from model import IncrementalBPE

def check(raw,vocab,rules,config):
    model=IncrementalBPE(vocab,rules,config)
    for n,c in enumerate(raw,1):
        model.append(c)
        actual=model.tokens();expected=reference(raw[:n],rules)
        if actual!=expected or b''.join(vocab[t] for t in actual)!=raw[:n]:
            raise ValueError(f'exact prefix BPE mismatch at byte {n}')

def run(mode,split,extra,config):
    vocab,rules=vocabulary();texts=streams(mode,split,extra)
    for raw in texts:check(raw,vocab,rules,config)
    times=[]
    for _ in range(7):
        models=[IncrementalBPE(vocab,rules,config) for _ in texts]  # construction excluded, disclosed
        start=time.perf_counter()
        for model,raw in zip(models,texts):
            for c in raw:model.append(c)
        times.append(time.perf_counter()-start)
    seconds=statistics.median(times);size=sum(map(len,texts));latency=seconds/size*1e7
    return dict(mse=latency,latency_ms_per_10kb=latency,throughput_mb_s=size/seconds/1e6,
                latency_us_per_kb=seconds/size*1e9,bytes=size,prefixes_checked=size,exact=True,
                timing_repeats=7,mode=mode,split=split,extra=extra)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['subset','full'],required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--extra',action='store_true');p.add_argument('--config',default='config.json');a=p.parse_args()
    print(json.dumps(run(a.mode,a.split,a.extra,json.loads(Path(a.config).read_text()))))
