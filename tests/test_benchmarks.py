import importlib.util,json,math,os,subprocess,sys,tempfile,unittest,shutil
from pathlib import Path
from unittest.mock import patch
from open_scientisttwo.audits import evaluate,hashes,protect,verify_claims
from open_scientisttwo.spec import better,load_spec
from open_scientisttwo.manuscript import write_manuscript
from open_scientisttwo.pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[1]
def run_eval(name,*args):
    folder=ROOT/'benchmarks'/name
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    p=subprocess.run([sys.executable,'-B','eval.py','--mode','subset',*args],cwd=folder,env=env,capture_output=True,text=True,timeout=30)
    if p.returncode:raise AssertionError(p.stderr)
    return json.loads(p.stdout)

class Benchmarks(unittest.TestCase):
    def test_bpe_baseline(self):
        r=run_eval('inc_bpe');self.assertTrue(r['exact']);self.assertEqual(r['bytes'],r['prefixes_checked']);self.assertGreater(r['throughput_mb_s'],0)
    def test_bpe_wrong_tokens_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(ROOT/'benchmarks/inc_bpe',d,dirs_exist_ok=True)
            Path(d,'model.py').write_text('class IncrementalBPE:\n def __init__(self,*a):pass\n def append(self,b):return 0\n def tokens(self):return [0]\n')
            with self.assertRaises(RuntimeError):evaluate(d,'subset')
    def test_bpe_priority_and_all_prefixes(self):
        script='''from data import vocabulary,reference
from model import IncrementalBPE
import random
v,r=vocabulary();rng=random.Random(42)
assert reference(b'aaa',[(97,97)])==[256,97]
for raw in [b'aaaaa',bytes(range(256)), '中文 def encode(x): return x'.encode(),bytes(rng.randrange(256) for _ in range(200))]:
 m=IncrementalBPE(v,r,{})
 for n,b in enumerate(raw,1):
  m.append(b);assert m.tokens()==reference(raw[:n],r),(n,m.tokens())
'''
        subprocess.run([sys.executable,'-c',script],cwd=ROOT/'benchmarks/inc_bpe',check=True,timeout=20)
    def test_pinet_baseline(self):
        r=run_eval('pinet_hcnn');self.assertTrue(math.isfinite(r['mse']));self.assertLess(r['eq_cv_max'],1e-10);self.assertGreaterEqual(r['cv_max'],0)
    def test_pinet_kkt_reference_and_bad_constraints(self):
        script='''from data import instances,objective
from eval import assess
import numpy as np
for p,ref,cert in instances('subset','test',True):
 r=assess(p,ref,ref,cert);assert r['cv']<1e-10 and r['rs']==0
 bad=assess(p,ref+10,ref,cert);assert bad['cv']>1
 try:assess(p,np.full_like(ref,np.nan),ref,cert)
 except ValueError:pass
 else:raise AssertionError('nonfinite accepted')
'''
        subprocess.run([sys.executable,'-c',script],cwd=ROOT/'benchmarks/pinet_hcnn',check=True,timeout=20)
    def test_hash_lock_both_tasks(self):
        for name in ('inc_bpe','pinet_hcnn'):
            with tempfile.TemporaryDirectory() as d:
                shutil.copytree(ROOT/'benchmarks'/name,d,dirs_exist_ok=True)
                lock=hashes(d);Path(d,'data.py').write_text('tampered')
                with self.assertRaises(RuntimeError):protect(d,lock)
    def test_metric_direction(self):
        for higher in (False,True):
            spec={'higher_is_better':higher};self.assertEqual(better(2,1,spec),higher);self.assertFalse(better(1,1,spec));self.assertTrue(better(1,1,spec,True))
    def test_custom_metric_evaluation_and_claims(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);snap=root/'s';snap.mkdir()
            (snap/'eval.py').write_text('import json;print(json.dumps({"speedup":2.5}))')
            (snap/'data.py').write_text('');(snap/'model.py').write_text('')
            (snap/'task_spec.json').write_text(json.dumps(dict(primary_metric='speedup',higher_is_better=True)))
            self.assertEqual(evaluate(snap)['speedup'],2.5)
            import hashlib
            claims=[dict(snapshot='s',label='a',mode='full',split='test',extra=False,metric='speedup',value=2.5,model_sha256=hashlib.sha256(b'').hexdigest())]
            (root/'claims.json').write_text(json.dumps(claims));(root/'paper.md').write_text('| a | 2.500000000 |')
            self.assertEqual(verify_claims(root,root/'claims.json',root/'paper.md'),1)
    def test_higher_better_trial(self):
        p=Pipeline.__new__(Pipeline);p.spec={'primary_metric':'speedup','higher_is_better':True};p.base_subset=p.base_full=1.;p.successes=[];p.claims=[{'label':'full'}];p.engineering=0
        p.code=lambda *a:None;p.event=lambda *a,**k:None;p.agent=lambda *a:{'decision':'Good'};p.snapshot=lambda *a:2.
        p.trial('x',{'config':{}});self.assertEqual(p.successes[0]['value'],2.)
    def test_structured_pdf_native_and_fallback(self):
        response=dict(title='A method',abstract='A measured study.',introduction='Problem and contribution. '*90,related_work='Verified context. '*60,
                      methodology=r'Projection $y=y_p+Vw$ and $$M y=b$$.',experiments='Measured validation and ablation results. '*100,limitations='Finite evidence. '*40)
        if not shutil.which('pdflatex'):self.skipTest('pdflatex absent')
        with tempfile.TemporaryDirectory() as d:
            write_manuscript(d,response,[],[],{'title':'Task','primary_metric':'speedup'})
            self.assertFalse(json.loads(Path(d,'pdf_build.json').read_text())['fallback']);self.assertTrue(Path(d,'manuscript.pdf').exists())
            self.assertIn('$y=y_p+Vw$',Path(d,'manuscript.tex').read_text())
            response['methodology']=r'Broken math $\notarealcommand{y}$ and Unicode δ 中文.'
            write_manuscript(d,response,[],[],{'title':'Task','primary_metric':'speedup'})
            self.assertTrue(json.loads(Path(d,'pdf_build.json').read_text())['fallback'])
