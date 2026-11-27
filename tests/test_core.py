import hashlib, json, tempfile, unittest, urllib.error
from pathlib import Path
from open_scientisttwo.pipeline import stage
from open_scientisttwo.audits import hashes, protect, evaluate, verify_claims
from open_scientisttwo.runner import execute
from open_scientisttwo.upload import Client, multipart
import shutil

class Core(unittest.TestCase):
    def test_stage(self):
        seen=[]
        result=stage(lambda:0, lambda x:'Good' if x==2 else 'Engineer', lambda x:seen.append(x) or x+1,2)
        self.assertEqual(result,(2,'Good')); self.assertEqual(seen,[0,1])
        self.assertEqual(stage(lambda:0,lambda x:'Bad',lambda x:1,2),(0,'Bad'))

    def test_snapshot_and_guard(self):
        from open_scientisttwo.pipeline import Pipeline
        with tempfile.TemporaryDirectory() as d:
            p=Pipeline(d,Path('examples/toy_ridge_ood'),True)
            value=p.snapshot('base',{})
            self.assertGreater(value,0)
            p.guard()
            (Path(d)/'snapshots/base/config.json').write_text('{"iterations":20}')
            with self.assertRaises(RuntimeError):p.guard()

    def test_trial_decisions(self):
        from open_scientisttwo.pipeline import Pipeline
        p = Pipeline.__new__(Pipeline)
        p.base_subset, p.base_full, p.successes, p.claims = 10., 8., [], [{'label':'mock'}]
        p.code=lambda *a:None
        p.engineering=2
        events=[]
        p.event=lambda *a,**kw:events.append(kw)
        scores=iter([9.,7.])
        p.snapshot=lambda *a,**kw:next(scores)
        p.agent=lambda *a,**kw:{'decision':'Good'}
        p.trial('good',{'config':{}})
        self.assertEqual(p.successes[0]['value'],7.)
        p.agent=lambda *a,**kw:{'decision':'Bad'}
        p.snapshot=lambda *a,**kw:1.
        p.trial('bad',{'config':{}})
        self.assertEqual(len(p.successes),1)
        p.agent=lambda *a,**kw:{'decision':'Engineer'}
        calls=[]
        p.refine=lambda label,value,mode:calls.append(label) or value
        p.trial('engineer',{'config':{}})
        self.assertEqual(len(calls),2)
        self.assertEqual(events[-1]['decision'],'Bad')

    def test_timeout(self):
        with self.assertRaises(TimeoutError):
            execute(['python','-c','import time;time.sleep(3)'],'/tmp',.05)

    def test_audits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); snap=root/'snap'; snap.mkdir()
            for name in ['eval.py','data.py','model.py']:
                shutil.copy2(Path('examples/toy_ridge_ood')/name,snap/name)
            (snap/'config.json').write_text('{}')
            lock=hashes(snap); protect(snap,lock)
            metric=evaluate(snap)['mse']; self.assertEqual(metric,evaluate(snap)['mse'])
            claim=dict(snapshot='snap',label='a',model_sha256=hashlib.sha256((snap/'model.py').read_bytes()).hexdigest(),mode='full',split='validation',extra=False,mse=metric)
            (root/'claims.json').write_text(json.dumps([claim])); (root/'paper.md').write_text(f'| a | {metric:.9f} |')
            self.assertEqual(verify_claims(root,root/'claims.json',root/'paper.md'),1)
            (root/'paper.md').write_text('| a | 99.0 |')
            with self.assertRaises(ValueError):verify_claims(root,root/'claims.json',root/'paper.md')
            (snap/'eval.py').write_text('tamper')
            with self.assertRaises(RuntimeError):protect(snap,lock)

    def test_upload(self):
        class Response:
            def __init__(self,status,data):self.status=status;self.data=data
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps(self.data).encode()
        calls=[]
        responses=[Response(200,dict(success=True,presigned_url='https://storage.invalid',s3_key='k',presigned_fields={'key':'k','policy':'p'})),Response(204,{}),Response(202,{'success':True,'token':'t'}),Response(202,{'detail':'processing'})]
        def opener(req,**kw):calls.append(req);return responses.pop(0)
        with tempfile.TemporaryDirectory() as d:
            pdf=Path(d)/'a.pdf';pdf.write_bytes(b'%PDF-1.4 test')
            client=Client(opener=opener)
            self.assertTrue(client.submit(pdf)['dry_run']);self.assertFalse(calls)
            self.assertEqual(client.submit(pdf,dry_run=False)['status'],202)
            self.assertTrue(client.review('t')['pending'])
            body=calls[1].data
            self.assertLess(body.index(b'name="policy"'),body.index(b'name="file"'))
            self.assertEqual(calls[2].full_url,'https://paperreview.ai/api/confirm-upload')
        def limited(req,**kw):raise urllib.error.HTTPError(req.full_url,429,'rate',{},None)
        with self.assertRaisesRegex(RuntimeError,'429'):Client(opener=limited).review('t')

if __name__=='__main__':unittest.main()
