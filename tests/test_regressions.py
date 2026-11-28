import json, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from open_scientisttwo.upload import Client
from open_scientisttwo.pipeline import Pipeline, stage
from open_scientisttwo.runner import CodexRunner
from open_scientisttwo.audits import verify_references

class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.s3_status, self.s3_body, self.confirm = 201, b'<PostResponse><Key>x</Key></PostResponse>', {'success':True,'token':'real-token'}
        self.limited = False
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                if self.path == '/s3':
                    status, data = owner.s3_status, owner.s3_body
                elif self.path.endswith('get-upload-url'):
                    status, data = 200, json.dumps(dict(success=True, presigned_url=owner.base+'/s3', s3_key='x', presigned_fields={'key':'x'})).encode()
                else:
                    status, data = 202, json.dumps(owner.confirm).encode()
                self.send_response(status); self.end_headers(); self.wfile.write(data)
            def do_GET(self):
                self.send_response(429 if owner.limited else 202); self.end_headers(); self.wfile.write(b'{"detail":"processing"}')
        self.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.base = 'http://127.0.0.1:'+str(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.tmp = tempfile.TemporaryDirectory(); self.pdf=Path(self.tmp.name)/'paper.pdf'; self.pdf.write_bytes(b'%PDF-1.4 test')
        self.client=Client(self.base)
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.tmp.cleanup()
    def test_xml_s3_and_confirmation(self):
        self.assertEqual(self.client.submit(self.pdf,dry_run=False)['token'],'real-token')
    def test_empty_s3(self):
        self.s3_status,self.s3_body=204,b''
        self.assertEqual(self.client.submit(self.pdf,dry_run=False)['status'],202)
    def test_s3_failure(self):
        self.s3_status=403
        with self.assertRaises(Exception): self.client.submit(self.pdf,dry_run=False)
    def test_confirmation_validation(self):
        for result in ({'success':False,'token':'t'},{'success':True},{'success':True,'token':''}):
            self.confirm=result
            with self.assertRaises(ValueError): self.client.submit(self.pdf,dry_run=False)
    def test_pending_and_rate_limit(self):
        self.assertTrue(self.client.review('token')['pending'])
        self.limited=True
        with self.assertRaisesRegex(RuntimeError,'429'): self.client.review('token')

    def test_cli_persists_token_before_query_failure(self):
        import submit_paperreview_ai as cli
        output=Path(self.tmp.name)/'receipt.json'
        self.limited=True
        with patch.object(cli,'Client',return_value=self.client), patch('sys.argv',['submit',str(self.pdf),'--poll','1','--interval','0','--output',str(output)]):
            with self.assertRaisesRegex(RuntimeError,'429'):cli.main()
        self.assertEqual(json.loads(output.read_text())['token'],'real-token')
    def test_cli_dry_run_and_query(self):
        import submit_paperreview_ai as cli
        output=Path(self.tmp.name)/'receipt.json'
        with patch.object(cli,'Client',return_value=self.client),patch('sys.argv',['submit',str(self.pdf),'--dry-run','--output',str(output)]),patch('builtins.print'):
            cli.main()
        self.assertTrue(json.loads(output.read_text())['dry_run'])
        with patch.object(cli,'Client',return_value=self.client),patch('sys.argv',['submit','--token','t','--poll','1','--interval','0','--output',str(output)]),patch('builtins.print'):
            cli.main()
        self.assertTrue(json.loads(output.read_text())['query']['pending'])

class IntegrityTests(unittest.TestCase):
    def test_stage_exhaustion_invalid_and_feedback(self):
        self.assertEqual(stage(lambda:0,lambda x:'Engineer',lambda x:x+1,1),(1,'Bad'))
        with self.assertRaises(ValueError): stage(lambda:0,lambda x:'unknown',lambda x:x,1)
        with self.assertRaises(ValueError): stage(lambda:0,lambda x:'Good',lambda x:x,-1)
    def test_actual_reference_extra_and_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'paper';p.write_text('## References\n- Real title. DOI: 10.1/a\n')
            with patch('open_scientisttwo.audits.reference',return_value={'title':'Real title','doi':'10.1/a'}):
                self.assertEqual(len(verify_references(p)),1)
                p.write_text(p.read_text()+'- Invented title. DOI: 10.1/a\n')
                with self.assertRaises(ValueError): verify_references(p)
    def test_failure_trace_precedes_guard(self):
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def guard():
                calls.append(1)
                if len(calls)>1: raise RuntimeError('guard failure')
            r=CodexRunner(d,guard,backend='custom',custom=['python','-c','import time;time.sleep(2)'],timeout=.03)
            with self.assertRaisesRegex(RuntimeError,'guard failure'):r.role('test','public prompt')
            records=[json.loads(line) for line in (Path(d)/'trace.jsonl').read_text().splitlines()]
            trace=records[0]
            self.assertEqual(records[-1]['status'],'integrity_failed')
            self.assertEqual(trace['status'],'failed');self.assertEqual(trace['error']['type'],'TimeoutError')
            self.assertGreater(trace['seconds'],0);self.assertIn('public prompt',trace['prompt'])
    def test_restore_selected_and_baseline_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Pipeline(d,Path('examples/toy_ridge_ood'),True)
            baseline=(Path(d)/'model.py').read_bytes()
            (Path(d)/'model.py').write_text('changed')
            p.restore('baseline_source');self.assertEqual((Path(d)/'model.py').read_bytes(),baseline)
            (Path(d)/'snapshots/baseline_source/model.py').write_text('tamper')
            with self.assertRaises(RuntimeError):p.guard()
    def test_full_engineering_budget_and_feedback(self):
        p=Pipeline.__new__(Pipeline)
        p.base_subset,p.base_full,p.successes,p.claims,p.engineering=10.,8.,[],[{'label':'full'}],1
        p.code=lambda *a:None
        p.event=lambda *a,**kw:None
        scores=iter([9.,9.,7.])
        p.snapshot=lambda *a,**kw:next(scores)
        decisions=iter([{'decision':'Good'},{'decision':'Engineer','feedback':'full-specific feedback'},{'decision':'Good'}])
        p.agent=lambda *a:next(decisions)
        received=[]
        def refine(label,value,mode):
            received.append((mode,p.feedback))
            return {},p.snapshot('engineered',{},mode)
        p.refine=refine
        p.trial('x',{'config':{}})
        self.assertEqual(received,[('full','full-specific feedback')])
        self.assertEqual(p.successes[0]['value'],7.)
    def test_candidate_resets_before_coding(self):
        with tempfile.TemporaryDirectory() as d:
            p=Pipeline(d,Path('examples/toy_ridge_ood'),True)
            baseline=(Path(d)/'model.py').read_text()
            starts=[]
            def agent(role,prompt):
                starts.append((Path(d)/'model.py').read_text())
                (Path(d)/'model.py').write_text(baseline+'\n# candidate change\n')
                return {}
            p.agent=agent
            p.code('first',{})
            p.snapshot('first',{})
            p.code('second',{})
            self.assertEqual(starts,[baseline,baseline])
    def test_zero_success_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Pipeline(d,Path('examples/toy_ridge_ood'),True,K=0)
            def role(name,prompt):
                return {'Limitation Extractor':{'limitations':['training contamination']},
                        'Limitation Verifier':{'complete':True},
                        'Seed Generator and Novelty Checker':{'ideas':[{'config':{},'novelty':0},{'config':{},'novelty':0}]}}[name]
            p.agent=role;p.trial=lambda *a:None
            with patch('open_scientisttwo.pipeline.lookup',return_value='<feed/>'),patch('open_scientisttwo.pipeline.reference',return_value={'doi':'10.1/x','title':'Real'}):
                with self.assertRaisesRegex(RuntimeError,'zero successful'):p.run()
            self.assertFalse(json.loads((Path(d)/'smoke_receipt.json').read_text())['success'])
    def test_invalid_budgets(self):
        for kwargs in ({'K':-1},{'engineering':-1},{'S':0},{'ablation_count':2},{'peer_rounds':3},{'ablation_refinement':2}):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):Pipeline(d,Path('examples/toy_ridge_ood'),**kwargs)
    def test_latex_braces_and_backslashes(self):
        import shutil, subprocess
        compiler=shutil.which('xelatex') or shutil.which('pdflatex')
        if not compiler:self.skipTest('no TeX compiler installed')
        with tempfile.TemporaryDirectory() as d:
            p=Pipeline(d,Path('examples/toy_ridge_ood'),True)
            p.best={'snapshot':'baseline_source'}
            p.refs=[{'title':'A reference','doi':'10.1/x'}]
            p.agent=lambda *a:{'method':r'coef={x_i}; weight=min(1, delta/abs(r)); path\name & 10% # $ ^ ~','limitations':'Toy only.'}
            p.draft()
            result=subprocess.run([compiler,'-interaction=nonstopmode','-halt-on-error','manuscript.tex'],cwd=d,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout[-2000:])
            self.assertTrue((Path(d)/'manuscript.pdf').exists())

    def test_trial_runtime_error_recovers_or_marks_bad(self):
        p=Pipeline.__new__(Pipeline)
        p.base_subset,p.base_full,p.successes,p.claims,p.engineering=10.,8.,[],[{'label':'full_ok'}],1
        p.code=lambda *a:None
        events=[]
        p.event=lambda *a,**kw:events.append(kw)
        calls=[0]
        def snap(label,cfg,mode):
            calls[0]+=1
            if calls[0]==1:
                raise RuntimeError('TypeError: int() got list')
            return 7. if mode=='full' else 9.
        p.snapshot=snap
        p.agent=lambda *a:{'decision':'Good'}
        p.refine=lambda label,value,mode:({'fixed':True},p.snapshot(label+'_fixed',{'fixed':True},mode))
        p.trial('recover',{'config':{}})
        self.assertEqual(len(p.successes),1)
        self.assertEqual(p.successes[0]['value'],7.)
        p.snapshot=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('always crash'))
        p.refine=lambda label,value,mode:(_ for _ in ()).throw(RuntimeError('still crash'))
        p.trial('crash_branch',{'config':{}})
        self.assertEqual(len(p.successes),1)
        self.assertEqual(events[-1]['decision'],'Bad')

    def test_extract_unescaped_latex_backslash(self):
        from open_scientisttwo.runner import extract
        raw = '{"method": "Differential $dy = Z_A\\,dw$ and \\\\top"}'
        parsed = extract(raw)
        self.assertEqual(parsed['method'], r'Differential $dy = Z_A\,dw$ and \top')
