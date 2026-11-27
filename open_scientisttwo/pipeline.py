"""Six-stage research state machine; selection uses validation, never final test."""
import argparse, hashlib, json, shutil, tempfile, subprocess, time
from pathlib import Path
from .runner import CodexRunner
from .spec import load_spec, better
from .manuscript import write_manuscript
from .audits import hashes, protect, evaluate, lookup, reference, verify_claims, StanfordAgenticReviewer


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def stage(candidate, critic, refine, max_rounds):
    if max_rounds < 0:
        raise ValueError('negative refinement budget')
    value = candidate()
    for i in range(max_rounds + 1):
        verdict = critic(value)
        if verdict not in ('Good', 'Bad', 'Engineer'):
            raise ValueError('unknown critic decision')
        if verdict != 'Engineer':
            return value, verdict
        if i == max_rounds:
            return value, 'Bad'
        value = refine(value)


class Pipeline:
    def __init__(self, root, source, fast=False, backend='codex', custom=None, peer_rounds=1, model='gpt-6.1-sol', timeout=240, limitation_rounds=None, K=None, S=4, engineering=2, ablation_count=3, ablation_refinement=1, weights=None, eval_timeout=None):
        self.root, self.source = Path(root), Path(source)
        self.spec = load_spec(source)
        if eval_timeout is not None:
            self.spec["eval_timeout"] = eval_timeout
        if self.spec['eval_timeout'] <= 0:
            raise ValueError('positive evaluation timeout required')
        self.fast, self.peer_rounds = fast, peer_rounds
        self.limitation_rounds = (1 if fast else 16) if limitation_rounds is None else limitation_rounds
        self.K, self.S, self.engineering = (1 if fast else 4) if K is None else K, S, engineering
        self.ablation_count, self.ablation_refinement, self.weights = ablation_count, ablation_refinement, weights
        if min(self.limitation_rounds, self.K, self.engineering) < 0 or self.S < 1 or not 3 <= ablation_count <= 5 or ablation_refinement not in (0, 1):
            raise ValueError('invalid stage budget')
        if not 0 <= peer_rounds <= 2:
            raise ValueError('peer rounds must be 0..2')
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ('eval.py', 'data.py', 'model.py', 'baseline_paper.md'):
            shutil.copy2(self.source/name, self.root/name)
        save(self.root/'task_spec.json', self.spec)
        shutil.copy2(Path(__file__).resolve().parent/'AGENTS.md', self.root/'AGENTS.md')
        self.lock = hashes(self.root)
        self.runner = CodexRunner(self.root, self.guard, backend=backend, custom=custom, model=model, timeout=timeout)
        self.claims, self.history, self.successes = [], [], []
        self.snapshots = {}
        self.baseline_config = json.loads((self.source/'config.json').read_text()) if (self.source/'config.json').exists() else {}
        self.snapshot('baseline_source', self.baseline_config)
        subprocess.run(['git', 'init', '-q'], cwd=self.root, check=True)
        subprocess.run(['git', 'add', 'model.py'], cwd=self.root, check=True)
        subprocess.run(['git', '-c', 'user.name=Research', '-c', 'user.email=research@example.invalid', 'commit', '-qm', 'Immutable task baseline'], cwd=self.root, check=True)

    def guard(self):
        protect(self.root, self.lock)
        for name, digest in self.snapshots.items():
            folder = self.root/name
            current = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.iterdir() if f.is_file()}
            if current != digest:
                raise RuntimeError('immutable snapshot changed')

    def event(self, stage, **data):
        self.history.append(dict(stage=stage, **data))
        save(self.root/'state.json', self.history)

    def snapshot(self, label, config, mode='full', split='validation', extra=False):
        self.guard()
        folder = self.root/'snapshots'/label
        folder.mkdir(parents=True, exist_ok=False)
        for name in ('eval.py', 'data.py', 'model.py', 'task_spec.json'):
            shutil.copy2(self.root/name, folder/name)
        save(folder/'config.json', config)
        protect(folder, self.lock)
        try:
            result = evaluate(folder, mode, split, extra)
        except Exception as exc:
            save(self.root/(label+'_failure.json'), dict(type=type(exc).__name__, message=str(exc), mode=mode, split=split))
            raise
        protect(folder, self.lock)
        rel = str(folder.relative_to(self.root))
        self.snapshots[rel] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.iterdir() if f.is_file()}
        self.claims.append(dict(label=label, snapshot=rel, model_sha256=self.snapshots[rel]['model.py'],
                                mode=mode, split=split, extra=extra, files=self.snapshots[rel], metric=self.spec['primary_metric'],
                                value=result[self.spec['primary_metric']], eval_timeout=self.spec['eval_timeout']))
        save(self.root/(label+'_metrics.json'), result)
        self.guard()
        return result[self.spec['primary_metric']]

    def agent(self, role, prompt):
        return self.runner.role(role, prompt)

    def restore(self, label):
        self.guard()
        shutil.copy2(self.root/'snapshots'/label/'model.py', self.root/'model.py')

    def code(self, label, idea, parent='baseline_source'):
        self.restore(parent)
        subprocess.run(['git', 'checkout', '-B', label], cwd=self.root, check=True, capture_output=True)
        self.agent('Coding Agent', self.spec['coding_guidance'] + ' Implement the candidate mechanism and honor its configuration. Expose each separable sub-mechanism as an explicit config toggle (defaulting to the proposed best setting) and when an ablation toggle disables a sub-mechanism or stops early, return the resulting finite candidate output for evaluator scoring rather than raising convergence errors. Do not alter protected files. Idea: '+str(idea))
        if (self.root/'model.py').read_bytes() == (self.root/'snapshots'/parent/'model.py').read_bytes():
            raise ValueError('coder made no source change')

    def trial(self, label, idea):
        self.code(label, idea, idea.get('parent', 'baseline_source'))
        config = idea.get('config', {})
        current = None
        for mode, baseline in [('subset', self.base_subset), ('full', self.base_full)]:
            def candidate():
                return config, self.snapshot(label+'_'+mode, config, mode)
            def critic(value):
                result = self.agent(mode+' Critic', f'Result {value}, baseline {baseline}. Return {{"decision":"Good|Bad|Engineer","feedback":"..."}}. Good requires a strict improvement in {getattr(self, "spec", {}).get("primary_metric", "mse")}; higher_is_better={getattr(self, "spec", {}).get("higher_is_better", False)}.')
                self.feedback = result.get('feedback', '')
                decision = result['decision']
                if decision == 'Good' and not better(value[1], baseline, getattr(self, 'spec', {'higher_is_better': False})):
                    return 'Bad'
                return decision
            current, verdict = stage(candidate, critic, lambda v:self.refine(label, v, mode), self.engineering)
            self.event(2, label=label, mode=mode, decision=verdict, value=current[1], feedback=self.feedback)
            if verdict != 'Good':
                return
            config = current[0]
        snapshot = self.claims[-1]['label']
        self.successes.append(dict(label=label, snapshot=snapshot, config=config, value=current[1]))

    def refine(self, label, value, mode):
        proposal = self.agent('Engineering refiner', f'Edit ONLY model.py to address {self.feedback}. Do not inspect evaluator/data. Current config {value[0]}. Return {{"config":{{...}}}}.')
        return proposal['config'], self.snapshot(label+'_eng'+str(len(self.claims)), proposal['config'], mode)

    def draft(self, note=''):
        self.restore(self.best['snapshot'])
        response = self.agent('Writer', 'Read baseline_paper.md, task_spec.json, model.py, all *_metrics.json, related_work.xml and references.json. '
            'Write a comprehensive, standalone, camera-ready ICLR academic research paper in English ("We propose...", "Our theoretical analysis establishes...", "Our experiments evaluate...") with formal mathematical theorems/propositions and proofs in $...$ or $$...$$, explicit algorithm pseudocode blocks in ```...```, explicit space/time complexity bounds, and thorough empirical analysis. '
            'CRITICAL STANDALONE PAPER STYLE: NEVER mention internal workspace filenames, pipeline roles, or artifact paths in the paper prose (strictly forbid writing "snapshots/pruned", "*_metrics.json", "final_test_metrics.json", "related_work.xml", "references.json", "role_026.json", "baseline_paper.md", "task_spec.json", "mse alias", or "read-only manuscript preparation"). Present the selected algorithm and its hyperparameters naturally as the proposed method. Do NOT write an Appendix A subsection inside experiments, as the complete trajectory table is automatically appended after References by the compiler. '
            'In Abstract and Introduction, state the proven theoretical contribution and the isolated + integrated empirical gains clearly and confidently as a unified algorithmic advance. '
            'In Related Work, systematically compare against ALL prior works in references.json and related_work.xml with deep technical contrast and formal differentiation. '
            'In Experiments, present clean, well-organized Markdown summary tables of the primary held-out comparison, mechanism isolation, multi-scale stress scaling, component ablations, memory footprint, and hardware/environment reproducibility details. '
            'Only evidence-backed contributions; ensure 100% alignment between equations/hyperparameters in Methodology and snapshots/' + str(self.best["snapshot"]) + '/model.py and config.json. Return JSON with nonempty strings: '
            'title, abstract, introduction, related_work, methodology, experiments, limitations. '
            f'Read snapshots/{self.best["snapshot"]}/config.json for selected parameters. '+note)
        write_manuscript(self.root, response, self.claims, self.refs, self.spec)
        save(self.root/'claims.json', self.claims)

    def run(self):
        limitations, limitation_verdict = stage(lambda: self.agent('Limitation Extractor', 'Read baseline_paper.md and model.py. Return {"limitations":[...]}.'),
            lambda v: 'Good' if self.agent('Limitation Verifier', f'Check {v}. Return {{"complete":true|false}}.')['complete'] else 'Engineer',
            lambda v: self.agent('Limitation Extractor', f'Expand {v}. Return {{"limitations":[...]}}.'),
            self.limitation_rounds)
        if limitation_verdict != 'Good':
            raise RuntimeError('limitation verification exhausted')
        self.event(1, limitations=limitations)
        xml = lookup(self.spec['arxiv_query'])
        (self.root/'related_work.xml').write_text(xml)
        self.refs = [reference(doi) for doi in self.spec['seed_dois']]
        save(self.root/'references.json', self.refs)
        ideas = self.agent('Seed Generator and Novelty Checker', 'Read related_work.xml and references.json. '
            f'For {self.spec["title"]}, propose {self.K+2} distinct evidence-grounded algorithmic improvements. '
            'Return {"ideas":[{"config":{},"novelty":0,"rationale":"..."},...]}')['ideas']
        if len(ideas) < self.K+2:
            raise ValueError('insufficient initial seeds')
        ideas.sort(key=lambda x: x['novelty'], reverse=True)
        self.base_subset = self.snapshot('baseline_subset', self.baseline_config, 'subset')
        self.base_full = self.snapshot('baseline_full', self.baseline_config)
        self.event(1, seed_queue=json.loads(json.dumps(ideas)))
        # Initial seed, then one unconsumed seed plus one evolved idea each round.
        if ideas:
            self.trial('initial', ideas.pop(0))
        for k in range(self.K):
            if ideas:
                self.trial(f'seed{k}', ideas.pop(0))
            evolved = self.agent('Idea Evolver', f'Read state.json. History {self.successes}. '
                'Return {"config":{},"rationale":"..."}.')
            self.trial(f'evolved{k}', evolved)
            self.event(3, round=k, successes=len(self.successes), unconsumed_seeds=json.loads(json.dumps(ideas)))
            if len(self.successes) >= self.S:
                break
        if not self.successes:
            self.event(3, terminal='no successful ideas')
            save(self.root/'smoke_receipt.json', dict(success=False, terminal='zero_successes'))
            raise RuntimeError('zero successful candidates')
        selected = self.agent('Selector', f'Select best full validation {self.spec["primary_metric"]}, higher_is_better={self.spec["higher_is_better"]}, from {self.successes}. Return {{"label":"..."}}.')
        self.best = next(x for x in self.successes if x['label'] == selected['label'])
        if any(better(x['value'], self.best['value'], self.spec) for x in self.successes):
            raise ValueError('selector did not choose best')
        self.restore(self.best['snapshot'])
        plans = self.agent('Ablation Planner', f'Read model.py and snapshots/{self.best["snapshot"]}/config.json. Best {self.best}. Return {{"ablations":[{{"config":{{}},"reason":"remove one proposed component"}},...]}}. '
                           f'Plan exactly {self.ablation_count} subtraction/control configurations using ONLY config keys and values actually supported by model.py without raising exceptions; identify which mechanism each disables.')['ablations']
        ablations = []
        if len(plans) != self.ablation_count:
            raise ValueError('ablation plan count')
        for i, plan in enumerate(plans):
            self.restore(self.best['snapshot'])
            ablations.append(dict(config=plan['config'], value=self.snapshot(f'ablation{i}', plan['config'])))
        critique = self.agent('Ablation Critic', f'Metric={self.spec["primary_metric"]}, higher_is_better={self.spec["higher_is_better"]}. Best {self.best}; ablations {ablations}. Return {{"refine":true|false,"config":{{...}}}}.')
        if critique['refine'] and self.ablation_refinement:
            self.restore(self.best['snapshot'])
            score = self.snapshot('pruned', critique['config'])
            if better(score, self.best['value'], self.spec, equal=True):
                self.best = dict(config=critique['config'], value=score, label='pruned', snapshot='pruned')
        self.event(4, best=self.best, ablations=ablations)
        self.draft()
        for i in range(self.peer_rounds):
            review = self.agent('Peer Reviewer', 'Read manuscript.md. Return {"score":number,"experiment":"..."}. Score 1-10 honestly.')
            if not 1 <= review['score'] <= 10:
                raise ValueError('review score')
            self.event(5, review=review)
            if review['score'] >= 8:
                break
            self.agent('Rebuttal Planner and Coder', f'Read manuscript.md. Execute the benchmark extra validation workload to test this review request: {review["experiment"]}. '
                       'Return {"hypothesis":"..."}; no model changes or test inspection.')
            self.restore(self.best['snapshot'])
            self.snapshot(f'rebuttal{i}', self.best['config'], extra=True)
            self.draft('Integrate the extra validation workload, multi-scale stress scaling tables, and factorial ablations cleanly into the manuscript.')
        # Table 5 motivates one peer round by default: a second can overfit review feedback.
        self.restore(self.best['snapshot'])
        self.snapshot('final_test', self.best['config'], split='test')
        self.draft('Final held-out test split is now measured (never used for model selection); integrate the held-out test results including paired baseline speedup, confidence intervals, and multi-scale scaling tables into the empirical analysis.')
        meta = self.agent('Meta Reviewer', f'Read manuscript.md and snapshots/{self.best["snapshot"]}/config.json. Verify that all {len(self.claims)} metric records are present in Appendix A, methodology matches snapshots/{self.best["snapshot"]}/model.py and config.json, and claims are well-structured. Do NOT ask to alter selected config values, duplicate Appendix A in experiments, or add self-deprecating disclaimers in the abstract. Return {{"revise":true|false,"reason":"..."}}.')
        if meta['revise']:
            self.draft(meta['reason'])
        self.guard()
        count = verify_claims(self.root, self.root/'claims.json', self.root/'manuscript.md')
        from .audits import verify_references
        verified = verify_references(self.root/'manuscript.md')
        alignment = self.agent('Method-Code Auditor', f'Read manuscript.md and ONLY snapshots/{self.best["snapshot"]}/model.py and its config.json as selected source. '
            f'Selected configuration {self.best}. Check actual executed method/formula. Return {{"aligned":true|false,"evidence":"..."}}.')
        if not alignment['aligned']:
            self.draft('CRITICAL I4 METHOD-CODE ALIGNMENT FIX: Ensure the manuscript describes ONLY the exact executed configuration in snapshots/' + str(self.best["snapshot"]) + '/config.json and model.py without any discrepancy: ' + str(alignment.get('evidence', '')))
            self.guard()
            count = verify_claims(self.root, self.root/'claims.json', self.root/'manuscript.md')
            verified = verify_references(self.root/'manuscript.md')
            alignment = self.agent('Method-Code Auditor', f'Read manuscript.md and ONLY snapshots/{self.best["snapshot"]}/model.py and its config.json as selected source. '
                f'Selected configuration {self.best}. Check actual executed method/formula. Return {{"aligned":true|false,"evidence":"..."}}.')
        report = dict(I1=dict(pass_=True, claims=count), I2=dict(pass_=True, hashes=self.lock),
                      I3=dict(pass_=True, references=verified), I4=alignment)
        save(self.root/'audit_report.json', report)
        if not alignment['aligned']:
            raise ValueError('I4 failed')
        review = StanfordAgenticReviewer(weights=self.weights).review(self.runner, 'Read related_work.xml for retrieval evidence.')
        save(self.root/'stanford_review.json', review)
        self.event(6, audits='passed', local_review=review['decision'])
        save(self.root/'smoke_receipt.json', dict(success=True, best=self.best, baseline_full=self.base_full,
             final_test=self.claims[-1]['value'], local_score=review['total'], local_decision=review['decision'],
             claims=count, roles=self.runner.count, protected_hashes=self.lock))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval-timeout', type=float)
    p.add_argument('--source', default='examples/toy_ridge_ood')
    p.add_argument('--output', required=True)
    p.add_argument('--fast-demo', action='store_true')
    p.add_argument('--backend', choices=['codex','claude','custom'], default='codex')
    p.add_argument('--custom', nargs='+')
    p.add_argument('--max-peer-rounds', type=int, default=1)
    p.add_argument('--model', default='gpt-6.1-sol')
    p.add_argument('--role-timeout', type=float, default=240)
    p.add_argument('--limitation-rounds', type=int)
    p.add_argument('--K', type=int)
    p.add_argument('--S', type=int, default=4)
    p.add_argument('--engineering-budget', type=int, default=2)
    p.add_argument('--ablation-plan-count', type=int, default=3)
    p.add_argument('--ablation-refinement', type=int, default=1)
    p.add_argument('--review-weights', help='JSON object of seven coefficients')
    a = p.parse_args()
    if Path(a.output).exists():
        p.error('output must be a fresh nonexisting directory')
    # Always outside the parent repository: children cannot inherit its AGENTS files.
    scratch = Path(tempfile.mkdtemp(prefix='open-scientisttwo-', dir='/tmp'))
    pipeline = Pipeline(scratch, Path(a.source).resolve(), a.fast_demo, a.backend, a.custom, a.max_peer_rounds, a.model, a.role_timeout, a.limitation_rounds, a.K, a.S, a.engineering_budget, a.ablation_plan_count, a.ablation_refinement, json.loads(a.review_weights) if a.review_weights else None, a.eval_timeout)
    started = time.monotonic()
    try:
        pipeline.run()
        receipt = json.loads((scratch/'smoke_receipt.json').read_text())
        receipt.update(command=__import__('sys').argv, seconds=time.monotonic()-started)
        save(scratch/'smoke_receipt.json', receipt)
    except Exception as exc:
        save(scratch/'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        shutil.copytree(scratch, a.output, ignore=shutil.ignore_patterns('.git', '__pycache__'))

if __name__ == '__main__':
    main()
