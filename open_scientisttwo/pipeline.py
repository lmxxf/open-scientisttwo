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
    def __init__(self, root, source, fast=False, backend='codex', custom=None, peer_rounds=1, model='gpt-6.1-sol', timeout=240, limitation_rounds=None, K=None, S=4, engineering=2, ablation_count=3, ablation_refinement=1, weights=None, eval_timeout=None, reasoning_effort='low'):
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
        self.runner = CodexRunner(self.root, self.guard, backend=backend, custom=custom, model=model, timeout=timeout, reasoning_effort=reasoning_effort)
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

    def _check_integrity_if_initialized(self):
        if hasattr(self, 'lock') and hasattr(self, 'snapshots'):
            self.guard()

    def _safe_trial_snapshot(self, snap_label, config, mode):
        try:
            return config, self.snapshot(snap_label, config, mode), None
        except Exception as exc:
            self._check_integrity_if_initialized()
            return config, None, f'{type(exc).__name__}: {exc}'

    def trial(self, label, idea):
        try:
            self.code(label, idea, idea.get('parent', 'baseline_source'))
        except Exception as exc:
            self._check_integrity_if_initialized()
            self.event(2, label=label, mode='code', decision='Bad', value=None, feedback=f'{type(exc).__name__}: {exc}')
            return
        config = idea.get('config', {})
        current = None
        for mode, baseline in [('subset', self.base_subset), ('full', self.base_full)]:
            attempts = [0]
            def candidate():
                attempts[0] += 1
                snap_name = label+'_'+mode if attempts[0] == 1 else f'{label}_{mode}_try{attempts[0]}'
                cfg, val, err = self._safe_trial_snapshot(snap_name, config, mode)
                if err is not None:
                    self.execution_error = err
                    return cfg, None
                self.execution_error = None
                return cfg, val
            def critic(value):
                if value[1] is None:
                    self.feedback = 'Runtime execution failed; fix the bug in model.py/config: ' + (getattr(self, 'execution_error', '') or 'unknown error')
                    return 'Engineer'
                result = self.agent(mode+' Critic', f'Result {value}, baseline {baseline}. Return {{"decision":"Good|Bad|Engineer","feedback":"..."}}. Good requires a strict improvement in {getattr(self, "spec", {}).get("primary_metric", "mse")}; higher_is_better={getattr(self, "spec", {}).get("higher_is_better", False)}.')
                self.feedback = result.get('feedback', '')
                decision = result['decision']
                if decision == 'Good' and not better(value[1], baseline, getattr(self, 'spec', {'higher_is_better': False})):
                    return 'Bad'
                return decision
            def refine_step(v):
                try:
                    res = self.refine(label, v, mode)
                    self.execution_error = None
                    return res
                except Exception as exc:
                    self._check_integrity_if_initialized()
                    self.execution_error = f'{type(exc).__name__}: {exc}'
                    return v[0], None
            current, verdict = stage(candidate, critic, refine_step, self.engineering)
            self.event(2, label=label, mode=mode, decision=verdict, value=current[1], feedback=getattr(self, 'feedback', ''))
            if verdict != 'Good' or current[1] is None:
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
            'CRITICAL STANDALONE PAPER STYLE: Write strictly as a polished, standalone ICLR conference paper presenting ONLY the final proposed method, its self-contained theoretical guarantees, and controlled scientific experiments. '
            '1) NEVER mention internal workspace filenames, pipeline roles, config flag names, raw snake_case dictionary keys (e.g., "cma_on_anneal_on", "trend_on_ridge_on"), SHA-256 hashes, or artifact paths in the paper prose or table cells (strictly forbid writing "snapshots/pruned", "*_metrics.json", "final_test_metrics.json", "related_work.xml", "references.json", "baseline_paper.md", "task_spec.json", "mse alias", "the evaluator", "the recorded condition", "the supplied abstract", or "disabled in the evaluated configuration"). Always translate experimental conditions into human-readable English labels and present ONLY the active mechanisms of the final selected method. '
            '2) NEVER narrate the automated development search history or rejected candidate trials in the paper prose or tables (strictly forbid mentioning "initial candidate", "seed0", "evolved0", "seed1", "evolved1", "Development candidate I", "screening validation", "subset screening", "rebuttal0", "additional validation generation", or "trajectory manifest"). '
            '3) Do NOT write any Appendix section, historical trajectory table, duplicate References block inside Related Work, or "Not measured / inapplicable" disclaimers. '
            '4) STRUCTURAL REQUIREMENTS FOR TOP ICLR SCORE: '
            '   - In Abstract and Introduction, state the proven theoretical contribution and the isolated + integrated empirical gains clearly and confidently as a unified algorithmic advance. '
            '   - In Related Work, systematically compare against ALL prior works in references.json (using their abstracts) and include a concise Markdown comparison table contrasting prior methods vs. our method across mathematical formulation, asymptotic complexity, and key guarantees. '
            '   - In Methodology, provide self-contained, rigorous mathematical proofs for ALL theorems/propositions/lemmas (including any foundational criteria so no proof relies on an unproved external recurrence), and include 1-2 explicit algorithm pseudocode blocks in ```text ... ``` (`Algorithm 1`, `Algorithm 2`). '
            '   - In Experiments, present clean, well-organized Markdown summary tables (at most 6-7 columns per table) covering the benchmark evaluation studies (' + self.spec.get('metric_description', '') + ') and component subtraction ablations. Under each table, provide positive quantitative mechanism analysis and paired confidence-interval interpretation; reserve all scope caveats strictly for the Limitations section rather than adding defensive disclaimers under every table. '
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
        dois = list(self.spec['seed_dois'])
        if not self.fast:
            try:
                q_plan = self.agent('Review retrieval planner', 'Read baseline_paper.md. Return {"query":"simple arXiv keyword query without nested parentheses for prior art retrieval"}.')
                extra_xml = lookup(q_plan.get('query', ''))
                import xml.etree.ElementTree as ET, re
                ns = {'a': 'http://www.w3.org/2005/Atom'}
                for entry in ET.fromstring(extra_xml).findall('a:entry', ns):
                    aid = entry.findtext('a:id', '', ns)
                    m_id = re.search(r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})', aid, flags=re.I)
                    if m_id:
                        cand_doi = '10.48550/arXiv.' + m_id.group(1)
                        if cand_doi.casefold() not in {d.casefold() for d in dois}:
                            dois.append(cand_doi)
            except Exception:
                pass
        seed_set = {d.casefold() for d in self.spec['seed_dois']}
        self.refs = []
        for doi in dois:
            try:
                self.refs.append(reference(doi))
            except Exception:
                if doi.casefold() in seed_set:
                    raise
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
        for _ab_try in range(2):
            self.restore(self.best['snapshot'])
            ab_err_note = '' if _ab_try == 0 else f' Previous ablation config crashed with: {_last_ab_err}. Use ONLY scalar boolean/numeric config keys supported by model.py.'
            plans = self.agent('Ablation Planner', f'Read model.py and snapshots/{self.best["snapshot"]}/config.json. Best {self.best}. Return {{"ablations":[{{"config":{{}},"reason":"remove one proposed component"}},...]}}. '
                               f'Plan exactly {self.ablation_count} subtraction/control configurations using ONLY config keys and values actually supported by model.py without raising exceptions; identify which mechanism each disables.' + ab_err_note)['ablations']
            ablations = []
            if len(plans) != self.ablation_count:
                raise ValueError('ablation plan count')
            _last_ab_err = None
            for i, plan in enumerate(plans):
                self.restore(self.best['snapshot'])
                lbl_ab = f'ablation{i}' if _ab_try == 0 else f'ablation_retry_{i}'
                try:
                    ablations.append(dict(config=plan['config'], value=self.snapshot(lbl_ab, plan['config'])))
                except Exception as exc:
                    self._check_integrity_if_initialized()
                    _last_ab_err = f'{type(exc).__name__}: {exc}'
                    break
            if _last_ab_err is None:
                break
        if _last_ab_err is not None:
            raise RuntimeError(f'ablation execution failed: {_last_ab_err}')
        critique = self.agent('Ablation Critic', f'Metric={self.spec["primary_metric"]}, higher_is_better={self.spec["higher_is_better"]}. Best {self.best}; ablations {ablations}. Return {{"refine":true|false,"config":{{...}}}} using ONLY concrete boolean/numeric parameter values (never write descriptive placeholder strings like "retain setting" inside numeric config fields).')
        if critique['refine'] and self.ablation_refinement:
            self.restore(self.best['snapshot'])
            base_cfg = dict(self.best.get('config', {}))
            for k_cfg, v_cfg in critique.get('config', {}).items():
                if k_cfg in base_cfg and isinstance(base_cfg[k_cfg], (int, float)) and not isinstance(base_cfg[k_cfg], bool) and isinstance(v_cfg, str):
                    continue
                base_cfg[k_cfg] = v_cfg
            try:
                score = self.snapshot('pruned', base_cfg)
            except Exception:
                self._check_integrity_if_initialized()
                score = None
            if score is not None and better(score, self.best['value'], self.spec, equal=True):
                self.best = dict(config=base_cfg, value=score, label='pruned', snapshot='pruned')
                # Iterative Pareto Pruning: re-plan and re-run subtraction ablations around pruned;
                # if removing any single remaining component improves or matches validation score,
                # automatically promote that cleaner subtraction config as the new pruned!
                for _prune_round in range(2):
                    self.restore(self.best['snapshot'])
                    cur_snap = self.best['snapshot']
                    re_plans = self.agent('Ablation Planner', f'Read model.py and snapshots/{cur_snap}/config.json. Final selected pruned method is {self.best}. Return {{"ablations":[{{"config":{{}},"reason":"remove one active component from pruned"}},...]}}. Plan exactly {self.ablation_count} single-component subtraction configurations around snapshots/{cur_snap}/config.json using ONLY supported boolean/numeric keys.')['ablations']
                    if len(re_plans) != self.ablation_count:
                        break
                    new_ablations = []
                    ab_ok = True
                    for i, plan in enumerate(re_plans):
                        self.restore(self.best['snapshot'])
                        p_lbl = f'pruned_ablation{i}' if _prune_round == 0 else f'pruned_r{_prune_round}_ablation{i}'
                        p_cfg = dict(base_cfg)
                        for k_p, v_p in plan.get('config', {}).items():
                            if k_p in p_cfg and isinstance(p_cfg[k_p], (int, float)) and not isinstance(p_cfg[k_p], bool) and isinstance(v_p, str):
                                continue
                            p_cfg[k_p] = v_p
                        try:
                            new_ablations.append(dict(config=p_cfg, value=self.snapshot(p_lbl, p_cfg)))
                        except Exception:
                            self._check_integrity_if_initialized()
                            ab_ok = False
                            break
                    if not ab_ok or not new_ablations:
                        break
                    ablations = new_ablations
                    higher = self.spec.get('higher_is_better', False)
                    best_ab = max(ablations, key=lambda x: x['value']) if higher else min(ablations, key=lambda x: x['value'])
                    if better(best_ab['value'], self.best['value'], self.spec):
                        base_cfg = dict(best_ab['config'])
                        next_lbl = f'pruned_r{_prune_round + 1}'
                        score = self.snapshot(next_lbl, base_cfg)
                        self.best = dict(config=base_cfg, value=score, label=next_lbl, snapshot=next_lbl)
                    else:
                        break
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
        meta = self.agent('Meta Reviewer', f'Read manuscript.md and snapshots/{self.best["snapshot"]}/config.json. Verify that methodology matches snapshots/{self.best["snapshot"]}/model.py and config.json, all references in references.json are discussed, and the paper reads as a clean standalone conference article without development-history narration (no seed0/evolved0/screening trial tables in prose). Do NOT ask to alter selected config values, add trajectory tables in prose, or add self-deprecating disclaimers in the abstract. Return {{"revise":true|false,"reason":"..."}}.')
        if meta['revise']:
            self.draft(meta['reason'])
        self.guard()
        count = verify_claims(self.root, self.root/'claims.json', self.root/'manuscript.md')
        from .audits import verify_references
        verified = verify_references(self.root/'manuscript.md')
        i4_prompt = (f'Read manuscript.md and ONLY snapshots/{self.best["snapshot"]}/model.py and its config.json as selected source. '
                     f'Selected configuration {self.best}. Check whether manuscript.md faithfully describes ONLY the algorithm, formulas, and active mechanisms actually executed by model.py under config.json (do NOT fail alignment over unused metadata/dead keys in config.json as long as manuscript.md itself makes no unimplemented claims). Return {{"aligned":true|false,"evidence":"..."}}.')
        alignment = self.agent('Method-Code Auditor', i4_prompt)
        for _i4_try in range(2):
            if alignment['aligned']:
                break
            self.draft('CRITICAL I4 METHOD-CODE ALIGNMENT FIX: Ensure manuscript.md describes ONLY the exact executed code path (including any numerical regularization/fallback branches) in snapshots/' + str(self.best["snapshot"]) + '/model.py (under its config.json) without claiming any unimplemented mechanism: ' + str(alignment.get('evidence', '')))
            self.guard()
            count = verify_claims(self.root, self.root/'claims.json', self.root/'manuscript.md')
            verified = verify_references(self.root/'manuscript.md')
            alignment = self.agent('Method-Code Auditor', i4_prompt)
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
    p.add_argument('--reasoning-effort', choices=['low', 'medium', 'high', 'xhigh'], default='low')
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
    pipeline = Pipeline(scratch, Path(a.source).resolve(), a.fast_demo, a.backend, a.custom, a.max_peer_rounds, a.model, a.role_timeout, a.limitation_rounds, a.K, a.S, a.engineering_budget, a.ablation_plan_count, a.ablation_refinement, json.loads(a.review_weights) if a.review_weights else None, a.eval_timeout, a.reasoning_effort)
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
