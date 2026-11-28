import hashlib, json, math, re, sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
from .runner import execute
from .spec import load_spec


def hashes(root):
    return {name: hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
            for name in ('eval.py', 'data.py', 'task_spec.json') if (Path(root)/name).exists()}


def protect(root, expected):
    if hashes(root) != expected:
        raise RuntimeError('I2 protected evaluator changed')


def evaluate(root, mode='full', split='validation', extra=False, timeout=None):
    spec = load_spec(root)
    cmd = [sys.executable, '-I', '-B', '-c',
           "import sys,runpy;sys.path.insert(0,'.');sys.argv=['eval.py']+sys.argv[1:];runpy.run_path('eval.py',run_name='__main__')",
           '--mode', mode, '--split', split]
    if extra:
        cmd.append('--extra')
    result = json.loads(execute(cmd, root, timeout or spec['eval_timeout']))
    if not math.isfinite(result[spec['primary_metric']]):
        raise ValueError('nonfinite score')
    return result


_LOOKUP_CACHE = {}
_REF_CACHE = {}


def lookup(query):
    if query in _LOOKUP_CACHE:
        return _LOOKUP_CACHE[query]
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': query, 'max_results': 2})
    last_err = None
    for attempt in range(4):
        try:
            res = execute(['curl', '--fail', '--silent', '--show-error', '--retry', '3', '--retry-delay', '2', '--max-time', '40', url], '/tmp', 90)
            _LOOKUP_CACHE[query] = res
            return res
        except Exception as exc:
            last_err = exc
            __import__('time').sleep(1.5 * (attempt + 1))
    raise last_err


def reference(doi):
    key = doi.casefold()
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    if key.startswith('10.48550/arxiv.'):
        identifier = doi[len('10.48550/arXiv.'):]
        tree = ET.fromstring(lookup('id:'+identifier))
        ns = {'a':'http://www.w3.org/2005/Atom'}
        entry = tree.find('a:entry', ns)
        if entry is None or identifier not in entry.findtext('a:id', '', ns):
            raise ValueError('arXiv reference not found')
        authors = [a.findtext('a:name', '', ns).strip() for a in entry.findall('a:author', ns) if a.findtext('a:name', '', ns)]
        pub = (entry.findtext('a:published', '', ns) or '')[:4]
        res = dict(doi=doi, title=' '.join(entry.findtext('a:title', '', ns).split()), url='https://arxiv.org/abs/'+identifier,
                   authors=', '.join(authors[:4]) + (' et al.' if len(authors) > 4 else ''), year=pub)
        _REF_CACHE[key] = res
        return res
    url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='')
    with urllib.request.urlopen(url, timeout=40) as response:
        message = json.load(response)['message']
    c_authors = [' '.join(filter(None, [a.get('given', ''), a.get('family', '')])).strip() for a in message.get('author', [])]
    c_year = str(message.get('issued', {}).get('date-parts', [['']])[0][0] or '')
    res = {'doi': message['DOI'], 'title': message['title'][0], 'url': url,
           'authors': ', '.join(c_authors[:4]) + (' et al.' if len(c_authors) > 4 else ''), 'year': c_year}
    _REF_CACHE[key] = res
    return res


def verify_claims(root, manifest, manuscript):
    records = json.loads(Path(manifest).read_text())
    text = Path(manuscript).read_text()
    for claim in records:
        candidate = Path(root)/claim['snapshot']
        if hashlib.sha256((candidate/'model.py').read_bytes()).hexdigest() != claim['model_sha256']:
            raise ValueError('snapshot changed')
        for name, digest in claim.get('files', {}).items():
            if hashlib.sha256((candidate/name).read_bytes()).hexdigest() != digest:
                raise ValueError('snapshot file changed')
        spec = load_spec(candidate)
        metric = claim.get('metric', spec['primary_metric'])
        actual = evaluate(candidate, claim['mode'], claim['split'], claim['extra'], claim.get('eval_timeout'))
        expected = claim.get('value', claim.get(metric))
        if not math.isclose(actual[metric], expected, abs_tol=spec['metric_atol'], rel_tol=spec['metric_rtol']):
            raise ValueError('I1 score mismatch')
        row = f"| {claim['label']} | {expected:.9f} |"
        if text.count(row) != 1:
            raise ValueError('I1 manuscript table mismatch')
    table_text = text.split('\n| Experiment |', 1)[1] if '\n| Experiment |' in text else text
    rows = re.findall(r'^\| ([^|]+) \| (-?[0-9.]+) \|$', table_text, re.M)
    if len(rows) != len(records):
        raise ValueError('unmanifested metric rows')
    return len(records)


DIMENSIONS = ['Originality', 'Importance', 'Claims_Supported', 'Soundness_of_Experiments',
              'Clarity_of_Writing', 'Value_to_Community', 'Contextualization']


class StanfordAgenticReviewer:
    """Local approximation only. Official fitted regression coefficients are undisclosed."""
    def __init__(self, weights=None, intercept=0):
        self.weights = weights or dict.fromkeys(DIMENSIONS, 1/7)
        if set(self.weights) != set(DIMENSIONS) or any(not math.isfinite(float(v)) for v in self.weights.values()):
            raise ValueError('seven weights required')
        self.intercept = intercept

    def review(self, runner, context):
        query = runner.role('Review retrieval planner', 'Read manuscript.md. Return {"query":"simple arXiv keyword query without nested parentheses grounded in this manuscript"}.')['query']
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        papers = []
        for q in (query, load_spec(runner.root).get('arxiv_query', 'all:optimization')):
            try:
                xml = lookup(q)
                tree = ET.fromstring(xml)
                papers = [dict(title=e.findtext('a:title', namespaces=ns), id=e.findtext('a:id', namespaces=ns),
                               abstract=e.findtext('a:summary', namespaces=ns)) for e in tree.findall('a:entry', ns)]
                if papers:
                    query = q
                    break
            except Exception:
                continue
        if not papers:
            raise ValueError('review retrieval returned no metadata')
        (runner.root/'review_retrieval.json').write_text(json.dumps(dict(query=query, papers=papers), indent=2))
        result = runner.role('independent local reviewer',
            'Read manuscript.md and references.json. No reviewer gaming. Score each dimension 1-10 '
            'with detailed comments. Assess originality against the retrieved literature and actual contribution. Return '
            '{"scores": {dimension: number}, "comments": "..."}. Dimensions: '+str(DIMENSIONS)+context+' Read review_retrieval.json for actual retrieved metadata.')
        scores = result['scores']
        if set(scores) != set(DIMENSIONS) or any(not 1 <= float(v) <= 10 for v in scores.values()):
            raise ValueError('invalid review')
        score = self.intercept + sum(self.weights[k]*float(scores[k]) for k in DIMENSIONS)
        return dict(**result, total=score, decision='Accept' if score >= 5 else 'Reject',
                    weights=self.weights, calibration='unknown official coefficients; equal-average proxy by default; uncalibrated local approximation; NOT official Stanford score')


def verify_references(manuscript):
    """Audit every actual bibliography entry, including unexpected additions."""
    text = Path(manuscript).read_text()
    if '## References\n' not in text:
        raise ValueError('missing bibliography')
    ref_block = text.split('## References\n', 1)[1].split('\n## ', 1)[0]
    entries = ref_block.strip().splitlines()
    verified = []
    for entry in entries:
        match = re.fullmatch(r'- (.+)\. DOI: (10\.\S+)', entry)
        if not match:
            raise ValueError('unverifiable bibliography entry')
        actual = reference(match[2])
        normalize = lambda x: re.sub(r'\W+', '', x).casefold()
        if normalize(match[1]) != normalize(actual['title']) or match[2].casefold() != actual['doi'].casefold():
            raise ValueError('I3 title/DOI mismatch')
        verified.append(actual)
    # Empty bibliography is explicit and left for contextualization review, never fabricated.
    return verified
