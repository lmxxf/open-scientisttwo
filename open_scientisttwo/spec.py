"""Declarative evaluator contract, shared by orchestration and audits."""
import json, math
from pathlib import Path

DEFAULT = dict(title="Research benchmark", primary_metric="mse", higher_is_better=False,
               arxiv_query='all:"machine learning"', seed_dois=[],
               coding_guidance="Edit only model.py; preserve its public API. Never modify eval.py or data.py.",
               eval_timeout=120, metric_atol=1e-9, metric_rtol=0.0)

def load_spec(root):
    path = Path(root)/"task_spec.json"
    spec = dict(DEFAULT)
    if path.exists():
        spec.update(json.loads(path.read_text()))
    if not isinstance(spec['higher_is_better'], bool) or not isinstance(spec['seed_dois'], list):
        raise ValueError("invalid task direction or references")
    if not isinstance(spec['primary_metric'], str) or not spec['primary_metric']:
        raise ValueError("metric key required")
    for name in ('eval_timeout', 'metric_atol', 'metric_rtol'):
        if not math.isfinite(float(spec[name])) or float(spec[name]) < 0:
            raise ValueError("invalid evaluation tolerance or timeout")
    if spec['eval_timeout'] <= 0:
        raise ValueError("positive timeout required")
    return spec

def better(a, b, spec, equal=False):
    return (a >= b if equal else a > b) if spec['higher_is_better'] else (a <= b if equal else a < b)
