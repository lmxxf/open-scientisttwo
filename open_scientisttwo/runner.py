"""Bounded CLI roles. Persist only prompts, commands and final answers, never session logs."""
import json, os, re, signal, subprocess, time
from pathlib import Path


def execute(command, cwd, timeout=120, input=None):
    proc = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = proc.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise TimeoutError(f'command timed out: {command[0]}')
    if proc.returncode:
        detail = (err or out or '').strip()[-500:]
        raise RuntimeError(f'{command[0]} failed (exit {proc.returncode})' + (f': {detail}' if detail else ''))
    return out


def extract(text):
    text = text.strip()
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        def _fix_escape(m):
            slashes, nxt = m.group(1), m.group(2)
            if len(slashes) % 2 == 1 and nxt not in '"\\/bfnrtu':
                return slashes + '\\' + nxt
            return m.group(0)
        return json.loads(re.sub(r'(\\+)(.)', _fix_escape, text))


class CodexRunner:
    def __init__(self, workdir, guard, backend='codex', model='gpt-6.1-sol', timeout=240, custom=None):
        self.root, self.guard = Path(workdir), guard
        self.backend, self.model, self.timeout, self.custom = backend, model, timeout, custom
        self.count = 0

    def role(self, role, instruction):
        self.guard()
        self.count += 1
        output = self.root / f'role_{self.count:03d}.json'
        policy = 'Edit ONLY model.py; configuration belongs in returned JSON. ' if role in ('Coding Agent', 'Engineering refiner') else 'Read-only role: do not edit any files. '
        prompt = (policy + f'Role: {role}. Read AGENTS.md. Work only in current workspace. '
                  'Do not modify evaluator or data loader. Return ONLY a JSON object. ' + instruction)
        if self.backend == 'codex':
            cmd = ['codex', 'exec', '--dangerously-bypass-approvals-and-sandbox',
                   '--skip-git-repo-check', '-C', str(self.root), '-m', self.model,
                   '-c', 'model_reasoning_effort="low"', '-o', str(output), prompt]
        elif self.backend == 'claude':
            cmd = ['claude', '-p', prompt]
        else:
            if not self.custom:
                raise ValueError('custom backend requires command argument list')
            cmd = self.custom + [prompt]
        started = time.time()
        answer = ''
        error = None
        status = 'failed'
        try:
            stdout = execute(cmd, self.root, self.timeout)
            answer = output.read_text() if self.backend == 'codex' else stdout
            result = extract(answer)
            if not isinstance(result, dict):
                raise ValueError('role must return object')
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            status = 'ok'
            return result
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            with (self.root / 'trace.jsonl').open('a') as f:
                f.write(json.dumps(dict(role=role, command=cmd, status=status,
                                       seconds=round(time.time()-started, 2), prompt=prompt, answer=answer, error=error))+'\n')
            try:
                self.guard()
            except Exception as exc:
                with (self.root / 'trace.jsonl').open('a') as f:
                    f.write(json.dumps(dict(role=role, status='integrity_failed', seconds=round(time.time()-started, 2), error=dict(type=type(exc).__name__, message=str(exc))))+'\n')
                raise
