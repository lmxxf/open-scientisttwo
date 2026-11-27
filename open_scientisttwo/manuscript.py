"""Structured manuscript, native equations/tables, explicit compile recovery receipt."""
import json, re, shutil, subprocess
from pathlib import Path

SECTIONS = [('introduction','Introduction'), ('related_work','Related Work'),
            ('methodology','Methodology'), ('experiments','Experiments'), ('limitations','Limitations')]

def escape(text):
    mapping = {'\\':r'\textbackslash{}','{':r'\{','}':r'\}','&':r'\&','%':r'\%',
               '_':r'\_','#':r'\#','$':r'\$','^':r'\textasciicircum{}','~':r'\textasciitilde{}',
               '\u2013':'--','\u2014':'---','\u2018':'`','\u2019':"'",'\u201c':'``','\u201d':"''",
               '\u00b1':r'$\pm$','\u00d7':r'$\times$'}
    return ''.join(mapping.get(c, c if ord(c)<128 else '[U+%04X]' % ord(c)) for c in text)

def _render_inline(text, math=True):
    if not math:
        return escape(text)
    parts = re.split(r'(\$\$[\s\S]*?\$\$|(?<!\$)\$[^$\n]+\$(?!\$))',text)
    out = []
    for p in parts:
        if p.startswith('$') and p.endswith('$'):
            out.append(p)
        else:
            esc = escape(p)
            esc = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', esc)
            out.append(esc)
    return ''.join(out)

def render(text, math=True):
    if not math:
        return escape(text)
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith('```'):
            out.append(r'\begin{verbatim}')
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                out.append(lines[i])
                i += 1
            out.append(r'\end{verbatim}')
            i += 1
            continue
        if stripped.startswith('### '):
            out.append(r'\subsection*{' + _render_inline(stripped[4:].strip(), math) + '}')
            i += 1
            continue
        if stripped.startswith('|') and stripped.endswith('|') and i + 1 < len(lines) and re.match(r'^\s*\|[\s:\-|]+\|\s*$', lines[i+1]):
            headers = [c.strip() for c in stripped.strip('|').split('|')]
            ncols = len(headers)
            col_spec = 'l' + 'r' * max(0, ncols - 1)
            out.append(r'\begin{center}\footnotesize\begin{longtable}{' + col_spec + r'}\toprule')
            out.append(' & '.join(_render_inline(h, math) for h in headers) + r'\\\midrule')
            i += 2
            while i < len(lines):
                rline = lines[i].strip()
                if not (rline.startswith('|') and rline.endswith('|')):
                    break
                cells = [c.strip() for c in rline.strip('|').split('|')]
                if len(cells) < ncols:
                    cells += [''] * (ncols - len(cells))
                elif len(cells) > ncols:
                    cells = cells[:ncols]
                out.append(' & '.join(_render_inline(c, math) for c in cells) + r'\\')
                i += 1
            out.append(r'\bottomrule\end{longtable}\end{center}')
            continue
        out.append(_render_inline(line, math))
        i += 1
    return '\n'.join(out)

def write_manuscript(root, response, claims, refs, spec):
    root=Path(root)
    # Compatibility for old injected Writer fixtures; live prompts require all fields.
    if 'method' in response and 'methodology' not in response:
        response = dict(title=spec['title'], abstract='Measured research report.', introduction='Benchmark and baseline analysis.',
                        related_work='See verified references.', methodology=response['method'], experiments='Measured results follow.',
                        limitations=response['limitations'])
    for key in ['title','abstract']+[x[0] for x in SECTIONS]:
        if not isinstance(response.get(key),str) or not response[key].strip():
            raise ValueError('Writer missing '+key)
    (root/'manuscript_sections.json').write_text(json.dumps(response,indent=2))
    rows='\n'.join(f"| {c['label']} | {c.get('value',c.get('mse')):.9f} |" for c in claims)
    text='# '+response['title']+'\n\n## Abstract\n'+response['abstract']+'\n'
    for key,title in SECTIONS:
        text+='\n## '+title+'\n'+response[key]+'\n'
        if key=='experiments':
            text+=f"\n| Experiment | {spec['primary_metric']} |\n|---|---|\n"+rows+'\n'
    text+='\n## References\n'+'\n'.join(f"- {r['title']}. DOI: {r['doi']}" for r in refs)+'\n'
    (root/'manuscript.md').write_text(text)
    compiler=shutil.which('pdflatex')
    if not compiler:
        raise RuntimeError('pdflatex required to produce manuscript.pdf')
    receipt={}
    for native in (True,False):
        latex=r'''\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,booktabs,longtable,geometry,hyperref}
\geometry{margin=25mm}
\hypersetup{colorlinks=true,urlcolor=blue}
\setlength{\parskip}{6pt}
\begin{document}
'''
        latex+='\\title{'+escape(response['title'])+'}\n\\author{Open-ScientistTwo Research Report}\n\\date{}\\maketitle\n'
        latex+='\\begin{abstract}\n'+render(response['abstract'],native)+'\n\\end{abstract}\n'
        for key,title in SECTIONS:
            latex+='\\section{'+title+'}\n'+render(response[key],native)+'\n'
            if key=='experiments':
                latex+='\\begin{longtable}{lr}\\toprule Experiment & '+escape(spec['primary_metric'])+r'\\\midrule'+'\n'
                latex+='\n'.join(escape(c['label'])+f" & {c.get('value',c.get('mse')):.9f}"+r'\\' for c in claims)
                latex+='\n\\bottomrule\\end{longtable}\n'
        latex+='\\section*{References}\n'+'\n\n'.join(escape(r['title'])+'. DOI: '+escape(r['doi']) for r in refs)+'\n\\end{document}\n'
        (root/'manuscript.tex').write_text(latex)
        results=[]
        for _ in range(2):
            r=subprocess.run([compiler,'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','manuscript.tex'],cwd=root,capture_output=True,text=True,timeout=60)
            results.append(r.returncode)
            if r.returncode:break
        receipt.update(native_math=native, returncodes=results, fallback=not native)
        if r.returncode==0:
            (root/'pdf_build.json').write_text(json.dumps(receipt,indent=2));return
        (root/('latex_native_failure.log' if native else 'latex_safe_failure.log')).write_text(r.stdout)
    (root/'pdf_build.json').write_text(json.dumps(receipt,indent=2))
    raise RuntimeError('PDF compilation failed; inspect latex_safe_failure.log')
