#!/usr/bin/env python3
"""Join the metamorphic run with the ViUniT reference set and compare selectors.

Usage: python butterfly/metamorphic_eval/export_metamorphic_eval.py <merged_dir> <eval_dir>
Reads  <merged>/dataset.jsonl (real_correct, ViUniT fields, baselines) and
       <eval>/chunk_*/run/data.p (metamorphic unit_test_results from generate.py).
Writes <eval>/comparison.jsonl and <eval>/summary.md.

Metamorphic selection follows ViUniT: pick the program(s) with the highest mean test score; ties
count as a uniform random choice among the tied programs (expected accuracy). Questions with no
eligible transform (listed in no_test_ids.txt) are all-tied, i.e. they score as random.
"""
import json
import pickle
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/work/11808/shumera/ls6/projects/butterfly')
from accuracy_fns import GQA_accuracy  # noqa: E402
from unit_test_generation.metamorphic.baselines import SKIP, resolve_answer, score_expected  # noqa: E402

merged, ev = Path(sys.argv[1]), Path(sys.argv[2])


def program_scores(d, n):
    """Per-program mean over scored tests, re-resolving rules with the current resolve_answer.

    Skipped tests (`unchecked`, or a mapping with no key for the program's baseline answer) are left out of
    the mean. A failed baseline counts as 0. A program with nothing scored takes the mean of the others
    (all tied if none scored), so it neither wins nor loses by default.
    """
    results, meta, rules = d['unit_test_results'][0], d['unit_test_metadata'], d['selected_unit_tests']
    base = d['original_program_outputs'][0]
    out, n_skipped = [], 0
    for p in range(n):
        scores = []
        for t, (_, rule) in enumerate(rules):
            r = results[p]['results'][t]
            if r.get('skipped'):                        # already skipped by the pipeline
                n_skipped += 1
                continue
            if r.get('baseline_error') or r.get('failed'):
                scores.append(0.0)
                continue
            if t >= len(meta) or meta[t].get('source') != 'metamorphic':
                scores.append(r['acc'])
                continue
            expected = resolve_answer(rule, base[p]['output'])
            if expected is SKIP:
                n_skipped += 1
                continue
            sub = r['results'][0]
            scores.append(0.0 if sub.get('error') else float(score_expected(sub['output'], expected, GQA_accuracy)))
        out.append(st.mean(scores) if scores else None)
    valid = [x for x in out if x is not None]
    fill = st.mean(valid) if valid else 0.0
    return [fill if x is None else x for x in out], n_skipped
rows = [json.loads(l) for l in open(merged / 'dataset.jsonl')]
by_q = {r['questionId']: r for r in rows}

meta = {}
for f in sorted(ev.glob('chunk_*/run/data.p')):
    for d in pickle.load(open(f, 'rb')):
        meta[d['questionId']] = d
no_test = set((ev / 'no_test_ids.txt').read_text().split()) if (ev / 'no_test_ids.txt').exists() else set()
missing = [q for q in by_q if q not in meta and q in set((ev / 'eval_ids.txt').read_text().split())]
if missing:
    print(f'WARNING: {len(missing)} evaluated questions have no result yet (e.g. {missing[:3]})')

out = []
for q, r in by_q.items():
    if 'real_correct' not in r:
        continue
    correct = r['real_correct']
    row = {'questionId': q, 'group': r['group'], 'question': r['question'], 'answer': r['answer'],
           'real_correct': correct, 'viunit_selected': r['viunit_selected'],
           'baselines': dict(r['baselines'])}
    d = meta.get(q)
    if d is not None and d.get('unit_test_results', {}).get(0):
        res = d['unit_test_results'][0]
        scores, n_skipped = program_scores(d, len(correct))
        top = max(scores)
        sel = [i for i, s in enumerate(scores) if s == top]
        row.update(has_tests=True, n_tests=len(d['selected_unit_tests']), n_skipped=n_skipped,
                   transformations=[m.get('transformation') for m in d.get('unit_test_metadata', [])],
                   test_descriptions=[t[0] for t in d['selected_unit_tests']],
                   metamorphic_scores=scores, metamorphic_selected=sel)
        row['baselines']['metamorphic'] = sum(correct[i] for i in sel) / len(sel)
    else:
        row.update(has_tests=False, n_tests=0, metamorphic_selected=list(range(len(correct))))
        row['baselines']['metamorphic'] = sum(correct) / len(correct)   # no evidence -> random
    out.append(row)

with open(ev / 'comparison.jsonl', 'w') as f:
    for row in out:
        f.write(json.dumps(row) + '\n')


def mean(key, subset):
    return st.mean(r['baselines'][key] for r in subset)


def paired(a, b, subset, n=5000, seed=0):
    diffs = [r['baselines'][a] - r['baselines'][b] for r in subset]
    rng = random.Random(seed)
    boots = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return st.mean(diffs), boots[int(.025 * n)], boots[int(.975 * n) - 1]


keys = ['random', 'majority_vote', 'viunit', 'metamorphic', 'oracle']
lines = ['# Metamorphic vs ViUniT on the frozen candidate pool', '']
with_tests = [r for r in out if r['has_tests']]
informative = [r for r in out if len(set(r['real_correct'])) > 1]
lines += [f'- questions: {len(out)}; with >=1 transform test: {len(with_tests)}; '
          f'no test (scored as random): {len(out) - len(with_tests)}',
          f'- questions where candidates differ in correctness (selection matters): {len(informative)}',
          f'- tests per question: {dict(sorted(Counter(r["n_tests"] for r in out).items()))}',
          f'- all programs tied under metamorphic selection: '
          f'{sum(len(r["metamorphic_selected"]) == len(r["real_correct"]) for r in out)}', '']
for title, subset in (('All questions', out), ('Questions where selection matters', informative),
                      ('Questions with transform tests', with_tests)):
    lines += [f'## {title} (n={len(subset)})', '', '| selector | accuracy |', '|---|---|']
    lines += [f'| {k} | {mean(k, subset):.3f} |' for k in keys] + ['']
lines += ['## Paired differences on all questions (mean, 95% bootstrap CI)', '']
for a, b in (('metamorphic', 'viunit'), ('metamorphic', 'random'), ('metamorphic', 'majority_vote'),
             ('viunit', 'random')):
    m, lo, hi = paired(a, b, out)
    lines.append(f'- {a} - {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]')
tcount = Counter(t for r in with_tests for t in r['transformations'])
lines += ['', '## Transform usage', ''] + [f'- {k}: {v}' for k, v in tcount.most_common()]
(ev / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
