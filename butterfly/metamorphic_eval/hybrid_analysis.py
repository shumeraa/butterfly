#!/usr/bin/env python3
"""Offline checks: (2) ViUniT+metamorphic hybrid selectors, (3) metamorphic score as a reliability signal.

Usage: python hybrid_analysis.py <merged_dir>      (needs dataset.jsonl and metamorphic_eval/comparison.jsonl)
Everything is computed from saved scores; no model runs.
"""
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

merged = Path(sys.argv[1])
ds = {}
for l in open(merged / 'dataset.jsonl'):
    r = json.loads(l)
    ds[r['questionId']] = r
cmp_ = {}
for l in open(merged / 'metamorphic_eval' / 'comparison.jsonl'):
    r = json.loads(l)
    cmp_[r['questionId']] = r

Q = []
for q, r in ds.items():
    if 'real_correct' not in r:
        continue
    n = len(r['real_correct'])
    c = cmp_[q]
    meta = c.get('metamorphic_scores') or [0.0] * n
    # a crashed program has no answer: it must not "agree" with other crashes, so each gets a unique key
    outs = [o if e is None and o is not None else f'__err{i}' for i, (o, e) in enumerate(zip(r['real_outputs'], r['real_errors']))]
    cnt = Counter(outs)
    Q.append(dict(q=q, rc=[float(x) for x in r['real_correct']], vu=list(r['unit_test_scores']),
                  meta=list(meta), has=c['has_tests'], nmeta=c.get('n_tests', 0),
                  vote=[cnt[o] for o in outs], fixed_n_vu=5))


def pick(scores, rc, *tiebreaks):
    """Expected accuracy: highest score, ties broken by successive keys, then uniform."""
    idx = list(range(len(rc)))
    for key in (scores,) + tiebreaks:
        top = max(key[i] for i in idx)
        idx = [i for i in idx if key[i] == top]
    return sum(rc[i] for i in idx) / len(idx)


def z(x):  # per-question standardisation so the two score scales are comparable
    m, s = st.mean(x), st.pstdev(x)
    return [(v - m) / s if s > 0 else 0.0 for v in x]


selectors = {
    'random': lambda d: sum(d['rc']) / len(d['rc']),
    'majority_vote': lambda d: pick(d['vote'], d['rc']),
    'viunit': lambda d: pick(d['vu'], d['rc']),
    'metamorphic': lambda d: pick(d['meta'], d['rc']),
    'viunit > metamorphic (tie-break)': lambda d: pick(d['vu'], d['rc'], d['meta']),
    'viunit > majority (tie-break)': lambda d: pick(d['vu'], d['rc'], d['vote']),
    'metamorphic > viunit (tie-break)': lambda d: pick(d['meta'], d['rc'], d['vu']),
    'pooled tests (vu 5 + meta n)': lambda d: pick(
        [(5 * a + d['nmeta'] * b) / (5 + d['nmeta']) for a, b in zip(d['vu'], d['meta'])], d['rc']),
    'mean z(vu) + z(meta)': lambda d: pick([a + b for a, b in zip(z(d['vu']), z(d['meta']))], d['rc']),
    'viunit + majority + metamorphic (z)': lambda d: pick(
        [a + b + c for a, b, c in zip(z(d['vu']), z(d['meta']), z(d['vote']))], d['rc']),
    'oracle': lambda d: max(d['rc']),
}
vals = {k: [f(d) for d in Q] for k, f in selectors.items()}


def paired(a, b, n=5000, seed=0):
    diffs = [x - y for x, y in zip(vals[a], vals[b])]
    rng = random.Random(seed)
    boots = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return st.mean(diffs), boots[int(.025 * n)], boots[int(.975 * n) - 1]


out = [f'# Hybrid selectors and reliability signal (n={len(Q)} questions)', '',
       '## 2. Selector accuracy (expected over ties)', '', '| selector | accuracy | vs viunit (95% CI) | vs majority (95% CI) |',
       '|---|---|---|---|']
for k in selectors:
    a = paired(k, 'viunit') if k != 'viunit' else (0, 0, 0)
    b = paired(k, 'majority_vote') if k != 'majority_vote' else (0, 0, 0)
    out.append(f'| {k} | {st.mean(vals[k]):.3f} | {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}] | '
               f'{b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}] |')

tied = [d for d in Q if len(set(d['vu'])) == 1]
out += ['', f'ViUniT all-tied questions: {len(tied)}. Accuracy on them: '
        f'viunit/random {st.mean(sum(d["rc"]) / 5 for d in tied):.3f}, '
        f'+metamorphic tie-break {st.mean(pick(d["vu"], d["rc"], d["meta"]) for d in tied):.3f}, '
        f'+majority tie-break {st.mean(pick(d["vu"], d["rc"], d["vote"]) for d in tied):.3f}']


# 3a. Within-question ranking quality: P(correct program scores above wrong program), ties = 0.5
def within_auc(key):
    num = den = 0.0
    for d in Q:
        s = d[key]
        for i, ci in enumerate(d['rc']):
            for j, cj in enumerate(d['rc']):
                if ci == 1 and cj == 0:
                    den += 1
                    num += 1 if s[i] > s[j] else .5 if s[i] == s[j] else 0
    return num / den, int(den)


out += ['', '## 3a. Within-question pairwise AUC (correct vs wrong program; 0.5 = chance)', '',
        '| score | AUC | pairs |', '|---|---|---|']
for key, name in (('vu', 'viunit'), ('meta', 'metamorphic'), ('vote', 'majority count')):
    a, n = within_auc(key)
    out.append(f'| {name} | {a:.3f} | {n} |')

# 3b. Question-level refusal: rank questions by confidence = top score of the selector, answer the top-k%
rng = random.Random(0)


def selective(conf_fn, acc_fn, reps=200):
    rows = []
    for frac in (1.0, .75, .5, .25):
        accs = []
        for _ in range(reps):
            order = sorted(Q, key=lambda d: (-conf_fn(d), rng.random()))
            keep = order[:max(1, int(frac * len(order)))]
            accs.append(st.mean(acc_fn(d) for d in keep))
        rows.append((frac, st.mean(accs)))
    return rows


conf = {
    'viunit top score': lambda d: max(d['vu']),
    'metamorphic top score': lambda d: max(d['meta']) if d['has'] else 0,
    'vote agreement (max count)': lambda d: max(d['vote']),
    'viunit + metamorphic top score': lambda d: max(d['vu']) + (max(d['meta']) if d['has'] else 0),
}
out += ['', '## 3b. Answer refusal: accuracy of the viunit-selected answer when answering only the most confident k% of questions', '',
        '| confidence | 100% | 75% | 50% | 25% |', '|---|---|---|---|---|']
for name, fn in conf.items():
    r = selective(fn, selectors['viunit'].__call__ if False else (lambda d: pick(d['vu'], d['rc'])))
    out.append(f'| {name} | ' + ' | '.join(f'{a:.3f}' for _, a in r) + ' |')

# 3c. does low metamorphic consistency flag wrong answers? program-level correct rate by meta score bucket
b = {'meta=1.0': [], '0.5<=meta<1': [], 'meta<0.5': []}
for d in Q:
    if not d['has']:
        continue
    for s, c in zip(d['meta'], d['rc']):
        b['meta=1.0' if s == 1 else '0.5<=meta<1' if s >= .5 else 'meta<0.5'].append(c)
out += ['', '## 3c. Program-level correct rate by metamorphic score', '', '| bucket | programs | correct rate |', '|---|---|---|']
out += [f'| {k} | {len(v)} | {st.mean(v):.3f} |' for k, v in b.items() if v]
text = '\n'.join(out) + '\n'
(merged / 'metamorphic_eval' / 'hybrid_analysis.md').write_text(text)
print(text)
