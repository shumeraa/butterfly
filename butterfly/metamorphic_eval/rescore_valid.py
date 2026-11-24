#!/usr/bin/env python3
"""Re-score the metamorphic selector after dropping occlusion tests that failed validation.

Usage: python rescore_valid.py <merged_dir> [threshold_key=occluded@0.35]
A test is invalid if the detector still finds the object in the occluded image (count > 0 for the key).
Programs whose baseline run failed keep score 0 on every test, as in the original run.
"""
import glob
import json
import pickle
import random
import statistics as st
import sys
from pathlib import Path

merged = Path(sys.argv[1])
key = sys.argv[2] if len(sys.argv) > 2 else 'occluded@0.35'
val = {(r['questionId'], r['test']): r for r in json.load(open(merged / 'metamorphic_eval/occlusion_validation.json'))}
ds = {}
for l in open(merged / 'dataset.jsonl'):
    r = json.loads(l)
    ds[r['questionId']] = r
recs = {d['questionId']: d for f in sorted(glob.glob(str(merged / 'metamorphic_eval/chunk_*/run/data.p')))
        for d in pickle.load(open(f, 'rb'))}


def pick(s, rc):
    top = max(s)
    i = [k for k, x in enumerate(s) if x == top]
    return sum(rc[k] for k in i) / len(i)


def run(drop_invalid):
    accs, dropped, remaining_occ = [], 0, []
    stats = {'correct': [], 'wrong': []}
    for q, r in ds.items():
        if 'real_correct' not in r:
            continue
        rc = [float(x) for x in r['real_correct']]
        d = recs.get(q)
        if d is None or not d.get('unit_test_results', {}).get(0):
            accs.append(sum(rc) / len(rc))
            continue
        P, meta = d['unit_test_results'][0], d['unit_test_metadata']
        keep = []
        for t, m in enumerate(meta):
            bad = m['transformation'] == 'occlude_bbox' and val[(q, t)][key] > 0
            if bad and drop_invalid:
                dropped += 1
                continue
            keep.append(t)
        if not keep:
            accs.append(sum(rc) / len(rc))
            continue
        scores = [st.mean(p['results'][t]['acc'] for t in keep) for p in P]
        accs.append(pick(scores, rc))
        for t in keep:
            if meta[t]['transformation'] == 'occlude_bbox':
                for i, p in enumerate(P):
                    if not (p['results'] and p['results'][0].get('baseline_error')):
                        stats['correct' if rc[i] else 'wrong'].append(p['results'][t]['acc'])
    return accs, dropped, stats


out = [f'# Occlusion validation (criterion: {key} > 0 means the object is still visible)', '']
v = list(val.values())
out += [f'- occlusion tests: {len(v)}; object still detected after occlusion: {sum(r[key] > 0 for r in v)}',
        f'- object was detected in the original image at this threshold: {sum(r[key.replace("occluded", "original")] > 0 for r in v)}', '']
for drop in (False, True):
    accs, dropped, stats = run(drop)
    c, w = stats['correct'], stats['wrong']
    out.append(f'- drop invalid occlusion tests = {drop}: metamorphic accuracy {st.mean(accs):.3f} (dropped {dropped} tests); '
               f'remaining occlusion pass rate: correct programs {st.mean(c) if c else float("nan"):.2f} (n={len(c)}), '
               f'wrong {st.mean(w) if w else float("nan"):.2f} (n={len(w)})')
    if not drop:
        base = accs
    else:
        diffs = [a - b for a, b in zip(accs, base)]
        rng = random.Random(0)
        boots = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(5000))
        out.append(f'- change vs original: {st.mean(diffs):+.4f} [{boots[124]:+.4f}, {boots[4874]:+.4f}]')
text = '\n'.join(out) + '\n'
(merged / 'metamorphic_eval/occlusion_rescore.md').write_text(text)
print(text)
