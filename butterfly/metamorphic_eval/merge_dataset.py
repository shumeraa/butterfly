#!/usr/bin/env python3
"""Merge the viz_program_eval chunks of one or more data roots into one directory (CPU, login node is fine).

Writes <ROOT>/merged/{dataset.jsonl, data.p, real_pass/data.p, questions.jsonl, MANIFEST.json}
in the frozen sample_ids.txt order. questions.jsonl is the LLM-judge input
({id, detailed_type, question}; the judge rejects extra keys).
"""
import json
import os
import pickle
from pathlib import Path

ROOT = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval')
ROOT_V2 = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval_v2')
# Env: SOURCES  colon-separated data roots, merged in order (default: the original 303 only)
#      OUT      output dir (default <first source>/merged)
#      DEDUPE   pair (default) | id | none. Later duplicates are dropped, so earlier sources win.
#               id   = same questionId; pair = same questionId OR same (question text, answer)
#      ALLOW_PARTIAL=1  skip unfinished chunks (testing only)
SOURCES = [Path(p) for p in os.environ.get('SOURCES', str(ROOT)).split(':')]
OUT = Path(os.environ.get('OUT', SOURCES[0] / 'merged'))
DEDUPE = os.environ.get('DEDUPE', 'pair')
(OUT / 'real_pass').mkdir(parents=True, exist_ok=True)

order, rows, rec, real, dropped = [], {}, {}, {}, []
seen_pairs, n_src = {}, {}
for src in SOURCES:
    src_order = src.joinpath('sample_ids.txt').read_text().split()
    chunk_dirs = sorted((src / 'full').glob('chunk_*'), key=lambda c: int(c.name.split('_')[1]))
    s_rows, s_rec, s_real = {}, {}, {}
    for c in chunk_dirs:
        if not (c / 'DONE').exists():
            assert os.environ.get('ALLOW_PARTIAL'), f'{c} not finished'
            continue
        for line in open(c / 'dataset.jsonl'):
            r = json.loads(line)
            s_rows[r['questionId']] = r
        for d in pickle.load(open(c / 'data.p', 'rb')):
            s_rec[d['questionId']] = d
        for d in pickle.load(open(c / 'real_pass' / 'data.p', 'rb')):
            s_real[d['questionId']] = d
    if not os.environ.get('ALLOW_PARTIAL'):
        assert set(src_order) == set(s_rows) == set(s_rec) == set(s_real), f'{src}: chunk ids do not match sample_ids.txt'
    n_src[str(src)] = 0
    for q in src_order:
        if q not in s_rows:
            continue
        r = s_rows[q]
        pair = (r['question'].strip().lower(), str(r['answer']).strip().lower())
        why = None
        if DEDUPE != 'none':
            if q in rows:
                why = 'questionId'
            elif DEDUPE == 'pair' and pair in seen_pairs:
                why = f'same question+answer as {seen_pairs[pair]}'
        if why:
            dropped.append({'questionId': q, 'source': str(src), 'reason': why})
            continue
        seen_pairs.setdefault(pair, q)
        order.append(q)
        rows[q], rec[q], real[q] = r, s_rec[q], s_real[q]
        n_src[str(src)] += 1
assert len(order) == len(set(order))

data = []
for i, q in enumerate(order):
    d = rec[q]
    d['index'] = i          # generate.py indexes by position; keep it contiguous in merged order
    data.append(d)
real_data = []
for i, q in enumerate(order):
    d = real[q]
    d['index'] = i
    real_data.append(d)

with open(OUT / 'dataset.jsonl', 'w') as f:
    for i, q in enumerate(order):
        rows[q]['index'] = i
        f.write(json.dumps(rows[q]) + '\n')
pickle.dump(data, open(OUT / 'data.p', 'wb'))
pickle.dump(real_data, open(OUT / 'real_pass' / 'data.p', 'wb'))

with open(OUT / 'questions.jsonl', 'w') as f:
    for q in order:
        r = rows[q]
        for key in ('question', 'group'):
            assert isinstance(r[key], str) and r[key].strip(), (q, key)
        f.write(json.dumps({'id': q, 'detailed_type': r['group'], 'question': r['question']}) + '\n')

json.dump({'n_questions': len(order), 'question_ids': order, 'sources': n_src, 'dedupe': DEDUPE,
           'dropped_duplicates': dropped,
           'files': ['dataset.jsonl', 'data.p', 'real_pass/data.p', 'questions.jsonl']},
          open(OUT / 'MANIFEST.json', 'w'), indent=2)
print(f'merged {len(order)} questions into {OUT}; per source {list(n_src.values())}; '
      f'dropped {len(dropped)} duplicates (see MANIFEST.json)')
