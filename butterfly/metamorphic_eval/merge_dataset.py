#!/usr/bin/env python3
"""Merge the 13 viz_program_eval chunks into one directory (CPU, login node is fine).

Writes <ROOT>/merged/{dataset.jsonl, data.p, real_pass/data.p, questions.jsonl, MANIFEST.json}
in the frozen sample_ids.txt order. questions.jsonl is the LLM-judge input
({id, detailed_type, question}; the judge rejects extra keys).
"""
import json
import pickle
from pathlib import Path

ROOT = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval')
N_CHUNKS = 13
OUT = ROOT / 'merged'
(OUT / 'real_pass').mkdir(parents=True, exist_ok=True)

order = ROOT.joinpath('sample_ids.txt').read_text().split()
rows, rec, real = {}, {}, {}
for k in range(N_CHUNKS):
    c = ROOT / 'full' / f'chunk_{k}'
    assert (c / 'DONE').exists(), f'chunk {k} not finished'
    for line in open(c / 'dataset.jsonl'):
        r = json.loads(line)
        rows[r['questionId']] = r
    for d in pickle.load(open(c / 'data.p', 'rb')):
        rec[d['questionId']] = d
    for d in pickle.load(open(c / 'real_pass' / 'data.p', 'rb')):
        real[d['questionId']] = d

assert set(order) == set(rows) == set(rec) == set(real), 'chunk ids do not match sample_ids.txt'
assert len(order) == len(set(order)) == 303

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

json.dump({'n_questions': len(order), 'question_ids': order, 'source': 'full/chunk_0..12',
           'files': ['dataset.jsonl', 'data.p', 'real_pass/data.p', 'questions.jsonl']},
          open(OUT / 'MANIFEST.json', 'w'), indent=2)
print(f'merged {len(order)} questions into {OUT}')
