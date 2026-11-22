#!/usr/bin/env python3
"""Prepare the metamorphic evaluation inputs from the judge's labels (CPU, login node).

Needs merged/judge_run/labels.json (or LABELS=...). The suite builder raises for a question with
no eligible test or no label, so questions are split here:
  * with >=1 eligible test -> evaluated, in chunks (input.p = merged/real_pass records, config)
  * with none               -> listed in no_test_ids.txt; the export scores them as ties
Writes <OUT>/{eval_ids.txt,no_test_ids.txt,n_tests.json,chunk_<k>/{input.p,ids.txt,config.yaml}}
Usage: python butterfly/metamorphic_eval/prepare_eval.py [n_chunks=8] [tests_per_question=5]
Env:   MERGED (default <ROOT>/merged), LABELS, OUT (default <MERGED>/metamorphic_eval)
"""
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/work/11808/shumera/ls6/projects/butterfly')
from apply_transformations.labels import select_label_tests, tests_from_labels  # noqa: E402

ROOT = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval')
PROJECT = Path('/work/11808/shumera/ls6/projects/butterfly')
MERGED = Path(os.environ.get('MERGED', ROOT / 'merged'))
LABELS = Path(os.environ.get('LABELS', MERGED / 'judge_run' / 'labels.json'))
OUT = Path(os.environ.get('OUT', MERGED / 'metamorphic_eval'))
n_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 8
tests_per_q = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEED = 42

records = pickle.load(open(MERGED / 'real_pass' / 'data.p', 'rb'))
by_q = tests_from_labels(json.load(open(LABELS)))
order = [r['questionId'] for r in records]
if os.environ.get('ONLY_IDS'):                      # optional subset (e.g. the 10-question test)
    keep = set(Path(os.environ['ONLY_IDS']).read_text().split())
    records = [r for r in records if r['questionId'] in keep]

n_tests, with_tests, without = {}, [], []
for r in records:
    q = r['questionId']
    chosen = select_label_tests(by_q.get(q, []), question_id=q, max_tests=tests_per_q, seed=SEED)
    n_tests[q] = len(chosen)
    (with_tests if chosen else without).append(r)

shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
(OUT / 'eval_ids.txt').write_text('\n'.join(r['questionId'] for r in with_tests) + '\n')
(OUT / 'no_test_ids.txt').write_text('\n'.join(r['questionId'] for r in without) + '\n')
json.dump(n_tests, open(OUT / 'n_tests.json', 'w'))

template = (PROJECT / 'viunit_configs/GQA/eval/metamorphic_303.yaml').read_text()
n_chunks = max(1, min(n_chunks, len(with_tests)))
for k in range(n_chunks):
    part = with_tests[k::n_chunks]                  # strided: mixes question types across chunks
    c = OUT / f'chunk_{k}'
    c.mkdir()
    for i, r in enumerate(part):
        r['index'] = i                              # generate.py indexes records by position
    pickle.dump(part, open(c / 'input.p', 'wb'))
    (c / 'ids.txt').write_text('\n'.join(r['questionId'] for r in part) + '\n')
    (c / 'config.yaml').write_text(template
        .replace('__OUTPUT_DIR__', str(c / 'run'))
        .replace('__IDS_FILE__', str(c / 'ids.txt'))
        .replace('__LABELS__', str(LABELS))
        .replace('__TESTS__', str(tests_per_q)))
print(f'{len(with_tests)} questions with tests ({n_chunks} chunks), {len(without)} without; '
      f'tests/question: min {min(n_tests.values())} max {max(n_tests.values())}')
