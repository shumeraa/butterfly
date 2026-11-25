#!/usr/bin/env python3
"""Prepare a real-image pass over every candidate program WITHOUT the simple_query fallback.

Why: the original real pass retried a crashed program with a fallback and recorded the fallback's
answer with error=None, so `real_correct` credited answers the generated program never produced. This
pass sets execution.real_pass_fixed_code: false, so a crash/timeout keeps its error (correct = wrong).

Writes <OUT>/{chunk_<k>/{input.p,ids.txt,config.yaml},worker.sh}; run with run_real_pass.sh.
Usage: python butterfly/metamorphic_eval/prepare_real_pass.py [n_chunks=3]
Env:   MERGED (default <ROOT>/merged), OUT (default <ROOT>/gqa-simple-v5/real_pass_nofallback)
"""
import os
import pickle
import shutil
import sys
from pathlib import Path

PROJECT = Path('/work/11808/shumera/ls6/projects/butterfly')
ROOT = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval_600')
MERGED = Path(os.environ.get('MERGED', ROOT / 'merged'))
OUT = Path(os.environ.get('OUT', ROOT / 'gqa-simple-v5' / 'real_pass_nofallback'))
n_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 3

records = pickle.load(open(MERGED / 'data.p', 'rb'))
for r in records:   # cached outputs would make generate.py skip execution entirely
    for key in ('code_outputs', 'max_accuracy_output', 'final_accuracy'):
        r.pop(key, None)

template = (PROJECT / 'viunit_configs/GQA/eval/viz_program_eval_full_v2/chunk_0.yaml').read_text()
lines = template.splitlines()
out_lines, in_exec = [], False
for line in lines:
    if line.startswith('do_unit_test:'):
        line = 'do_unit_test: False'
    elif line.startswith('output_dir:'):
        line = 'output_dir: "__OUTPUT_DIR__"'
    elif line.strip().startswith('sample_ids_file:'):
        line = '    sample_ids_file: __IDS_FILE__'
    out_lines.append(line)
    if line.startswith('execution:'):
        out_lines.append('  real_pass_fixed_code: false   # keep crashes as errors; no simple_query fallback')
template = '\n'.join(out_lines) + '\n'
assert '__OUTPUT_DIR__' in template and '__IDS_FILE__' in template and 'real_pass_fixed_code' in template

shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
for k in range(n_chunks):
    part = records[k::n_chunks]
    c = OUT / f'chunk_{k}'
    c.mkdir()
    for i, r in enumerate(part):
        r['index'] = i          # generate.py indexes records by position
    pickle.dump(part, open(c / 'input.p', 'wb'))
    (c / 'ids.txt').write_text('\n'.join(r['questionId'] for r in part) + '\n')
    (c / 'config.yaml').write_text(template.replace('__OUTPUT_DIR__', str(c / 'run'))
                                   .replace('__IDS_FILE__', str(c / 'ids.txt')))
print(f'{len(records)} questions, {sum(len(r["generated_code"][0]) for r in records)} programs, '
      f'{n_chunks} chunks -> {OUT}')
