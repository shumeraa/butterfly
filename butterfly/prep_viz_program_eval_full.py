#!/usr/bin/env python3
"""Freeze the full ViUniT GQA eval sample (3 questions per group, seed 42) and split it
into chunks that each fit in a 2h gpu-a100-dev job.

Writes <DATA>/sample_ids.txt, <DATA>/chunks/chunk_<k>.txt and one config per chunk
under viunit_configs/GQA/eval/viz_program_eval_full/.
Usage: python butterfly/prep_viz_program_eval_full.py [n_chunks=8]
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path('/work/11808/shumera/ls6/projects/butterfly')
DATA = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval')
ANN = PROJECT / 'my_datasets/annotation_files/GQA/balanced_val_annotation.json'
n_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 13  # ~23 questions each

ann = json.load(open(ANN))
by_group = defaultdict(list)
for qs in ann.values():
    for q in qs:
        by_group[q['group']].append(q['questionId'])

rng = random.Random(42)
ids = []
for g in sorted(by_group):
    ids += rng.sample(sorted(by_group[g]), min(3, len(by_group[g])))
rng.shuffle(ids)  # mix question types across chunks

import shutil
shutil.rmtree(DATA / 'chunks', ignore_errors=True)
shutil.rmtree(PROJECT / 'viunit_configs/GQA/eval/viz_program_eval_full', ignore_errors=True)
DATA.mkdir(parents=True, exist_ok=True)
(DATA / 'chunks').mkdir(exist_ok=True)
(DATA / 'sample_ids.txt').write_text('\n'.join(ids) + '\n')

cfg_dir = PROJECT / 'viunit_configs/GQA/eval/viz_program_eval_full'
cfg_dir.mkdir(parents=True, exist_ok=True)
template = (PROJECT / 'viunit_configs/GQA/eval/viz_program_eval_10.yaml').read_text()
for k in range(n_chunks):
    chunk = ids[k::n_chunks]
    cf = DATA / 'chunks' / f'chunk_{k}.txt'
    cf.write_text('\n'.join(chunk) + '\n')
    cfg = (template
           .replace('viz_program_eval/run_10', f'viz_program_eval/full/chunk_{k}')
           .replace('sample_ids_file: null', f'sample_ids_file: {cf}')
           .replace('num_samples: 10', 'num_samples: -1'))
    (cfg_dir / f'chunk_{k}.yaml').write_text(cfg)
print(f'{len(ids)} questions, {n_chunks} chunks of ~{len(ids)//n_chunks}')
