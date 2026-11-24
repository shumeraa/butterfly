#!/usr/bin/env python3
"""Freeze the full ViUniT GQA eval sample (3 questions per group, seed 42) and split it
into chunks that each fit in a 2h gpu-a100-dev job.

Writes <DATA>/sample_ids.txt, <DATA>/chunks/chunk_<k>.txt and one config per chunk
under viunit_configs/GQA/eval/viz_program_eval_full/.
Usage: python butterfly/prep_viz_program_eval_full.py [n_chunks=13]
Env overrides (to add a disjoint batch without touching the original one):
  DATA_DIR   output dir (default: .../viz_program_eval)
  SEED       sampling seed (default 42)
  EXCLUDE    file of question ids to leave out (e.g. the original sample_ids.txt)
  CFG_NAME   config dir name under viunit_configs/GQA/eval/ (default viz_program_eval_full)
Refuses to run if the target already has sample_ids.txt, chunks, or configs; nothing is deleted.
"""
import json, os, random, sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path('/work/11808/shumera/ls6/projects/butterfly')
DATA = Path(os.environ.get('DATA_DIR', '/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval'))
SEED = int(os.environ.get('SEED', 42))
CFG_NAME = os.environ.get('CFG_NAME', 'viz_program_eval_full')
EXCLUDE = os.environ.get('EXCLUDE')
ANN = PROJECT / 'my_datasets/annotation_files/GQA/balanced_val_annotation.json'
n_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 13  # ~23 questions each

ann = json.load(open(ANN))
by_group = defaultdict(list)
for qs in ann.values():
    for q in qs:
        by_group[q['group']].append(q['questionId'])

excluded = set(Path(EXCLUDE).read_text().split()) if EXCLUDE else set()
cfg_dir = PROJECT / 'viunit_configs/GQA/eval' / CFG_NAME
for existing in (DATA / 'sample_ids.txt', DATA / 'chunks', cfg_dir):
    if existing.exists():
        sys.exit(f'refusing to overwrite {existing}; choose a new DATA_DIR / CFG_NAME')

rng = random.Random(SEED)
ids = []
for g in sorted(by_group):
    pool = [q for q in sorted(by_group[g]) if q not in excluded]
    ids += rng.sample(pool, min(3, len(pool)))
rng.shuffle(ids)  # mix question types across chunks

DATA.mkdir(parents=True, exist_ok=True)
(DATA / 'chunks').mkdir()
(DATA / 'sample_ids.txt').write_text('\n'.join(ids) + '\n')

cfg_dir.mkdir(parents=True)
template = (PROJECT / 'viunit_configs/GQA/eval/viz_program_eval_10.yaml').read_text()
for k in range(n_chunks):
    chunk = ids[k::n_chunks]
    cf = DATA / 'chunks' / f'chunk_{k}.txt'
    cf.write_text('\n'.join(chunk) + '\n')
    cfg = (template
           .replace('viz_program_eval/run_10', f'{DATA.name}/full/chunk_{k}')
           .replace('sample_ids_file: null', f'sample_ids_file: {cf}')
           .replace('num_samples: 10', 'num_samples: -1'))
    (cfg_dir / f'chunk_{k}.yaml').write_text(cfg)
print(f'{len(ids)} questions, {n_chunks} chunks of ~{len(ids)//n_chunks}')
