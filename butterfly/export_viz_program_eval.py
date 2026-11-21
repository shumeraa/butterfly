#!/usr/bin/env python3
"""Export a run directory (ViUniT data.p + real_pass/data.p) to a readable dataset.jsonl.

Usage (from the repo root): python butterfly/export_viz_program_eval.py <run_dir>
Writes <run_dir>/dataset.jsonl and <run_dir>/MANIFEST.json. The data.p files stay untouched.

Per question the row holds the 5 candidate programs, ViUniT's unit tests and scores,
every candidate's real-image output/correctness, which program(s) ViUniT selected
(all programs tied at the top unit-test score), and selector baselines.
"""
import json
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/work/11808/shumera/ls6/projects/butterfly')
from accuracy_fns import GQA_accuracy  # noqa: E402


def plain(x):
    """JSON-safe copy of a pickled value (PIL images become a marker string)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, 'size') and hasattr(x, 'mode'):
        return f'<image {x.size[0]}x{x.size[1]}>'
    return str(x)


def real_pass_rows(run, n):
    """Per-question lists of (output, error, correct) for each candidate, or None."""
    f = run / 'real_pass' / 'data.p'
    if not f.exists():
        return [None] * n
    rows = []
    for d in pickle.load(open(f, 'rb')):
        outs = d.get('code_outputs', {}).get(0, {})
        rows.append([outs.get(i) for i in range(len(d['generated_code'][0]))])
    return rows


def main(run_dir):
    run = Path(run_dir)
    data = pickle.load(open(run / 'data.p', 'rb'))
    real = real_pass_rows(run, len(data))
    keep = ['index', 'questionId', 'imageId', 'question', 'answer', 'group',
            'generated_code', 'unit_tests', 'selected_unit_tests', 'unit_test_results']
    n_real = 0
    with open(run / 'dataset.jsonl', 'w') as f:
        for d, outs in zip(data, real):
            row = {k: plain(d[k]) for k in keep if k in d}
            row['image_path'] = f"{d.get('imageId')}.jpg"  # under $GQA_IMAGE_PATH
            ut = d['unit_test_results'][0]
            scores = [ut[i]['acc'] for i in range(len(ut))]
            top = max(scores)
            row['unit_test_scores'] = scores
            row['viunit_selected'] = [i for i, s in enumerate(scores) if s == top]
            if outs is not None and all(o is not None for o in outs):
                n_real += 1
                correct = [float(o['error'] is None and GQA_accuracy([o['output']], [d['answer']]))
                           for o in outs]
                good = [o['output'] for o in outs if o['error'] is None]
                vote = Counter(good).most_common(1)[0][0] if good else None
                row['real_outputs'] = [plain(o['output']) for o in outs]
                row['real_errors'] = [o['error'] for o in outs]
                row['real_correct'] = correct
                row['baselines'] = {
                    'random': sum(correct) / len(correct),
                    'oracle': max(correct),
                    'majority_vote': float(vote is not None and GQA_accuracy([vote], [d['answer']])),
                    # ties at the top unit-test score broken uniformly at random
                    'viunit': sum(correct[i] for i in row['viunit_selected']) / len(row['viunit_selected']),
                }
            f.write(json.dumps(row) + '\n')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            cwd='/work/11808/shumera/ls6/projects/butterfly').stdout.strip()
    json.dump({
        'n_questions': len(data),
        'n_with_real_pass': n_real,
        'question_ids': [d.get('questionId') for d in data],
        'git_commit': commit,
        'config': 'config_used.yaml',
        'raw_artifacts': ['data.p', 'real_pass/data.p'],
    }, open(run / 'MANIFEST.json', 'w'), indent=2)
    print(f'wrote {len(data)} rows ({n_real} with real-image results) to {run / "dataset.jsonl"}')


if __name__ == '__main__':
    main(sys.argv[1])
