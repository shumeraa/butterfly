#!/usr/bin/env python3
"""Create one permanent, reproducible question-only GQA sample (standard library only)."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random

from common import save_json, save_jsonl


def resolve_dataset(explicit, split):
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ValueError(f'Dataset does not exist: {path}')
        return path
    image_path = os.environ.get('GQA_IMAGE_PATH')
    if not image_path:
        raise ValueError('Supply --dataset or source export_paths.sh on the execution machine (GQA_IMAGE_PATH is unset)')
    root = Path(image_path).expanduser().parent
    filename = f'{split}_balanced_questions.json'
    candidates = sorted({p.resolve() for p in (root / filename, root / 'questions' / filename,
                        root / 'questions1.2' / filename) if p.is_file()})
    if len(candidates) != 1:
        raise ValueError(f'Expected one {filename} beside GQA_IMAGE_PATH or under questions/ or questions1.2/; found {len(candidates)}. Supply --dataset explicitly.')
    return candidates[0]


def sample_questions(data, seed):
    if not isinstance(data, dict) or not data:
        raise ValueError('Expected a nonempty GQA question-ID mapping')
    groups = defaultdict(list)
    for qid, record in data.items():
        if not isinstance(record, dict) or not isinstance(record.get('types'), dict):
            raise ValueError(f'Question {qid} has no types mapping')
        detailed = record['types'].get('detailed')
        if not isinstance(detailed, str) or not detailed:
            raise ValueError(f'Question {qid} has no detailed type')
        if not isinstance(record.get('question'), str) or not record['question'].strip():
            raise ValueError(f'Question {qid} has no question text')
        groups[detailed].append(qid)
    counts = {name: len(groups[name]) for name in sorted(groups)}
    print(json.dumps(counts, indent=2))
    sparse = [name for name, count in counts.items() if count < 2]
    if sparse:
        raise ValueError('Types with fewer than two questions: ' + ', '.join(sparse))
    rng = random.Random(seed)
    rows = []
    for name in sorted(groups):
        for qid in sorted(rng.sample(sorted(groups[name]), 2)):
            rows.append({'id': qid, 'detailed_type': name, 'question': data[qid]['question']})
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'testdev'], default='val')
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        sample = args.experiment_dir / 'questions.jsonl'
        summary = args.experiment_dir / 'sample_summary.json'
        if sample.exists() or summary.exists():
            raise ValueError('Permanent sample already exists; reuse it or choose a new experiment directory')
        dataset = resolve_dataset(args.dataset, args.split)
        payload = dataset.read_bytes()
        rows, counts = sample_questions(json.loads(payload), args.seed)
        save_jsonl(sample, rows)
        save_json(summary, {'dataset': str(dataset), 'dataset_sha256': hashlib.sha256(payload).hexdigest(),
                            'seed': args.seed, 'type_counts': counts, 'sample_size': len(rows),
                            'selected_ids': {name: [r['id'] for r in rows if r['detailed_type'] == name] for name in counts},
                            'sample_sha256': hashlib.sha256(sample.read_bytes()).hexdigest()})
        print(f'Saved {len(rows)} questions to {sample}')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
