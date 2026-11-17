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


def sample_questions(data, seed, sample_size=200):
    """Allocate proportional detailed-type quotas, then sample without replacement."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError('Sample size must be a positive integer')
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
    total = len(data)
    target = min(sample_size, total)
    # Largest-remainder allocation keeps each quota within one of its ideal
    # share. Integer arithmetic avoids floating-point ties; prefer more common
    # types, then type name, when remainders tie. No minimum quota is imposed.
    quotas = {name: target * count // total for name, count in counts.items()}
    remainder_order = sorted(counts, key=lambda name: (
        -(target * counts[name] % total), -counts[name], name))
    for name in remainder_order[:target - sum(quotas.values())]:
        quotas[name] += 1
    for name, quota in quotas.items():
        if quota == 0:
            print(f'Ignoring type {name}: {counts[name]} question(s); proportional allocation is 0.')
    rng = random.Random(seed)
    rows = []
    for name in sorted(groups):
        for qid in sorted(rng.sample(sorted(groups[name]), quotas[name])):
            rows.append({'id': qid, 'detailed_type': name, 'question': data[qid]['question']})
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'testdev'], default='val')
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-size', type=int, default=200,
                        help='Number of questions to sample proportionally by detailed type (default: 200)')
    args = parser.parse_args()
    try:
        sample = args.experiment_dir / 'questions.jsonl'
        summary = args.experiment_dir / 'sample_summary.json'
        if sample.exists() or summary.exists():
            raise ValueError('Permanent sample already exists; reuse it or choose a new experiment directory')
        dataset = resolve_dataset(args.dataset, args.split)
        payload = dataset.read_bytes()
        rows, counts = sample_questions(json.loads(payload), args.seed, args.sample_size)
        selected_ids = {name: [] for name in counts}
        for row in rows:
            selected_ids[row['detailed_type']].append(row['id'])
        save_jsonl(sample, rows)
        save_json(summary, {'dataset': str(dataset), 'dataset_sha256': hashlib.sha256(payload).hexdigest(),
                            'seed': args.seed, 'type_counts': counts, 'sample_size': len(rows),
                            'requested_sample_size': args.sample_size,
                            'sampling_method': 'proportional_largest_remainder',
                            'sample_type_counts': {name: len(ids) for name, ids in selected_ids.items()},
                            'skipped_types': {name: count for name, count in counts.items() if not selected_ids[name]},
                            'selected_ids': selected_ids,
                            'sample_sha256': hashlib.sha256(sample.read_bytes()).hexdigest()})
        print(f'Saved {len(rows)} questions to {sample}')
        print('\nCreated dataset distribution (question type: count, percentage):')
        for name in sorted(selected_ids, key=lambda name: (-len(selected_ids[name]), name)):
            count = len(selected_ids[name])
            print(f'  {name}: {count} ({100 * count / len(rows):.2f}%)')
        omitted = sum(not ids for ids in selected_ids.values())
        print(f'Question types not represented: {omitted} of {len(counts)}')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
