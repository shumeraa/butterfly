#!/usr/bin/env python3
"""Validate reviewed GQA predictions and generate CSV and Markdown summaries."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

if __package__:
    from .gqa_transform_pilot import FIELDS, TRANSFORMS, write_csv
else:
    from gqa_transform_pilot import FIELDS, TRANSFORMS, write_csv

STATUSES = ('correct', 'incorrect', 'uncertain', 'invalid_source')


def load_review(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError('Review CSV columns must match the exported review.csv')
        rows = list(reader)
    manifest = json.loads(path.with_name('review_manifest.json').read_text(encoding='utf-8'))
    key = lambda r: (r['image_id'], r['question_id'], r['transformation'])
    reference = {key(r): r for r in manifest}
    if not rows or len(rows) != len(reference) or len({key(r) for r in rows}) != len(rows):
        raise ValueError('Review rows are missing or duplicated')
    for row in rows:
        original = reference.get(key(row))
        if original is None or any(row[f] != str(original[f]) for f in FIELDS
                                   if f not in ('status', 'review_note')):
            raise ValueError('Prediction fields changed; edit only status and review_note')
        status = row['status']
        if status not in STATUSES:
            raise ValueError(f'Incomplete or invalid status for {key(row)}: {status!r}')
        if row['relation'] == 'unsafe' and status not in ('uncertain', 'invalid_source'):
            raise ValueError('unsafe is an abstention: use uncertain or invalid_source')
        if row['prediction_error'] and status not in ('incorrect', 'invalid_source'):
            raise ValueError('Malformed/inapplicable predictions must be incorrect or invalid_source')
    questions = {}
    for row in rows:
        questions.setdefault((row['image_id'], row['question_id']), []).append(row)
    for group in questions.values():
        if len(group) != 6 or {r['transformation'] for r in group} != set(TRANSFORMS):
            raise ValueError('Each question must have exactly six transformations')
        invalid = sum(r['status'] == 'invalid_source' for r in group)
        if invalid not in (0, 6):
            raise ValueError('invalid_source must apply to all six rows of a question')
    return rows


def summarize(rows):
    counts = Counter(r['status'] for r in rows)
    # Invalid sources are excluded from rates and accuracy, but retained in counts.
    eligible = [r for r in rows if r['status'] != 'invalid_source']
    abstentions = sum(r['relation'] == 'unsafe' for r in eligible)
    concrete = len(eligible) - abstentions
    judged = counts['correct'] + counts['incorrect']
    rate = lambda n, d: f'{n / d:.6f}' if d else 'unavailable'
    return dict(total=len(rows), **{s: counts[s] for s in STATUSES},
                malformed_predictions=sum(bool(r['prediction_error']) for r in rows),
                abstentions=abstentions, reviewer_uncertain=sum(
                    r['status'] == 'uncertain' and r['relation'] != 'unsafe' for r in rows),
                rate_denominator=len(eligible), concrete_predictions=concrete,
                abstention_rate=rate(abstentions, len(eligible)),
                concrete_coverage=rate(concrete, len(eligible)),
                accuracy_numerator=counts['correct'], accuracy_denominator=judged,
                accuracy=rate(counts['correct'], judged))


def report(args):
    rows = load_review(args.csv)
    args.output.mkdir(parents=True, exist_ok=True)
    markdown = ['# GQA transformation pilot', '',
                'Validity on sampled images; these results do not establish guarantees from question text alone.', '',
                'Rates exclude invalid sources. Concrete predictions include malformed/inapplicable predictions, '
                'which count as incorrect. Accuracy excludes reviewer uncertainty and abstentions. '
                'Malformed counts include structurally invalid responses and inapplicable relations.', '']
    for dimension in ('overall', 'transformation', 'type', 'group'):
        groups = {'all': rows} if dimension == 'overall' else {
            value: [r for r in rows if r[dimension] == value] for value in sorted({r[dimension] for r in rows})}
        summaries = [dict(category=value, **summarize(group)) for value, group in groups.items()]
        write_csv(args.output / f'{dimension}.csv', summaries, list(summaries[0]))
        markdown.extend([f'## {dimension}', '',
                         '| Category | Total | Correct | Incorrect | Uncertain | Invalid source | Malformed | Abstention | Coverage | Reviewer uncertain | Accuracy |',
                         '|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|'])
        for s in summaries:
            category = s['category'].replace('|', '\\|').replace('\n', ' ')
            markdown.append(f'| {category} | {s["total"]} | {s["correct"]} | {s["incorrect"]} | '
                            f'{s["uncertain"]} | {s["invalid_source"]} | {s["malformed_predictions"]} | '
                            f'{s["abstentions"]}/{s["rate_denominator"]} ({s["abstention_rate"]}) | '
                            f'{s["concrete_predictions"]}/{s["rate_denominator"]} ({s["concrete_coverage"]}) | '
                            f'{s["reviewer_uncertain"]} | {s["accuracy_numerator"]}/{s["accuracy_denominator"]} ({s["accuracy"]}) |')
        markdown.append('')
    (args.output / 'report.md').write_text('\n'.join(markdown), encoding='utf-8')
    print(f'Report saved to {args.output / "report.md"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
