"""Run experiment/questions.jsonl through the merged apply_transformations pipeline.

Selection (equivariant first, fill with seeded invariant, max N) uses
apply_transformations.labels; image edits use apply_transformations.materialize.
fill_color/occlude_bbox need GroundedSAM (GPU/transformers); with --skip-sam they are
recorded in the manifest but not rendered.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apply_transformations.labels import select_label_tests, tests_from_labels  # noqa: E402
from apply_transformations.materialize import materialize_selected_tests  # noqa: E402

GQA = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/gqa')
SAM_OPS = {'fill_color', 'occlude_bbox'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', type=Path, default=ROOT / 'experiment/questions.jsonl')
    ap.add_argument('--labels', type=Path, default=ROOT / 'experiment/runs/run_004/labels.json')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'experiment/transform_selection')
    ap.add_argument('--max-per-question', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-sam', action='store_true', default=True)
    args = ap.parse_args()

    questions = [json.loads(l) for l in args.questions.read_text().splitlines() if l.strip()]
    labels = json.loads(args.labels.read_text())
    candidates = tests_from_labels(labels)
    gqa = json.loads((GQA / 'questions/val_balanced_questions.json').read_text())
    ids = [q['id'] for q in questions]
    assert set(ids) == set(candidates), 'question ids and label ids differ'

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in questions:
        qid = q['id']
        g = gqa[qid]
        image = GQA / 'images' / f"{g['imageId']}.jpg"
        valid = candidates[qid]
        chosen = select_label_tests(valid, question_id=qid, max_tests=args.max_per_question, seed=args.seed)
        renderable = [t for t in chosen if t['transformation'] not in SAM_OPS]
        rendered = materialize_selected_tests(image, renderable, args.out_dir / 'images' / qid,
                                              seed=args.seed) if renderable else []
        by_desc = {r['description']: r for r in rendered}
        selected = []
        for t in chosen:
            r = by_desc.get(t['description'])
            selected.append({**t, 'rendered': r is not None,
                             'image_paths': r['image_paths'] if r else []})
        rows.append(dict(id=qid, detailed_type=q['detailed_type'], question=q['question'],
                         image_id=g['imageId'], original_image=str(image),
                         gqa_answer=g['answer'], n_valid=len(valid),
                         n_valid_equivariant=sum(t['relation'] == 'equivariant' for t in valid),
                         n_valid_invariant=sum(t['relation'] == 'invariant' for t in valid),
                         selected=selected))

    with (args.out_dir / 'selected_transformations.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    from collections import Counter
    sizes = Counter(len(r['selected']) for r in rows)
    ops = Counter((t['transformation'], t['relation']) for r in rows for t in r['selected'])
    rend = Counter(t['rendered'] for r in rows for t in r['selected'])
    lines = ['# Transformation selection summary', '',
             f'questions: {len(rows)}  seed: {args.seed}  max per question: {args.max_per_question}',
             f'labels: {args.labels}', '', '## Selected count per question']
    lines += [f'- {n} selected: {sizes.get(n, 0)} questions' for n in range(args.max_per_question, -1, -1)]
    lines += ['', '## Transformation x relation (selected)']
    lines += [f'- {op} / {rel}: {c}' for (op, rel), c in sorted(ops.items())]
    lines += ['', f'rendered images: {rend.get(True, 0)}; selected but not rendered (need GroundedSAM): {rend.get(False, 0)}']
    (args.out_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
