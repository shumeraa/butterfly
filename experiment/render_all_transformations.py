"""Render EVERY non-unsafe transformation for every LLM output (labels.json).

Unlike run_transform_selection.py (which samples <=3), this renders all invariant
and equivariant candidates. Output layout under --out-dir:
  images/<question_id>/<NN>/000.png   transformed image for candidate NN
  manifest.jsonl                       one row per question, all candidates + status
Resumable: candidates whose PNG already exists are skipped.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apply_transformations.labels import tests_from_labels  # noqa: E402
from apply_transformations.materialize import materialize_selected_tests  # noqa: E402

GQA = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/gqa')
SAM_OPS = {'fill_color', 'occlude_bbox'}


def _compat_threshold(segmenter):
    """transformers<4.51 names Grounding DINO's `threshold` arg `box_threshold`."""
    import inspect
    proc = segmenter.detector_processor
    if 'threshold' in inspect.signature(proc.post_process_grounded_object_detection).parameters:
        return
    original = proc.post_process_grounded_object_detection

    def patched(*a, threshold=None, **kw):
        if threshold is not None:
            kw['box_threshold'] = threshold
        return original(*a, **kw)
    proc.post_process_grounded_object_detection = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', type=Path, default=ROOT / 'experiment/questions.jsonl')
    ap.add_argument('--labels', type=Path, default=ROOT / 'experiment/runs/run_004/labels.json')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'experiment/all_transformations')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fill-strength', type=float, default=0.75,
                    help='fill_color tint strength in (0,1]; 1 = solid fill of every masked pixel')
    ap.add_argument('--examples-per-transformation', type=int, default=0, metavar='N',
                    help='example mode: visit questions in seeded random order and render all '
                         'candidates of a question only while some transformation type has fewer '
                         'than N successfully rendered images; stop once every type has N')
    ap.add_argument('--skip-sam', action='store_true', help='skip fill_color/occlude_bbox (no GPU models)')
    args = ap.parse_args()

    questions = [json.loads(l) for l in args.questions.read_text().splitlines() if l.strip()]
    candidates = tests_from_labels(json.loads(args.labels.read_text()))
    gqa = json.loads((GQA / 'questions/val_balanced_questions.json').read_text())
    missing = [q['id'] for q in questions if q['id'] not in candidates]
    if missing:
        sys.exit(f'No labels for {len(missing)} questions, e.g. {missing[:3]}')

    examples = args.examples_per_transformation
    if examples:
        random.Random(args.seed).shuffle(questions)
    wanted = {t['transformation'] for q in questions for t in candidates[q['id']]}
    if args.skip_sam:
        wanted -= SAM_OPS
    got = {op: [] for op in sorted(wanted)}  # op -> question ids with a rendered image

    segmenter = None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    totals = dict(ok=0, cached=0, failed=0, skipped=0)
    with (args.out_dir / 'manifest.jsonl').open('w') as manifest:
        for n, q in enumerate(questions, 1):
            qid = q['id']
            if examples:
                if all(len(ids) >= examples for ids in got.values()):
                    break
                ops = {t['transformation'] for t in candidates[qid]}
                if not any(len(got[op]) < examples for op in ops if op in got):
                    continue
            image = GQA / 'images' / f"{gqa[qid]['imageId']}.jpg"
            results = []
            if examples:
                (args.out_dir / 'images' / qid).mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, args.out_dir / 'images' / qid / f'original{image.suffix}')
            for i, test in enumerate(candidates[qid]):
                if test['transformation'] == 'fill_color':
                    test = {**test, 'parameters': {**test['parameters'], 'strength': args.fill_strength}}
                dest = args.out_dir / 'images' / qid / f'{i:02d}'
                entry = {**test, 'index': i, 'image_paths': [], 'status': None, 'error': None}
                if test['transformation'] in SAM_OPS and args.skip_sam:
                    entry['status'] = 'skipped'
                elif (dest / '000.png').exists():
                    entry.update(status='cached', image_paths=[str(dest / '000.png')])
                else:
                    try:
                        if test['transformation'] in SAM_OPS and segmenter is None:
                            from image_transforms import GroundedSAM
                            segmenter = GroundedSAM()
                            _compat_threshold(segmenter)
                        out = materialize_selected_tests(image, [test], dest, seed=args.seed,
                                                         segmenter=segmenter)
                        entry.update(status='ok', image_paths=out[0]['image_paths'])
                    except Exception as e:  # keep going; record the failure
                        entry.update(status='failed', error=f'{type(e).__name__}: {e}')
                totals[entry['status']] += 1
                results.append(entry)
                if entry['status'] in ('ok', 'cached') and test['transformation'] in got:
                    got[test['transformation']].append(qid)
            manifest.write(json.dumps(dict(id=qid, detailed_type=q['detailed_type'],
                question=q['question'], image_id=gqa[qid]['imageId'], original_image=str(image),
                gqa_answer=gqa[qid]['answer'], transformations=results)) + '\n')
            manifest.flush()
            print(f'[{n}/{len(questions)}] {qid}: {len(results)} candidates', flush=True)
    print('TOTALS', totals)
    if examples:
        lines = [f'# Example transformations (>= {examples} images per type)', '']
        for op, ids in got.items():
            lines.append(f'- {op}: {len(ids)} images, questions {", ".join(ids)}')
        (args.out_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines))
        short = [op for op, ids in got.items() if len(ids) < examples]
        if short:
            print(f'WARNING: fewer than {examples} examples for {short}', file=sys.stderr)
        sys.exit(1 if short else 0)
    sys.exit(1 if totals['failed'] else 0)


if __name__ == '__main__':
    main()
