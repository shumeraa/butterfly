#!/usr/bin/env python3
"""Benchmark occlusion variants on every occlusion test the pipeline attempted.

Usage: python benchmark_occlusion.py <pipeline_dir> [device=cuda] [out_dir]
Cases = occlude_bbox tests kept in <pipeline_dir>/metamorphic_eval/chunk_*/run/data.p plus the ones
rejected in chunk_*/worker.log ("skipping test '<item>, occluded bounding box'"), minus "no object
matched". Each variant re-renders the occlusion from the original image; an accepted result is then
checked with an independent detector (grounding-dino-tiny) that ignores detections sitting on the fill.
Reports acceptance rate, "valid" rate (accepted and nothing found outside the fill) and filled area.
"""
import glob
import json
import pickle
import re
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
root = Path(sys.argv[1])
device = sys.argv[2] if len(sys.argv) > 2 else 'cuda'
out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else root / 'metamorphic_eval' / 'occlusion_benchmark'
out_dir.mkdir(parents=True, exist_ok=True)

from PIL import Image  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: E402

from image_transforms.masking import GroundedSAM, occlude_objects  # noqa: E402

IMAGES = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/gqa/images')
image_of = {}
for line in open(root / 'dataset.jsonl'):
    r = json.loads(line)
    image_of[str(r['questionId'])] = IMAGES / r['image_path']

cases, seen = [], set()
for f in sorted(glob.glob(str(root / 'metamorphic_eval/chunk_*/run/data.p'))):
    for d in pickle.load(open(f, 'rb')):
        for m in d.get('unit_test_metadata', []):
            if m.get('transformation') == 'occlude_bbox':
                key = (str(d['questionId']), m['parameters']['item'])
                if key not in seen:
                    seen.add(key); cases.append(dict(qid=key[0], item=key[1], image=m['original_image_path'], was='kept'))
pat = re.compile(r"question (\S+): skipping test '(.+?), occluded bounding box': .*?: (.*)")
for f in sorted(glob.glob(str(root / 'metamorphic_eval/chunk_*/worker.log'))):
    for line in open(f, errors='replace'):
        m = pat.search(line)
        if m and 'No object matched' not in m.group(3):
            key = (m.group(1), m.group(2))
            if key not in seen and key[0] in image_of:
                seen.add(key); cases.append(dict(qid=key[0], item=key[1], image=str(image_of[key[0]]),
                                                 was='area_cap' if 'over 60%' in m.group(3) else 'still_detected'))
print(len(cases), 'cases:', {k: sum(c['was'] == k for c in cases) for k in ('kept', 'area_cap', 'still_detected')}, flush=True)

base_id = 'IDEA-Research/grounding-dino-base'
backend = GroundedSAM(device=device, detector_id=base_id)
tiny = types.SimpleNamespace(
    detector_processor=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'), device=device,
    text_threshold=.25, box_threshold=.35,
    detector=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to(device).eval())


def outside(boxes, changed):
    n = 0
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
        if changed[y0:y1, x0:x1].sum() / max(1, (x1 - x0) * (y1 - y0)) < .5:
            n += 1
    return n


VARIANTS = {
    'A_box, cap .6 (old default)': dict(max_area=.6),
    'A2_box, no cap (new default)': dict(),
    'B_box cap .85': dict(max_area=.85),
    'C_mask only, cap .6': dict(shape='mask', max_area=.6),
    'D_box -> mask fallback, cap .6': dict(fallback='mask', max_area=.6),
    'E_box -> mask, cap .85, 3 rounds': dict(fallback='mask', max_area=.85, max_rounds=3),
    'F_mask -> box, cap .85, 3 rounds': dict(shape='mask', fallback='box', max_area=.85, max_rounds=3),
}
rows = []
for c in cases:
    original = Image.open(c['image']).convert('RGB')
    for name, opts in VARIANTS.items():
        row = dict(c, variant=name)
        try:
            out = occlude_objects(original, c['item'], segmenter=backend, **opts)
        except ValueError as error:
            row.update(status='rejected', reason=str(error)[:70])
        else:
            changed = np.abs(np.asarray(original, dtype=np.int16) - np.asarray(out, dtype=np.int16)).sum(axis=2) > 0
            row.update(status='accepted', area=round(float(changed.mean()), 3))
            for thr in (.35, .25):
                tiny.box_threshold = thr
                row[f'out@{thr}'] = outside(GroundedSAM.detect_boxes(tiny, out, c['item']), changed)
            if name.startswith(('D', 'E')) and c['was'] != 'kept':
                slug = re.sub(r'\W+', '_', c['item'])
                out.save(out_dir / f"{c['qid']}_{slug}_{name[0]}.jpg")
        rows.append(row)
    print(c['qid'], c['item'], c['was'], [r['status'][0] + str(r.get('area', '')) for r in rows[-len(VARIANTS):]], flush=True)
(out_dir / 'benchmark.json').write_text(json.dumps(rows, indent=1))

lines = [f'# Occlusion benchmark ({len(cases)} attempted tests)\n',
         '| variant | accepted | valid @.35 | valid @.25 | mean area | accepted/valid on previously area-capped | on previously still-detected |', '|---|---|---|---|---|---|---|']
for name in VARIANTS:
    rs = [r for r in rows if r['variant'] == name]
    acc = [r for r in rs if r['status'] == 'accepted']
    v35 = [r for r in acc if r['out@0.35'] == 0]
    v25 = [r for r in acc if r['out@0.25'] == 0]
    def sub(kind):
        k = [r for r in rs if r['was'] == kind]
        return f"{sum(r['status'] == 'accepted' for r in k)}/{len(k)} acc, {sum(r['status'] == 'accepted' and r['out@0.25'] == 0 for r in k)} valid"
    lines.append(f"| {name} | {len(acc)}/{len(rs)} ({len(acc) / len(rs):.0%}) | {len(v35)} ({len(v35) / len(rs):.0%}) | "
                 f"{len(v25)} ({len(v25) / len(rs):.0%}) | {np.mean([r['area'] for r in acc]):.2f} | {sub('area_cap')} | {sub('still_detected')} |")
(out_dir / 'benchmark.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
